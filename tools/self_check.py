"""Hi3DGen environment self check.

    python tools/self_check.py            # everything
    python tools/self_check.py --quick    # skip the slower GPU kernels

It reports the versions that matter, whether the GPU kernels the app depends on work on THIS GPU
(attention in fp16/fp32, sparse convolution in every algorithm) and whether the mesh libraries behave.
Every GPU check runs in its own process, so a native crash is reported instead of ending the whole check.
Exit code 0 means nothing the app needs is broken.
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = []


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace", line_buffering=True)
            except (OSError, TypeError, ValueError):
                pass


def report(name, ok, detail="", critical=True):
    RESULTS.append((name, ok, critical))
    mark = "OK  " if ok else ("FAIL" if critical else "WARN")
    print(f"[{mark}] {name}" + (f"  -  {detail}" if detail else ""), flush=True)


def run_snippet(code, timeout=300, env_extra=None):
    """Run python code in a fresh interpreter; returns (returncode, stdout, stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT))
    return proc.returncode, proc.stdout, proc.stderr


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    return None


# ------------------------------------------------------------------------------------------------
def check_versions():
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info[:2] != (3, 12):
        report("Python version", False, f"Hi3DGen is tested on Python 3.12, found {platform.python_version()}", critical=False)
    packages = ["torch", "torchvision", "numpy", "scipy", "pillow", "scikit-image", "trimesh", "open3d", "xatlas", "diffusers",
                "transformers", "huggingface_hub", "safetensors", "einops", "gradio", "xformers", "mslk", "flash_attn", "spconv", "cumm"]
    from importlib import metadata

    line = []
    for name in packages:
        try:
            line.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            report(f"package {name}", False, "not installed")
    for i in range(0, len(line), 4):
        print("   " + " | ".join(line[i:i + 4]))


def check_gpu():
    code = (
        "import torch, json\n"
        "info = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}\n"
        "if info['available']:\n"
        "    p = torch.cuda.get_device_properties(0)\n"
        "    info.update(name=p.name, capability=list(torch.cuda.get_device_capability(0)), vram_gb=round(p.total_memory / 1024**3, 1),"
        " arch_list=torch.cuda.get_arch_list())\n"
        "print(json.dumps(info))\n"
    )
    rc, out, err = run_snippet(code, timeout=180)
    info = last_json(out) if rc == 0 else None
    if not info:
        report("CUDA / torch", False, (err or out).strip().splitlines()[-1] if (err or out).strip() else f"exit code {rc}")
        return None
    if not info["available"]:
        report("CUDA GPU", False, "torch cannot see a CUDA GPU (driver too old for this CUDA 13 build?)")
        return None
    cap = tuple(info["capability"])
    sm = f"sm_{cap[0]}{cap[1]}"
    covered = any(a.startswith(sm) or a == f"compute_{cap[0]}{cap[1]}" for a in info["arch_list"])
    report("CUDA GPU", True, f"{info['name']}  {info['vram_gb']} GB  capability {cap[0]}.{cap[1]}  torch {info['torch']} (CUDA {info['cuda']})")
    report("torch has kernels for this GPU", covered, f"{sm} in {','.join(info['arch_list'])}" if not covered else sm)
    return info


ATTENTION_CODE = r"""
import json, torch
res = {}
dev = "cuda"
def qkv(dtype, length=1024):
    return [torch.randn(1, length, 16, 64, device=dev, dtype=dtype) for _ in range(3)]
try:
    import xformers.ops as xops
except Exception as exc:
    print(json.dumps({"import xformers.ops": ["fail", str(exc).splitlines()[0][:200]]})); raise SystemExit(0)
def attempt(name, fn):
    try:
        out = fn(); torch.cuda.synchronize()
        res[name] = ["ok", float(out.float().abs().mean())]
    except Exception as exc:
        res[name] = ["fail", (str(exc).strip().splitlines() or [type(exc).__name__])[0][:200]]
q, k, v = qkv(torch.float16)
attempt("xformers fp16", lambda: xops.memory_efficient_attention(q, k, v))
mask = xops.fmha.BlockDiagonalMask.from_seqlens([512, 512])
attempt("xformers fp16 block-diagonal", lambda: xops.memory_efficient_attention(q.reshape(1, -1, 16, 64), k.reshape(1, -1, 16, 64), v.reshape(1, -1, 16, 64), mask))
q32, k32, v32 = qkv(torch.float32, 1374)
attempt("xformers fp32 (DINOv2)", lambda: xops.memory_efficient_attention(q32, k32, v32))
try:
    import flash_attn
    attempt("flash_attn fp16", lambda: flash_attn.flash_attn_func(q, k, v))
except Exception as exc:
    res["flash_attn fp16"] = ["fail", str(exc).splitlines()[0][:200]]
import torch.nn.functional as F
attempt("torch SDPA fp32", lambda: F.scaled_dot_product_attention(q32.transpose(1, 2), k32.transpose(1, 2), v32.transpose(1, 2)))
print(json.dumps(res))
"""


def check_attention():
    rc, out, err = run_snippet(ATTENTION_CODE, timeout=300)
    res = last_json(out) if rc == 0 else None
    if res is None:
        report("attention kernels", False, (err or out).strip().splitlines()[-1] if (err or out).strip() else f"process exit code {rc}")
        return {}
    for name, (state, detail) in res.items():
        needed = name != "xformers fp32 (DINOv2)"
        if state == "ok":
            report(f"attention: {name}", True)
        elif name == "xformers fp32 (DINOv2)":
            report(f"attention: {name}", False, "no xformers kernel on this GPU; the app switches DINOv2 to its built-in attention automatically - " + detail, critical=False)
        else:
            report(f"attention: {name}", False, detail, critical=needed)
    return res


SPCONV_CODE = r"""
import json, os, time, torch
import spconv.pytorch as spconv
algo_name = os.environ.get("HI3DGEN_CHECK_ALGO", "native")
algo = {"native": spconv.ConvAlgo.Native, "implicit_gemm": spconv.ConvAlgo.MaskImplicitGemm, "auto": None}[algo_name]
torch.manual_seed(0)
coords = torch.unique(torch.randint(0, 64, (30000, 3), dtype=torch.int32), dim=0)
coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords], 1).cuda()
feats = torch.randn(len(coords), 32, device="cuda")
x = spconv.SparseConvTensor(feats, coords, [64, 64, 64], 1)
torch.manual_seed(1)
# the layers the model really uses: submanifold convs (3x3x3 pair + 1x1x1 skip) sharing one indice key, as in the mesh decoder
conv1 = spconv.SubMConv3d(32, 48, 3, bias=True, indice_key="res", algo=algo).cuda()
conv2 = spconv.SubMConv3d(48, 48, 3, bias=True, indice_key="res", algo=algo).cuda()
skip = spconv.SubMConv3d(32, 48, 1, bias=True, indice_key="res", algo=algo).cuda()
torch.cuda.synchronize(); t = time.time()
y = conv2(conv1(x)); s = skip(x)
out = y.features + s.features
torch.cuda.synchronize()
print(json.dumps({"algo": algo_name, "seconds": round(time.time() - t, 3), "checksum": round(float(out.double().sum()), 3), "voxels": [len(y.indices), len(s.indices)]}))
"""


def check_spconv(quick):
    algos = ["native"] if quick else ["native", "auto", "implicit_gemm"]
    sums = {}
    for algo in algos:
        try:
            rc, out, err = run_snippet(SPCONV_CODE, timeout=600, env_extra={"HI3DGEN_CHECK_ALGO": algo})
        except subprocess.TimeoutExpired:
            report(f"spconv sparse conv ({algo})", False, "timed out", critical=(algo == "native"))
            continue
        res = last_json(out) if rc == 0 else None
        if res:
            sums[algo] = res["checksum"]
            report(f"spconv sparse conv ({algo})", True, f"{res['seconds']} s, voxels {res['voxels']}, checksum {res['checksum']}", critical=(algo == "native"))
        else:
            tail = (err or out).strip().splitlines()
            report(f"spconv sparse conv ({algo})", False, tail[-1][:200] if tail else f"process crashed, exit code {rc}", critical=(algo == "native"))
    if len(sums) > 1:
        spread = max(sums.values()) - min(sums.values())
        print(f"   spconv algorithms agree to within {spread:.4f} (sum over all outputs)")


def check_mesh_libraries():
    code = r"""
import json, numpy as np, trimesh, open3d as o3d, xatlas, tempfile, os
from skimage import measure
res = {}
x, y, z = np.mgrid[-1:1:48j, -1:1:48j, -1:1:48j]
vol = (x**2 + y**2 + z**2 - 0.5).astype(np.float32)
verts, faces, normals, _ = measure.marching_cubes(vol, level=0.0, gradient_direction="ascent")
res["marching_cubes"] = [len(verts), len(faces)]
m = o3d.geometry.TriangleMesh()
m.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64)); m.triangles = o3d.utility.Vector3iVector(faces)
s = m.simplify_quadric_decimation(target_number_of_triangles=len(faces) // 2)
res["open3d_decimate"] = [len(s.vertices), len(s.triangles)]
tm = trimesh.Trimesh(np.asarray(s.vertices), np.asarray(s.triangles), process=True)
atlas = xatlas.Atlas()
atlas.add_mesh(tm.vertices.astype(np.float32), tm.faces.astype(np.uint32), np.ascontiguousarray(tm.vertex_normals, dtype=np.float32))
atlas.generate()
vmap, idx, uv = atlas.get_mesh(0)
res["xatlas"] = [int(atlas.width), int(atlas.height), len(uv)]
out = trimesh.Trimesh(tm.vertices[vmap], idx, process=False)
out.visual = trimesh.visual.TextureVisuals(uv=uv / np.array([atlas.width, atlas.height], dtype=np.float32), material=trimesh.visual.material.PBRMaterial(name="m"))
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "t.glb"); out.export(p)
    back = trimesh.load_mesh(p)
    r = {}
    for fmt in ("obj", "ply", "stl"):
        q = os.path.join(d, "t." + fmt); back.export(q, file_type=fmt); r[fmt] = len(trimesh.load_mesh(q).faces)
    res["trimesh_glb_roundtrip"] = [len(back.faces), r]
print(json.dumps(res))
"""
    rc, out, err = run_snippet(code, timeout=300)
    res = last_json(out) if rc == 0 else None
    if res:
        report("mesh libraries (marching cubes, Open3D decimation, xatlas, trimesh export)", True, json.dumps(res))
    else:
        tail = (err or out).strip().splitlines()
        report("mesh libraries", False, tail[-1][:240] if tail else f"exit code {rc}")


def check_model_stack():
    code = r"""
import json, importlib
res = {}
import diffusers, transformers
res["diffusers"] = diffusers.__version__; res["transformers"] = transformers.__version__
try:
    importlib.import_module("diffusers.models.controlnet"); res["legacy_controlnet_module"] = "present"
except ModuleNotFoundError:
    res["legacy_controlnet_module"] = "absent (the vendored StableNormal adds a compatibility alias)"
importlib.import_module("diffusers.models.controlnets.controlnet")
from hi3dgen.third_party.birefnet import load_birefnet
res["vendored_birefnet"] = "importable"
print(json.dumps(res))
"""
    rc, out, err = run_snippet(code, timeout=300)
    res = last_json(out) if rc == 0 else None
    if res:
        report("model libraries (diffusers / transformers / vendored BiRefNet)", True, json.dumps(res))
    else:
        tail = (err or out).strip().splitlines()
        report("model libraries", False, tail[-1][:240] if tail else f"exit code {rc}")


def check_weights():
    weights = ROOT / "weights"
    needed = {
        "Hi3DGen pipeline": weights / "models--Stable-X--trellis-normal-v0-1" / "pipeline.json",
        "StableNormal weights": weights / "models--Stable-X--yoso-normal-v1-8-1" / "model_index.json",
        "BiRefNet weights": weights / "models--ZhengPeng7--BiRefNet" / "model.safetensors",
        "DINOv2 code": weights / "github--facebookresearch--dinov2" / "hubconf.py",
        "DINOv2 weights": weights / "dinov2_vitl14_reg4_pretrain.pth",
    }
    for label, path in needed.items():
        report(f"weights: {label}", path.is_file(), "" if path.is_file() else f"missing {path.relative_to(ROOT)}  (run DownloadModels.py)", critical=False)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="Hi3DGen environment self check")
    parser.add_argument("--quick", action="store_true", help="skip the slower spconv algorithm comparison")
    args = parser.parse_args()

    print("=" * 78 + "\nHi3DGen self check\n" + "=" * 78)
    check_versions()
    print()
    gpu = check_gpu()
    if gpu and gpu["available"]:
        check_attention()
        check_spconv(args.quick)
    check_mesh_libraries()
    check_model_stack()
    check_weights()

    failed = [name for name, ok, critical in RESULTS if not ok and critical]
    warned = [name for name, ok, critical in RESULTS if not ok and not critical]
    print("\n" + "=" * 78)
    print(f"{len(RESULTS)} checks, {len(failed)} failed, {len(warned)} warnings")
    for name in failed:
        print(f"   FAILED: {name}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
