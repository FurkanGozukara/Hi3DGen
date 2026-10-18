"""Headless end-to-end check and regression harness for the Hi3DGen pipeline.

    python tools/golden_check.py run --out runs/baseline            # run the real pipeline on example images
    python tools/golden_check.py run --out runs/new --count 3 --seed 12345
    python tools/golden_check.py compare runs/baseline runs/new     # normal maps + meshes must match

`run` uses exactly the code path of the app (preprocess upload -> ProcessingCore.process_single_image), with a fixed
seed, and writes per image: <stem>/<stem>.glb|obj|ply|stl, <stem>_normal.png and summary.json (timings, face counts,
peak GPU memory). `compare` reports normal map PSNR and mesh differences (counts, bounding box, area, chamfer distance).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SPCONV_ALGO", "native")


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace", line_buffering=True)
            except (OSError, TypeError, ValueError):
                pass


def pick_images(explicit, count):
    if explicit:
        return [Path(p) for p in explicit]
    folder = ROOT / "assets" / "example_image"
    files = sorted(folder.glob("*.png"), key=lambda p: (len(p.stem), p.stem))
    if count >= len(files):
        return files
    step = len(files) / count
    return [files[int(i * step)] for i in range(count)]


def cmd_run(args):
    import torch
    from PIL import Image

    from hi3dgen.pipelines import Hi3DGenPipeline
    from logic.processing_core import ProcessingCore, ProcessingParameters

    weights = ROOT / "weights"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipeline = Hi3DGenPipeline.from_pretrained(str(weights / "models--Stable-X--trellis-normal-v0-1"))
    pipeline.cpu()
    print(f"pipeline loaded in {time.time() - t0:.1f}s")
    core = ProcessingCore(pipeline, str(weights), str(ROOT / "tmp"), 2 ** 31 - 1)
    if args.memory:
        core.memory.set_mode(args.memory)
    print(core.memory.status())

    params = ProcessingParameters(
        seed=args.seed, ss_sampling_steps=args.ss_steps, slat_sampling_steps=args.slat_steps,
        auto_save_obj=True, auto_save_glb=True, auto_save_ply=False, auto_save_stl=False,
    )
    summary = {"seed": args.seed, "ss_steps": args.ss_steps, "slat_steps": args.slat_steps, "images": {}}
    for repeat in range(args.repeat):
        for path in pick_images(args.images, args.count):
            stem = path.stem
            image = Image.open(path)
            t1 = time.time()
            prepared = pipeline.preprocess_image(image, resolution=1024)  # what the app does on upload
            preprocess_time = time.time() - t1
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def progress(stage, details=""):
                pass

            result = core.process_single_image(prepared, params, progress, custom_output_folder=str(out_dir), custom_filename_base=stem)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
            entry = {
                "success": result.success, "error": result.error_message, "preprocess_s": round(preprocess_time, 2),
                "normal_s": round(result.normal_prediction_time, 2), "pipeline_s": round(result.pipeline_time, 2),
                "mesh_s": round(result.mesh_processing_time, 2), "uv_s": round(result.uv_unwrapping_time, 2),
                "export_s": round(result.export_time, 2), "total_s": round(result.total_time, 2),
                "faces_raw": result.original_face_count, "faces_simplified": result.simplified_face_count,
                "faces_final": result.final_face_count, "vertices_final": result.final_vertex_count, "peak_vram_gb": round(peak, 2),
            }
            key = stem if repeat == 0 else f"{stem}#run{repeat + 1}"
            summary["images"][key] = entry
            print(f"[{key}] {json.dumps(entry)}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed = [k for k, v in summary["images"].items() if not v["success"]]
    print(f"done: {len(summary['images']) - len(failed)} ok, {len(failed)} failed -> {out_dir}")
    return 1 if failed else 0


def load_mesh_stats(path):
    import numpy as np
    import trimesh

    mesh = trimesh.load_mesh(str(path))
    points = mesh.sample(20000, seed=0) if hasattr(mesh, "sample") else mesh.vertices
    return {
        "vertices": len(mesh.vertices), "faces": len(mesh.faces), "bounds": np.asarray(mesh.bounds),
        "area": float(mesh.area), "points": np.asarray(points), "watertight": bool(mesh.is_watertight),
    }


def cmd_compare(args):
    import numpy as np
    from PIL import Image
    from scipy.spatial import cKDTree

    dir_a, dir_b = Path(args.a), Path(args.b)
    stems = sorted({p.name for p in dir_a.iterdir() if p.is_dir()} & {p.name for p in dir_b.iterdir() if p.is_dir()})
    if not stems:
        print("no common result folders")
        return 1
    worst_failed = False
    print(f"{'image':26s} {'normal PSNR':>11s} {'faces A':>9s} {'faces B':>9s} {'dBBox':>9s} {'dArea%':>7s} {'chamfer':>9s}")
    for stem in stems:
        na, nb = dir_a / stem / f"{stem}_normal.png", dir_b / stem / f"{stem}_normal.png"
        psnr = float("nan")
        if na.exists() and nb.exists():
            a = np.asarray(Image.open(na).convert("RGB"), dtype=np.float64)
            b = np.asarray(Image.open(nb).convert("RGB"), dtype=np.float64)
            if a.shape == b.shape:
                mse = float(((a - b) ** 2).mean())
                psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        ga, gb = dir_a / stem / f"{stem}.glb", dir_b / stem / f"{stem}.glb"
        if not (ga.exists() and gb.exists()):
            print(f"{stem:26s} missing glb")
            worst_failed = True
            continue
        sa, sb = load_mesh_stats(ga), load_mesh_stats(gb)
        dbbox = float(np.abs(sa["bounds"] - sb["bounds"]).max())
        darea = abs(sa["area"] - sb["area"]) / max(sa["area"], 1e-12) * 100
        ta, tb = cKDTree(sa["points"]), cKDTree(sb["points"])
        chamfer = float(ta.query(sb["points"])[0].mean() + tb.query(sa["points"])[0].mean()) / 2
        scale = float(np.linalg.norm(sa["bounds"][1] - sa["bounds"][0]))
        ok = (np.isinf(psnr) or psnr >= args.min_psnr) and chamfer <= args.max_chamfer * scale and darea <= args.max_area_pct
        worst_failed = worst_failed or not ok
        print(f"{stem:26s} {psnr:11.2f} {sa['faces']:9d} {sb['faces']:9d} {dbbox:9.5f} {darea:7.3f} {chamfer:9.6f} {'OK' if ok else 'DIFFERENT'}")
    return 1 if worst_failed else 0


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline on example images")
    run.add_argument("--out", required=True)
    run.add_argument("--images", nargs="*", help="image files (default: evenly spaced examples)")
    run.add_argument("--count", type=int, default=3)
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--ss-steps", type=int, default=50)
    run.add_argument("--slat-steps", type=int, default=6)
    run.add_argument("--repeat", type=int, default=1, help="repeat the whole set (second run shows the warm timings)")
    run.add_argument("--memory", choices=["auto", "keep", "offload"], help="model memory mode")
    run.set_defaults(func=cmd_run)
    cmp_ = sub.add_parser("compare", help="compare two result folders")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--min-psnr", type=float, default=45.0)
    cmp_.add_argument("--max-chamfer", type=float, default=1e-3, help="relative to the bounding box diagonal")
    cmp_.add_argument("--max-area-pct", type=float, default=0.5)
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
