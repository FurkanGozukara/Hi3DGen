"""Heavy CPU mesh work (Open3D decimation + xatlas UV unwrapping) in a child process.

Both libraries keep Python's GIL for their whole run, about 70 s together for a typical result. In the app process that
froze everything for that long: no progress updates, no way to cancel. In a child process the app stays responsive, a
cancel stops the worker at once, and several results can be unwrapped while the GPU already works on the next image.

The worker (logic/mesh_worker.py) runs the same functions as the in-process path (logic/mesh_ops.py) on the same data,
so the meshes are identical. Set HI3DGEN_MESH_SUBPROCESS=0 to run in-process again.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
STAGE_MARK = "@@STAGE "
PREVIEW_MARK = "@@PREVIEW"

_live = set()
_live_lock = threading.Lock()


class MeshJobCancelled(Exception):
    """The job was cancelled while it ran."""


class MeshJobError(RuntimeError):
    """The mesh worker failed."""


@dataclass
class MeshJobResult:
    mesh: trimesh.Trimesh
    original_faces: int
    simplified_faces: int
    simplify_s: float
    unwrap_s: float


def subprocess_enabled() -> bool:
    return os.environ.get("HI3DGEN_MESH_SUBPROCESS", "1").strip().lower() not in ("0", "false", "no", "off")


def _kill_live_workers():
    with _live_lock:
        procs = list(_live)
    for proc in procs:
        try:
            proc.kill()
        except OSError:
            pass


atexit.register(_kill_live_workers)


def _remove_folder(folder, attempts: int = 25):
    """rmtree that waits for Windows to release files of a worker that has just been killed."""
    for attempt in range(attempts):
        shutil.rmtree(folder, ignore_errors=True)
        if not Path(folder).exists():
            return
        time.sleep(0.2)
    print(f"Mesh job: could not remove the temporary folder {folder}")


def cleanup_stale_jobs(tmp_root, max_age_s: float = 3600.0):
    """Remove work folders that a crashed run left behind (only our own `mesh_job_*` folders, older than an hour)."""
    try:
        now = time.time()
        for folder in Path(tmp_root).glob("mesh_job_*"):
            if folder.is_dir() and now - folder.stat().st_mtime > max_age_s:
                _remove_folder(folder, attempts=2)
    except OSError as exc:
        print(f"Mesh job: could not clean up old work folders: {exc}")


def _forward_output(pipe, on_stage, on_preview):
    """Print the worker's console output as it arrives and turn its marker lines into callbacks."""
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith(STAGE_MARK):
            callback, arguments = on_stage, (line[len(STAGE_MARK):].strip(),)
        elif line.startswith(PREVIEW_MARK):
            callback, arguments = on_preview, ()
        else:
            print(line, flush=True)
            continue
        if callback is not None:
            try:
                callback(*arguments)
            except Exception as exc:
                print(f"Mesh job: callback failed: {exc}")


def _run_in_process(vertices, faces, poly_count_pcnt, xatlas_options, cancel_check, on_stage, unwrap=True):
    from .mesh_ops import simplify_mesh_open3d, unwrap_mesh_with_xatlas  # imports Open3D: only needed in this process here

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    original_faces = int(len(mesh.faces))
    if on_stage:
        on_stage("simplify")
    started = time.time()
    simplified = simplify_mesh_open3d(mesh, poly_count_pcnt)
    simplify_s = time.time() - started
    if cancel_check and cancel_check():
        raise MeshJobCancelled()
    if not unwrap:
        return MeshJobResult(simplified, original_faces, int(len(simplified.faces)), simplify_s, 0.0)
    if on_stage:
        on_stage("unwrap")
    started = time.time()
    unwrapped = unwrap_mesh_with_xatlas(simplified, **xatlas_options)
    return MeshJobResult(unwrapped, original_faces, int(len(simplified.faces)), simplify_s, time.time() - started)


def run_mesh_job(vertices, faces, poly_count_pcnt: float, xatlas_max_cost: float = 8.0, xatlas_normal_seam_weight: float = 1.0,
                 xatlas_resolution: int = 1024, xatlas_padding: int = 2,
                 cancel_check: Optional[Callable[[], bool]] = None, on_stage: Optional[Callable[[str], None]] = None,
                 on_preview: Optional[Callable[[str], None]] = None, preview_path: Optional[str] = None,
                 tmp_root: Optional[str] = None, unwrap: bool = True) -> MeshJobResult:
    """Decimate the mesh to `poly_count_pcnt` of its faces and UV-unwrap it, like the app always did.

    `cancel_check()` is polled while the worker runs (True kills it and raises MeshJobCancelled); `on_stage(name)` is called
    with "simplify" and "unwrap" when those steps start. With `preview_path`, the decimated mesh is written there as a GLB
    before the long UV step and `on_preview(preview_path)` is called (not in the in-process fallback). With
    `unwrap=False` only the decimation runs and the mesh has no UVs.
    """
    xatlas_options = dict(max_cost_param=xatlas_max_cost, normal_seam_weight_param=xatlas_normal_seam_weight,
                          resolution_param=xatlas_resolution, padding_param=xatlas_padding)
    vertices = np.ascontiguousarray(vertices)
    faces = np.ascontiguousarray(faces)
    if not subprocess_enabled():
        return _run_in_process(vertices, faces, poly_count_pcnt, xatlas_options, cancel_check, on_stage, unwrap)

    workdir = Path(tempfile.mkdtemp(prefix="mesh_job_", dir=tmp_root))
    proc = None
    try:
        input_path, output_path, job_path = workdir / "input.npz", workdir / "output.npz", workdir / "job.json"
        np.savez(input_path, vertices=vertices, faces=faces)
        job_path.write_text(json.dumps({
            "input": str(input_path), "output": str(output_path), "parent_pid": os.getpid(),
            "poly_count_pcnt": poly_count_pcnt, "xatlas_max_cost": xatlas_max_cost,
            "xatlas_normal_seam_weight": xatlas_normal_seam_weight, "xatlas_resolution": xatlas_resolution,
            "xatlas_padding": xatlas_padding, "unwrap": bool(unwrap),
            "preview": str(workdir / "preview.glb") if (preview_path and on_preview) else None,
        }), encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        try:
            proc = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "logic.mesh_worker", str(job_path)], cwd=str(ROOT), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as exc:
            print(f"Mesh job: could not start the worker process ({exc}); running in this process instead.")
            return _run_in_process(vertices, faces, poly_count_pcnt, xatlas_options, cancel_check, on_stage, unwrap)
        with _live_lock:
            _live.add(proc)

        def preview_ready():
            try:
                shutil.copyfile(workdir / "preview.glb", preview_path)  # the work folder is removed when the job ends
            except OSError as exc:
                print(f"Mesh job: could not keep the preview ({exc})")
                return
            on_preview(preview_path)

        reader = threading.Thread(target=_forward_output, daemon=True, name="mesh-worker-output",
                                  args=(proc.stdout, on_stage, preview_ready if (preview_path and on_preview) else None))
        reader.start()
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel_check is not None and cancel_check():
                    proc.kill()
                    proc.wait()
                    raise MeshJobCancelled()
        reader.join(timeout=5)
        if proc.returncode != 0:
            raise MeshJobError(f"The mesh worker stopped with exit code {proc.returncode}; its output is in the console above.")
        with np.load(output_path) as data:
            vertices_out, faces_out = data["vertices"], data["faces"]
            uv_out = data["uv"] if "uv" in data.files else None
        stats = json.loads(Path(str(output_path) + ".json").read_text(encoding="utf-8"))
        # Same construction as the end of unwrap_mesh_with_xatlas
        mesh = trimesh.Trimesh(vertices=vertices_out, faces=faces_out, process=False)
        if uv_out is not None:
            material = trimesh.visual.material.PBRMaterial(name="defaultXatlasMat")
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv_out, material=material)
        return MeshJobResult(mesh, stats["original_faces"], stats["simplified_faces"], stats["simplify_s"], stats["unwrap_s"])
    finally:
        if proc is not None:
            with _live_lock:
                _live.discard(proc)
            if proc.poll() is None:
                proc.kill()
        _remove_folder(workdir)
