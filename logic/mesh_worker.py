"""Child process of logic/mesh_jobs.py: Open3D decimation and xatlas UV unwrapping of one mesh.

    python -m logic.mesh_worker <job.json>

The job file names the input arrays (.npz with vertices and faces), the output file and the parameters. The worker runs the
very same functions as the in-process path (logic/mesh_ops.py), writes the unwrapped mesh to the output file and prints
`@@STAGE <name>` (and `@@PREVIEW`) marker lines that the parent turns into progress. Everything else it prints is console
output.
"""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _exit_with_parent(parent_pid):
    """Stop when the app that started this worker has gone (crash, kill, closed console)."""
    try:
        import psutil
    except ImportError:
        return

    def watch():
        while True:
            time.sleep(2.0)
            if not psutil.pid_exists(parent_pid):
                print("Mesh worker: the app is gone, stopping.", flush=True)
                import os
                os._exit(3)

    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    if job.get("parent_pid"):
        _exit_with_parent(int(job["parent_pid"]))

    import numpy as np
    import trimesh

    from logic.mesh_ops import simplify_mesh_open3d, unwrap_mesh_with_xatlas

    with np.load(job["input"]) as data:
        mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
    original_faces = int(len(mesh.faces))

    print("@@STAGE simplify", flush=True)
    started = time.time()
    simplified = simplify_mesh_open3d(mesh, job["poly_count_pcnt"])
    simplify_s = time.time() - started
    del mesh

    if job.get("preview"):
        # The decimated mesh is the final shape: show it while xatlas still works on the UVs (a fresh Trimesh has no cached
        # normals, so it is shaded like the final GLB)
        try:
            trimesh.Trimesh(vertices=simplified.vertices, faces=simplified.faces, process=False).export(job["preview"])
            print("@@PREVIEW", flush=True)
        except Exception as exc:
            print(f"Mesh worker: could not write the preview ({exc})", flush=True)

    if job.get("unwrap", True):
        print("@@STAGE unwrap", flush=True)
        started = time.time()
        unwrapped = unwrap_mesh_with_xatlas(
            simplified,
            max_cost_param=job["xatlas_max_cost"],
            normal_seam_weight_param=job["xatlas_normal_seam_weight"],
            resolution_param=job["xatlas_resolution"],
            padding_param=job["xatlas_padding"],
        )
        unwrap_s = time.time() - started
        np.savez(job["output"], vertices=np.asarray(unwrapped.vertices), faces=np.asarray(unwrapped.faces),
                 uv=np.asarray(unwrapped.visual.uv))
    else:
        unwrap_s = 0.0
        np.savez(job["output"], vertices=np.asarray(simplified.vertices), faces=np.asarray(simplified.faces))
    Path(job["output"] + ".json").write_text(json.dumps({
        "original_faces": original_faces,
        "simplified_faces": int(len(simplified.faces)),
        "simplify_s": simplify_s,
        "unwrap_s": unwrap_s,
    }), encoding="utf-8")
    print("@@STAGE done", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
