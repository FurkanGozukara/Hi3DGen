"""GPU memory policy for the big models: Hi3DGen pipeline, StableNormal predictor and BiRefNet.

Modes (environment HI3DGEN_MODEL_MEMORY, or the Performance setting of the app):

  auto     keep every model on the GPU between generations when the card has enough memory (>= 20 GB), otherwise
           offload to the CPU after each stage, which is how the app always behaved
  keep     always keep the models on the GPU (fastest)
  offload  always move the models to the CPU after each stage (smallest GPU footprint)

The StableNormal predictor is created once (on the CPU) and then only moved, instead of being rebuilt from disk for every
generation. Moving models never changes their weights, so results are identical in every mode.
"""
import contextlib
import gc
import os
import threading
from pathlib import Path

import torch

MODES = ("auto", "keep", "offload")
RESIDENT_MIN_VRAM_GB = 20.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STABLENORMAL_DIR = PROJECT_ROOT / "third_party" / "StableNormal"


def _free_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelMemory:
    def __init__(self, hi3dgen_pipeline, weights_dir, mode=None):
        self.pipeline = hi3dgen_pipeline
        self.weights_dir = Path(weights_dir)
        self._predictor = None
        self._lock = threading.RLock()
        self._out_of_memory = False  # set when 'auto' ran out of GPU memory: it then offloads for the rest of the session
        self.mode = "auto"
        self.set_mode(mode or os.environ.get("HI3DGEN_MODEL_MEMORY", "auto"))

    # ---- policy --------------------------------------------------------------------------------
    def set_mode(self, mode):
        mode = str(mode or "auto").strip().lower()
        if mode not in MODES:
            print(f"Model memory: unknown mode '{mode}', using 'auto'")
            mode = "auto"
        self.mode = mode
        self._out_of_memory = False
        self._apply_policy()
        return self.mode

    @property
    def has_cuda(self):
        return torch.cuda.is_available()

    @property
    def device(self):
        return torch.device("cuda") if self.has_cuda else torch.device("cpu")

    def total_vram_gb(self):
        if not self.has_cuda:
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    @property
    def resident(self):
        """True when the models stay on the GPU between generations."""
        if not self.has_cuda:
            return False
        if self.mode == "keep":
            return True
        if self.mode == "offload":
            return False
        return self.total_vram_gb() >= RESIDENT_MIN_VRAM_GB and not self._out_of_memory

    def note_out_of_memory(self):
        """The GPU ran out of memory: give everything back, and in 'auto' stop keeping the models on the GPU (another
        program probably uses it too). 'keep' stays as the user chose it."""
        if self.mode == "auto" and not self._out_of_memory:
            self._out_of_memory = True
            print("Model memory: the GPU ran out of memory, the models are moved to the CPU after each stage from now on "
                  "(Performance -> Model memory: keep brings the old behaviour back).")
        if self.pipeline is not None:
            self.pipeline.birefnet_keep_on_gpu = self.resident
        self.free_gpu_memory(quiet=True)

    def _apply_policy(self):
        if self.pipeline is not None:
            self.pipeline.birefnet_keep_on_gpu = self.resident
        if not self.resident:
            self.free_gpu_memory(quiet=True)

    # ---- StableNormal predictor ----------------------------------------------------------------
    def normal_predictor(self):
        """The StableNormal predictor, created once on the CPU."""
        with self._lock:
            if self._predictor is None:
                yoso_dir = self.weights_dir / "models--Stable-X--yoso-normal-v1-8-1"
                if not yoso_dir.exists():
                    raise FileNotFoundError(f"YOSO model not found at {yoso_dir}. Please ensure models are downloaded.")
                if not (STABLENORMAL_DIR / "hubconf.py").exists():
                    raise FileNotFoundError(f"StableNormal code not found at {STABLENORMAL_DIR}")
                print("Normal Prediction: Loading StableNormal model (once, kept in memory)...")
                self._predictor = torch.hub.load(
                    str(STABLENORMAL_DIR), "StableNormal_turbo", source="local", trust_repo=True,
                    yoso_version=str(yoso_dir.resolve()), device="cpu",
                )
            return self._predictor

    @staticmethod
    def _move_predictor(predictor, device):
        pipe = getattr(predictor, "model", None)
        if pipe is None:
            return
        try:
            pipe.to(device, silence_dtype_warnings=True)
        except TypeError:
            pipe.to(device)
        handler = predictor.segmentation_handler
        handler.device = str(device)
        # the background model of the handler (only created for images without alpha) must follow, or it would stay on the GPU
        if getattr(handler, "birefnet_model", None) is not None:
            handler.birefnet_model.to(device)

    @contextlib.contextmanager
    def use_normal_predictor(self):
        """Yield the predictor on the working device; park it on the CPU afterwards unless the models are resident."""
        predictor = self.normal_predictor()
        self._move_predictor(predictor, self.device)
        try:
            yield predictor
        finally:
            if not self.resident:
                self._move_predictor(predictor, "cpu")
                _free_cuda_cache()

    # ---- Hi3DGen pipeline ----------------------------------------------------------------------
    @contextlib.contextmanager
    def use_pipeline(self):
        """Yield the Hi3DGen pipeline on the working device; park it on the CPU afterwards unless resident."""
        moved = False
        if self.has_cuda:
            self.pipeline.cuda()
            moved = True
        try:
            yield self.pipeline
        finally:
            if moved and not self.resident:
                self.park_pipeline()

    def park_pipeline(self):
        if self.pipeline is None:
            return
        try:
            if self.has_cuda:
                torch.cuda.synchronize()
            self.pipeline.cpu()
        except Exception as exc:  # a CUDA failure must never stop the app from cleaning up
            print(f"Model memory: could not move the pipeline to the CPU: {exc}")
        finally:
            _free_cuda_cache()

    # ---- explicit control ----------------------------------------------------------------------
    @staticmethod
    def release_cached_memory():
        """Give unused cached GPU memory back (models stay where they are)."""
        _free_cuda_cache()

    def free_gpu_memory(self, quiet=False):
        """Move every model to the CPU and release cached GPU memory."""
        with self._lock:
            if self._predictor is not None:
                self._move_predictor(self._predictor, "cpu")
            if self.pipeline is not None:
                self.park_pipeline()
                park_birefnet = getattr(self.pipeline, "park_birefnet", None)
                if park_birefnet is not None:
                    park_birefnet()
        _free_cuda_cache()
        message = "GPU memory released: all models are on the CPU." if self.has_cuda else "No CUDA GPU in use."
        if not quiet:
            print(message)
        return message

    def status(self):
        if not self.has_cuda:
            return "CPU only (no CUDA GPU)"
        free, total = torch.cuda.mem_get_info()
        where = "on the GPU between runs" if self.resident else "offloaded to the CPU after each stage"
        return (f"Model memory: {self.mode} ({where}) | GPU {torch.cuda.get_device_name(0)} "
                f"| {free / 1024 ** 3:.1f} GB free of {total / 1024 ** 3:.1f} GB")
