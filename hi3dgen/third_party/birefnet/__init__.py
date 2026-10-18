"""Inference-only vendored BiRefNet (MIT, https://github.com/ZhengPeng7/BiRefNet).

Loads the published ``ZhengPeng7/BiRefNet`` checkpoint (``model.safetensors``) WITHOUT ``trust_remote_code`` and
without any dependency on transformers / timm / kornia (see model.py for the exact list of local modifications).

    from hi3dgen.third_party.birefnet import load_birefnet
    model = load_birefnet("weights/models--ZhengPeng7--BiRefNet")   # fp32, eval(), on CPU
    model = model.to(device)
    mask_logits = model(x)[-1]                                      # x: (B, 3, 1024, 1024), ImageNet-normalised fp32

Behaviour that is deliberately identical to the old ``AutoModelForImageSegmentation.from_pretrained(...)`` path on
transformers 4.46.3: weights are stored as fp16 in the checkpoint and are up-cast (exactly) to fp32; the model is
returned in eval mode; ``model(x)`` returns a list whose last element is the final (full-resolution) logit map.
"""
import contextlib
import glob
import os

import torch

from .model import BiRefNet

__all__ = ["BiRefNet", "load_birefnet", "find_birefnet_weights", "ieee_fp32_convs", "WEIGHTS_FILENAME"]

WEIGHTS_FILENAME = "model.safetensors"


@contextlib.contextmanager
def ieee_fp32_convs():
    """Run fp32 convolutions without TF32 for the duration of the block.

    PyTorch enables TF32 for cuDNN convolutions by default, which is slightly less precise than the fp32 the CPU
    uses. Switching it off around the BiRefNet forward keeps masks computed on the GPU as close as possible to the
    CPU reference. The previous setting is restored afterwards.
    """
    cudnn = torch.backends.cudnn
    conv = getattr(cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):  # PyTorch >= 2.9
        previous = conv.fp32_precision
        conv.fp32_precision = "ieee"
        try:
            yield
        finally:
            conv.fp32_precision = previous
    else:
        previous = cudnn.allow_tf32
        cudnn.allow_tf32 = False
        try:
            yield
        finally:
            cudnn.allow_tf32 = previous


def find_birefnet_weights(model_dir: str) -> str:
    """Return the path of ``model.safetensors`` inside ``model_dir``.

    ``model_dir`` is normally the plain folder written by ``snapshot_download(local_dir=...)``; a Hugging Face hub
    cache repo folder (``snapshots/<revision>/model.safetensors``) is accepted as well.
    """
    direct = os.path.join(model_dir, WEIGHTS_FILENAME)
    if os.path.isfile(direct):
        return direct
    cached = sorted(
        glob.glob(os.path.join(model_dir, "snapshots", "*", WEIGHTS_FILENAME)),
        key=os.path.getmtime,
    )
    if cached:
        return cached[-1]
    raise FileNotFoundError(
        f"BiRefNet weights '{WEIGHTS_FILENAME}' not found in {model_dir}. "
        f"Please run DownloadModels.py to download required models."
    )


def load_birefnet(model_dir: str, device=None) -> BiRefNet:
    """Build BiRefNet (fp32), load ``model.safetensors`` with ``strict=True`` and return it in eval mode."""
    from safetensors.torch import load_file

    weights_path = find_birefnet_weights(model_dir)

    # Building the module runs a random init (overwritten right below) that consumes the global CPU RNG. Fork the
    # RNG so that loading BiRefNet never changes the random stream the 3D pipeline seeds/relies on.
    with torch.random.fork_rng(devices=[]):
        model = BiRefNet(bb_pretrained=False)

    # float tensors are stored as fp16, buffers (relative_position_index, num_batches_tracked) as int64.
    state_dict = load_file(weights_path, device="cpu")
    # strict=True: the checkpoint contains exactly the 754 tensors of this architecture. load_state_dict() copies
    # fp16 -> fp32 (exact, same values the transformers 4.x loader produced) and keeps the int64 buffers as they are.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model
