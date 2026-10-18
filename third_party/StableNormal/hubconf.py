# Vendored from https://github.com/hugoycj/StableNormal @ 1b7203c988384cb62286a25865d7354be8fdbbf7 (Apache-2.0).
# Modified for Hi3DGen (python 3.12 / torch 2.13 / diffusers 0.41 / transformers 5.x):
#   * only the YOSO "turbo" entry point is kept (the two-stage StableNormal() needs network DINOv2 + other weights),
#   * diffusers.models.controlnet compatibility alias (module removed in diffusers 0.37.0, still imported by the
#     custom code shipped inside Stable-X/yoso-normal-v1-8-1/controlnet/controlnetvae.py),
#   * dtype= instead of the deprecated torch_dtype= on diffusers >= 0.40,
#   * yoso_version may also be a bare name resolved inside local_cache_dir (also as models--Stable-X--<name>),
#   * optional STABLENORMAL_LOAD_MODE=explicit: build the pipeline from its three components with the vendored
#     stablenormal/controlnetvae.py instead of diffusers' trust_remote_code path (default behaviour is unchanged).
#   * the BiRefNet background mask (only used for RGB input without alpha) is loaded from the local Hi3DGen weights
#     with the vendored hi3dgen.third_party.birefnet instead of a Hub id with trust_remote_code,
#   * the transformers imports of the unused "outdoor" mode are lazy.
# Model computation is unchanged.
from __future__ import annotations

dependencies = ["torch", "numpy", "diffusers", "PIL", "transformers"]

import enum
import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline

# <app root>/third_party/StableNormal/hubconf.py -> <app root>/weights/models--ZhengPeng7--BiRefNet
DEFAULT_BIREFNET_DIR = Path(__file__).resolve().parents[2] / "weights" / "models--ZhengPeng7--BiRefNet"


def _alias_legacy_diffusers_controlnet_module() -> None:
    """Make ``from diffusers.models.controlnet import ControlNetOutput`` work on diffusers >= 0.37.

    The weights repo ships ``controlnet/controlnetvae.py`` which is executed by diffusers (trust_remote_code) and
    imports ``ControlNetOutput`` from the old module path. diffusers 0.37.0 deleted that deprecated shim module;
    the class now lives in ``diffusers.models.controlnets.controlnet``. Registering the new module under the old name
    is enough because only ``ControlNetOutput``/``ControlNetModel`` are looked up there.
    """
    legacy = "diffusers.models.controlnet"
    if legacy in sys.modules:
        return
    try:
        importlib.import_module(legacy)  # still shipped by diffusers <= 0.36
    except ModuleNotFoundError:
        sys.modules[legacy] = importlib.import_module("diffusers.models.controlnets.controlnet")


def _dtype_kwargs(dtype: "torch.dtype") -> dict:
    """``dtype=`` is accepted by DiffusionPipeline.from_pretrained since diffusers 0.40 (``torch_dtype=`` then warns and
    is scheduled for removal in 1.0); older diffusers only understand ``torch_dtype=``."""
    import diffusers
    from packaging.version import Version

    if Version(diffusers.__version__).release >= (0, 40):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _resolve_yoso_dir(local_cache_dir: Optional[str], yoso_version: str) -> str:
    """Original behaviour is ``os.path.join(local_cache_dir or "Stable-X", yoso_version)`` (an absolute yoso_version
    wins over local_cache_dir). Additionally accept the layout written by DownloadModels.py
    (``<weights>/models--Stable-X--<yoso_version>``) when the plain join does not point at a model directory."""
    candidates = []
    if os.path.isabs(yoso_version) or os.path.isdir(yoso_version):
        candidates.append(yoso_version)
    if local_cache_dir:
        candidates.append(os.path.join(local_cache_dir, yoso_version))
        candidates.append(os.path.join(local_cache_dir, f"models--Stable-X--{yoso_version}"))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "model_index.json")):
            return cand
    return os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)


def _load_yoso_pipeline_explicit(path: str) -> "YOSONormalsPipeline":
    """Same object that ``YOSONormalsPipeline.from_pretrained(path, variant='fp16', t_start=0, safety_checker=None)``
    builds, but without diffusers' trust_remote_code machinery (no HF_MODULES_CACHE copy, no model_index.json custom
    class lookup): the three components named in model_index.json are loaded with their own from_pretrained and the
    custom ControlNetVAEModel comes from the vendored stablenormal/controlnetvae.py (class body identical to the
    one in the weights repo)."""
    from diffusers.models import AutoencoderKL, UNet2DConditionModel
    from stablenormal.controlnetvae import ControlNetVAEModel

    kw = _dtype_kwargs(torch.float16)
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", variant="fp16", **kw)
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", variant="fp16", **kw)
    controlnet = ControlNetVAEModel.from_pretrained(path, subfolder="controlnet", variant="fp16", **kw)
    return YOSONormalsPipeline(vae=vae, unet=unet, controlnet=controlnet, safety_checker=None, t_start=0)


class DataType(enum.Enum):
    INDOOR = "indoor"  # No masking
    OBJECT = "object"  # Mask background using BiRefNet or alpha channel
    OUTDOOR = "outdoor"  # Mask vegetation and sky using Mask2Former

class SegmentationHandler:
    def __init__(self, device: str = "cuda", birefnet_dir: Optional[str] = None):
        self.device = device
        self.birefnet_dir = str(birefnet_dir or os.environ.get("HI3DGEN_BIREFNET_DIR") or DEFAULT_BIREFNET_DIR)
        self.mask2former_processor = None
        self.mask2former_model = None
        self.birefnet_model = None

    def _lazy_load_mask2former(self):
        """Lazy loading of the Mask2Former model"""
        if self.mask2former_model is None:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            self.mask2former_processor = AutoImageProcessor.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic"
            )
            self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic"
            ).to(self.device)
            self.mask2former_model.eval()

    def _lazy_load_birefnet(self):
        """Lazy loading of BiRefNet from the local Hi3DGen weights (vendored model code, no Hub access)."""
        if self.birefnet_model is None:
            from hi3dgen.third_party.birefnet import load_birefnet

            self.birefnet_model = load_birefnet(self.birefnet_dir).to(self.device)
            self.birefnet_model.eval()

    def _get_birefnet_mask(self, image: Image.Image) -> np.ndarray:
        """Get object mask using BiRefNet"""
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to(self.device)

        # Prediction (fp32 convolutions without TF32, so GPU masks stay as close as possible to the CPU reference)
        from hi3dgen.third_party.birefnet import ieee_fp32_convs

        with torch.no_grad(), ieee_fp32_convs():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)
        
        return (mask_np > 128).astype(np.uint8)

    def _get_mask2former_mask(self, image: Image.Image) -> np.ndarray:
        """Get outdoor mask using Mask2Former"""
        inputs = self.mask2former_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.mask2former_model(**inputs)
            
        predicted_semantic_map = self.mask2former_processor.post_process_semantic_segmentation(
            outputs, 
            target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()
        
        # Mask vegetation (class 9) and sky (class 10)
        mask = ~np.isin(predicted_semantic_map, [9, 10])
        return mask.astype(np.uint8)

    def get_mask(self, image: Image.Image, data_type: DataType) -> Optional[np.ndarray]:
        """
        Get segmentation mask based on data type.
        
        Args:
            image: Input PIL Image
            data_type: Type of data processing required
            
        Returns:
            Optional numpy array mask where 1 indicates areas to keep
        """
        if data_type == DataType.INDOOR:
            return None
            
        if data_type == DataType.OBJECT:
            self._lazy_load_birefnet()
            return self._get_birefnet_mask(image)
        else:  # OUTDOOR
            self._lazy_load_mask2former()
            return self._get_mask2former_mask(image)

class Predictor:
    def __init__(self, model):
        self.model = model
        self.segmentation_handler = SegmentationHandler()

    def to(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.model.to(device, dtype)
        self.segmentation_handler.device = device
        return self
    
    def _apply_mask(self, 
                    prediction: np.ndarray, 
                    mask: Optional[np.ndarray]
                   ) -> np.ndarray:
        """Apply mask to normal map prediction if mask exists"""
        if mask is not None:
            prediction = prediction.copy()
            prediction[mask == 0] = 1
        return prediction
    
    def _process_rgba_image(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Process RGBA image by extracting alpha channel as mask and creating white background
        
        Args:
            img: RGBA PIL Image
            
        Returns:
            Tuple of (RGB image with white background, alpha mask)
        """
        # Split alpha channel
        rgb = img.convert('RGB')
        alpha = img.split()[-1]
        
        # Create white background image
        white_bg = Image.new('RGB', img.size, (255, 255, 255))
        
        # Composite the image onto white background
        composite = Image.composite(rgb, white_bg, alpha)
        
        # Convert alpha to numpy mask
        alpha_mask = (np.array(alpha) > 128).astype(np.uint8)
        
        return composite, alpha_mask
    
    @torch.no_grad()
    def __call__(
        self, 
        img: Image.Image, 
        resolution: int = 1024,
        match_input_resolution: bool = True,
        data_type: Union[DataType, str] = DataType.INDOOR,
        num_inference_steps: int = None
    ) -> Image.Image:
        """
        Generate normal map from input image.
        
        Args:
            img: Input PIL Image
            resolution: Target processing resolution
            match_input_resolution: Whether to match input image resolution
            data_type: Type of data (indoor/object/outdoor) affecting masking
            num_inference_steps: Optional number of inference steps
            
        Returns:
            PIL Image containing the normal map
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type.lower())
        
        # Handle RGBA images
        alpha_mask = None
        orig_size = img.size
        if img.mode == 'RGBA':
            img, alpha_mask = self._process_rgba_image(img)
            img = resize_image(img, resolution)
            alpha_mask = Image.fromarray(alpha_mask).resize(img.size, Image.Resampling.NEAREST)
            alpha_mask = np.array(alpha_mask)
            mask = alpha_mask
        else:
            # Regular RGB image processing
            img = resize_image(img, resolution)
            mask = self.segmentation_handler.get_mask(img, data_type) if data_type != DataType.INDOOR else None
        
        # Generate normal map
        kwargs = {}
        if num_inference_steps is not None:
            kwargs['num_inference_steps'] = num_inference_steps
            
        pipe_out = self.model(
            img,
            match_input_resolution=match_input_resolution,
            **kwargs
        )

        # Apply mask if exists
        prediction = pipe_out.prediction[0]
        prediction = self._apply_mask(prediction, mask)
        
        # Convert prediction to image
        normal_map = (prediction.clip(-1, 1) + 1) / 2
        normal_map = (normal_map * 255).astype(np.uint8)
        normal_map = Image.fromarray(normal_map)
        
        # Resize back to original dimensions if needed
        if match_input_resolution:
            normal_map = normal_map.resize(
                orig_size, 
                Image.Resampling.LANCZOS
            )
            
        return normal_map
    
    def visualize_normals(self, img: Image.Image, **kwargs) -> Image.Image:
        """Convert normal map to RGB visualization."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        prediction = np.array(img).astype(np.float32) / 255.0 * 2 - 1
        prediction = np.expand_dims(prediction, axis=0)
        return self.model.image_processor.visualize_normals(prediction)[-1]

def StableNormal(*args, **kwargs):
    """Not vendored: the two-stage StableNormal needs `stablenormal.pipeline_stablenormal` (einops), a network
    `torch.hub.load('facebookresearch/dinov2')` and the `Stable-X/stable-normal-v0-1` weights. Use StableNormal_turbo."""
    raise NotImplementedError("StableNormal() is not available in this vendored copy; use StableNormal_turbo().")

def StableNormal_turbo(local_cache_dir: Optional[str] = None, device="cuda:0", yoso_version='yoso-normal-v1-5') -> Predictor:
    """Load the StableNormal_turbo pipeline for a faster inference."""

    yoso_weight_path = _resolve_yoso_dir(local_cache_dir, yoso_version)
    if os.environ.get("STABLENORMAL_LOAD_MODE", "pipeline").lower() == "explicit":
        pipe = _load_yoso_pipeline_explicit(yoso_weight_path).to(device, silence_dtype_warnings=True)
    else:
        _alias_legacy_diffusers_controlnet_module()
        pipe = YOSONormalsPipeline.from_pretrained(yoso_weight_path,
                                                   trust_remote_code=True, safety_checker=None, variant="fp16",
                                                   t_start=0, **_dtype_kwargs(torch.float16))
        # fp16 weights parked on the CPU are fine (they only run on the GPU); diffusers would warn about it
        pipe = pipe.to(device, silence_dtype_warnings=True)

    return Predictor(pipe)

def resize_image(input_image: Image.Image, resolution: int = 1024) -> Image.Image:
    """
    Resize image to target resolution while maintaining aspect ratio and ensuring dimensions are multiples of 64.
    
    Args:
        input_image: PIL Image to resize
        resolution: Target resolution for the shorter dimension
        
    Returns:
        Resized PIL Image
    """
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    input_image_np = np.asarray(input_image)
    H, W, C = input_image_np.shape
    H, W = float(H), float(W)
    
    k = float(resolution) / max(H, W)
    new_H = H * k
    new_W = W * k
    new_H = int(np.round(new_H / 64.0)) * 64
    new_W = int(np.round(new_W / 64.0)) * 64
    
    resized_image = input_image.resize((new_W, new_H), Image.Resampling.LANCZOS)
    return resized_image