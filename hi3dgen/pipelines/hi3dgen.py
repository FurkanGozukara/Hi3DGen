# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
#
# Original file was released under MIT, with the full license text
# available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
#
# This modified file is released under the same license.
from typing import *
from contextlib import contextmanager
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


# torch.hub name of the image encoder -> checkpoint file DownloadModels.py stores in weights/
DINOV2_WEIGHT_FILES = {
    "dinov2_vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
}

# The BiRefNet model is shared: one background removal at a time (an upload may happen while a multi-image job runs)
_BIREFNET_LOCK = threading.RLock()


def _prepare_dinov2_attention():
    """DINOv2 runs its attention in fp32 and uses xformers whenever xformers imports.

    xformers only ships fp16/bf16 flash kernels and cutlass kernels up to compute capability 9.0, so on newer GPUs
    (RTX 50 series) fp32 attention has no kernel and the first generation would fail. Probe once; when the kernel is
    missing, DINOv2 falls back to its own attention (same fp32 math). Must run before DINOv2 is first imported.
    """
    import os

    if "XFORMERS_DISABLED" in os.environ:
        return
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        import xformers.ops as xops

        probe = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float32)
        xops.memory_efficient_attention(probe, probe, probe)
        torch.cuda.synchronize()
    except Exception as exc:
        os.environ["XFORMERS_DISABLED"] = "1"
        print(f"[DINOv2] xformers fp32 attention is not available here ({type(exc).__name__}); DINOv2 uses its built-in attention.")


class Hi3DGenPipeline(Pipeline):

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "Hi3DGenPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(Hi3DGenPipeline, Hi3DGenPipeline).from_pretrained(path)
        new_pipeline = Hi3DGenPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def to(self, device) -> None:
        super().to(device)
        if torch.device(device).type != "cuda":
            # the mesh decoder keeps its dense marching-cubes grid on the GPU: release it with the models
            extractor = getattr(self.models.get('slat_decoder_mesh'), 'mesh_extractor', None)
            if extractor is not None and hasattr(extractor, 'clear_cache'):
                extractor.clear_cache()

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model from local cache only.
        """
        import os

        # First, check in our custom weights directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        weights_dir = os.path.join(project_root, 'weights')
        local_dinov2_path = os.path.join(weights_dir, "github--facebookresearch--dinov2")

        if os.path.exists(local_dinov2_path):
            print(f"Loading DINOv2 from local weights: {local_dinov2_path}")
            _prepare_dinov2_attention()
            weights_file = os.path.join(weights_dir, DINOV2_WEIGHT_FILES.get(name, f"{name}_pretrain.pth"))
            if os.path.isfile(weights_file):
                # Same checkpoint torch.hub would download, read from the weights folder (no network at start).
                dinov2_model = torch.hub.load(local_dinov2_path, name, source='local', pretrained=False)
                dinov2_model.load_state_dict(torch.load(weights_file, map_location='cpu', weights_only=True), strict=True)
            else:
                print(f"DINOv2 weights not found at {weights_file}; downloading them now (run DownloadModels.py to fetch them up front).")
                dinov2_model = torch.hub.load(local_dinov2_path, name, source='local', pretrained=True)
        else:
            # Fallback to torch.hub cache directory
            hub_dir = torch.hub.get_dir()
            torch_hub_dinov2_path = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
            if os.path.exists(torch_hub_dinov2_path):
                print(f"Loading DINOv2 from torch hub cache: {torch_hub_dinov2_path}")
                dinov2_model = torch.hub.load(torch_hub_dinov2_path, name, source='local', pretrained=True)
            else:
                raise FileNotFoundError(
                    f"DINOv2 model not found in local cache.\n"
                    f"Expected locations:\n"
                    f"  1. {local_dinov2_path}\n"
                    f"  2. {torch_hub_dinov2_path}\n"
                    f"Please run DownloadModels.py to download required models."
                )
        
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image, resolution=518, device=None) -> Image.Image:
        """
        Preprocess the input image using BiRefNet for background removal.
        Includes padding to maintain aspect ratio when resizing to 518x518.

        `device="cpu"` removes the background on the CPU (for when a generation is using the GPU); by default the GPU
        is used when there is one.
        """
        # if has alpha channel, use it directly
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, -1]
            if not np.all(alpha == 255):
                has_alpha = True
        
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            
            with _BIREFNET_LOCK:
                # Load BiRefNet model if not already loaded
                if getattr(self, 'birefnet_model', None) is None:
                    try:
                        print("BiRefNet: Loading model for preprocessing...")
                        self._lazy_load_birefnet()
                    except Exception as e:
                        print(f"BiRefNet: Error during lazy loading: {e}")
                        # Reset state and try once more
                        self.reset_birefnet_state()
                        print("BiRefNet: Retrying after state reset...")
                        self._lazy_load_birefnet()

                # Get mask using BiRefNet
                mask = self._get_birefnet_mask(input, device)

            # Convert input to RGBA and apply mask
            input_rgba = input.convert('RGBA')
            input_array = np.array(input_rgba)
            input_array[:, :, 3] = mask * 255  # Apply mask to alpha channel
            output = Image.fromarray(input_array)

        # Process the output image
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:  # Handle case where no foreground is detected
            return input.convert('RGB')
        
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        
        # Calculate and apply crop bbox
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2)
        )
        
        # Ensure bbox is within image bounds
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(output.width, bbox[2]),
            min(output.height, bbox[3])
        )
        
        output = output.crop(bbox)
        
        # Add padding to maintain aspect ratio
        width, height = output.size
        if width > height:
            new_height = width
            padding = (width - height) // 2
            padded_output = Image.new('RGBA', (width, new_height), (0, 0, 0, 0))
            padded_output.paste(output, (0, padding))
        else:
            new_width = height
            padding = (height - width) // 2
            padded_output = Image.new('RGBA', (new_width, height), (0, 0, 0, 0))
            padded_output.paste(output, (padding, 0))
        
        # Resize padded image to target size
        padded_output = padded_output.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        # Final processing
        output = np.array(padded_output).astype(np.float32) / 255
        output = np.dstack((
            output[:, :, :3] * output[:, :, 3:4],  # RGB channels premultiplied by alpha
            output[:, :, 3]                         # Original alpha channel
        ))
        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        
        return output

    def _lazy_load_birefnet(self):
        """Lazy loading of the BiRefNet model from local cache only.

        The model code ships with the app (hi3dgen/third_party/birefnet) and only model.safetensors is read, so no
        remote code, no transformers/timm/kornia and no network are involved. The model is created on the CPU and
        moved to the GPU only while a mask is computed (see _get_birefnet_mask).
        """
        from ..third_party.birefnet import load_birefnet

        import os

        # Check for local cached model in weights directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root
        weights_dir = os.path.join(project_root, 'weights')
        local_birefnet_path = os.path.join(weights_dir, "models--ZhengPeng7--BiRefNet")

        if os.path.exists(local_birefnet_path):
            print(f"BiRefNet: Using local cached model at {local_birefnet_path}")
            self.birefnet_model = load_birefnet(local_birefnet_path)
        else:
            # Check HuggingFace cache as fallback
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            hf_birefnet_path = os.path.join(hf_cache_dir, "models--ZhengPeng7--BiRefNet")
            if os.path.exists(hf_birefnet_path):
                print(f"BiRefNet: Using HuggingFace cached model at {hf_birefnet_path}")
                self.birefnet_model = load_birefnet(hf_birefnet_path)
            else:
                raise FileNotFoundError(
                    f"BiRefNet model not found in local cache.\n"
                    f"Expected locations:\n"
                    f"  1. {local_birefnet_path}\n"
                    f"  2. {hf_birefnet_path}\n"
                    f"Please run DownloadModels.py to download required models."
                )

        self.birefnet_model.eval()
    
    def reset_birefnet_state(self):
        """Reset BiRefNet model state - useful after cancellation or errors"""
        if hasattr(self, 'birefnet_model'):
            try:
                print("Hi3DGen Pipeline: Resetting BiRefNet model state...")
                del self.birefnet_model
                # Force garbage collection and CUDA cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Hi3DGen Pipeline: BiRefNet model state reset successfully")
            except Exception as e:
                print(f"Hi3DGen Pipeline: Error during BiRefNet reset: {e}")
        else:
            print("Hi3DGen Pipeline: No BiRefNet model to reset")

    def _run_birefnet(self, input_images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Run BiRefNet on `device` (the model is moved there first) and return the sigmoid map on the CPU."""
        from ..third_party.birefnet import ieee_fp32_convs

        model = self.birefnet_model
        if next(model.parameters()).device != device:
            model.to(device)
        with torch.no_grad(), ieee_fp32_convs():
            return model(input_images.to(device))[-1].sigmoid().cpu()

    def park_birefnet(self):
        """Move the BiRefNet model to the CPU (frees its GPU memory)."""
        with _BIREFNET_LOCK:
            model = getattr(self, 'birefnet_model', None)
            if model is not None:
                model.to("cpu")

    def _get_birefnet_mask(self, image: Image.Image, device=None) -> np.ndarray:
        """Get object mask using BiRefNet.

        Runs in fp32 on the GPU when there is one (the model is parked on the CPU again afterwards unless
        `birefnet_keep_on_gpu` is set), with a CPU fallback when the GPU is out of memory. Convolutions run without
        TF32, so the mask matches the CPU result as closely as possible. `device="cpu"` forces the CPU.
        """
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0)

        use_gpu = torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda")
        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        with _BIREFNET_LOCK:
            try:
                try:
                    preds = self._run_birefnet(input_images, device)
                except torch.cuda.OutOfMemoryError:
                    print("BiRefNet: not enough GPU memory, running on the CPU instead")
                    self.birefnet_model.to("cpu")
                    torch.cuda.empty_cache()
                    preds = self._run_birefnet(input_images, torch.device("cpu"))
            finally:
                if use_gpu and not getattr(self, "birefnet_keep_on_gpu", False):
                    self.birefnet_model.to("cpu")
                    torch.cuda.empty_cache()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        return (mask_np > 128).astype(np.uint8)

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        with samplers.sampling_stage("Sparse structure"):
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            )["samples"]
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh',],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        with samplers.sampling_stage("Structured latent"):
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            )["samples"]

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh',],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            # restore even when sampling raised (out of memory, cancel), otherwise the sampler stays patched
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
