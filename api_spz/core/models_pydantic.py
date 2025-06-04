from enum import Enum
from typing import Optional, Dict
from fastapi import Form
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm:
    def __init__(
        self,
        seed: int = Form(42), # Default from hi3dgen.py, app.py uses 0 then random
        # Parameters for Sparse Structure Sampler
        ss_guidance_strength: float = Form(3.0),
        ss_sampling_steps: int = Form(50),
        # Parameters for Structured Latent (slat) Sampler
        slat_guidance_strength: float = Form(3.0),
        slat_sampling_steps: int = Form(6),
        # Post-processing
        poly_count_pcnt: float = Form(0.7), # if UI sends 0-100, will be normalized to 0.0-1.0
        # Output format
        output_format: str = Form("glb"),
        # Texturing - Hi3DGen does not seem to have an integrated texturing pipeline.
        # Keep these for potential future integration or if a separate texturing step is added.
        # For now, apply_texture will effectively be ignored in the core hi3dgen part.
        apply_texture: bool = Form(False),
        texture_size: int = Form(1024), # Default, but not used by hi3dgen core

    ):
        self.seed = seed
        self.ss_guidance_strength = ss_guidance_strength
        self.ss_sampling_steps = ss_sampling_steps
        self.slat_guidance_strength = slat_guidance_strength
        self.slat_sampling_steps = slat_sampling_steps
        self.poly_count_pcnt = poly_count_pcnt # Will be normalized in generation.py
        self.output_format = output_format
        self.apply_texture = apply_texture
        self.texture_size = texture_size


class GenerationResponse(BaseModel):
    status: TaskStatus
    progress: int = 0
    message: str = ""
    model_url: Optional[str] = None # Only used if generation is complete


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool