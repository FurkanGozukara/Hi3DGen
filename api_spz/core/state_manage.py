import logging
import torch
from pathlib import Path
import os
from typing import Optional
from hi3dgen.pipelines import Hi3DGenPipeline # Assuming this is the correct import path for your project structure

logger = logging.getLogger("stable3dgen_api")


class Hi3DGenState:
    
    def __init__(self):
        self.temp_dir = Path("temp") # Make sure this path is correct relative to your execution
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline: Optional[Hi3DGenPipeline] = None
        self.normal_predictor: Optional[torch.nn.Module] = None # Keep as torch.nn.Module for flexibility
        self._device_preference = "cpu" # Store user's device preference (cuda or cpu)
        self._active_device = "cpu" # Actual device models are currently on (for dynamic moving)


    def cleanup(self):
        logger.info("Hi3DGenState cleanup: Releasing models.")
        # Ensure models are on CPU before deleting to free GPU VRAM if they were moved
        if self.pipeline and hasattr(self.pipeline, 'cpu'):
            self.pipeline.cpu()
        del self.pipeline
        self.pipeline = None

        if self.normal_predictor:
            # The normal_predictor wrapper from torch.hub might have an internal 'model'
            if hasattr(self.normal_predictor, 'model') and hasattr(self.normal_predictor.model, 'cpu'):
                self.normal_predictor.model.cpu()
            elif hasattr(self.normal_predictor, 'cpu'): # If the wrapper itself is an nn.Module
                self.normal_predictor.cpu()
        del self.normal_predictor
        self.normal_predictor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._active_device = "cpu"


    def initialize_pipeline(self, 
                           model_path: str,
                           normal_predictor_repo: str, 
                           normal_predictor_model: str = "StableNormal_turbo",
                           yoso_version: str = "yoso-normal-v1-8-1",
                           weights_cache_dir: str = "./weights", # Ensure this path is correct
                           device: Optional[str] = None):
        
        self._device_preference = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Models will be loaded to CPU first, then moved to _device_preference when used.
        self._active_device = "cpu" # Initial load to CPU
        
        logger.info(f"Initializing Hi3DGen state. Preferred active device: {self._device_preference}. Initial load to CPU.")
        logger.info(f"Pipeline path: '{model_path}', NormalPredictor: '{normal_predictor_repo}/{normal_predictor_model}'")
        # Ensure weights_cache_dir exists, as torch.hub and from_pretrained might need it.
        # This directory should ideally be handled by your app.py's cache_weights or similar setup.
        Path(weights_cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            # 1. Load Hi3DGenPipeline to CPU
            logger.info(f"Loading Hi3DGenPipeline from '{model_path}' to CPU...")
            self.pipeline = Hi3DGenPipeline.from_pretrained(model_path)
            if self.pipeline is None: 
                raise RuntimeError(f"Hi3DGenPipeline.from_pretrained('{model_path}') returned None.")
            
            self.pipeline.cpu() # Explicitly move to CPU after loading
            self._set_eval_mode(self.pipeline, "Hi3DGenPipeline")
            logger.info("Hi3DGenPipeline loaded to CPU and configured.")

            # 2. Load Normal Predictor to CPU
            is_local_repo = Path(normal_predictor_repo).is_dir() and \
                            Path(normal_predictor_repo, "hubconf.py").exists()
            hub_source_type = 'local' if is_local_repo else 'github'
            
            logger.info(f"Loading Normal Predictor '{normal_predictor_model}' from '{normal_predictor_repo}' (source: {hub_source_type}) to CPU. Cache dir: {weights_cache_dir}")
            
            # torch.hub.load loads the model wrapper. The wrapper's __init__ might try to move its internal model to CUDA.
            # We want to control this. For now, we load it and then immediately ensure its internal model is on CPU if possible.
            self.normal_predictor = torch.hub.load(
                repo_or_dir=normal_predictor_repo,
                model=normal_predictor_model,
                source=hub_source_type,
                trust_repo=True, 
                yoso_version=yoso_version,
                local_cache_dir=weights_cache_dir,
            )
            if self.normal_predictor is None: 
                raise RuntimeError(f"torch.hub.load for '{normal_predictor_model}' from '{normal_predictor_repo}' returned None.")

            # Ensure the loaded normal_predictor (and its internal model) are on CPU
            if hasattr(self.normal_predictor, 'model') and hasattr(self.normal_predictor.model, 'cpu'):
                self.normal_predictor.model.cpu() # Target the internal YOSONormalsPipeline
                self._set_eval_mode(self.normal_predictor.model, "NormalPredictor.model")
            elif hasattr(self.normal_predictor, 'cpu'): # If the wrapper itself is an nn.Module
                self.normal_predictor.cpu()
                self._set_eval_mode(self.normal_predictor, "NormalPredictor (wrapper)")

            logger.info("Normal Predictor loaded to CPU and configured.")
            logger.info("Hi3DGenState initialization successful. Models are on CPU.")

        except Exception as e:
            logger.error(f"Hi3DGenState initialization failed: {e}", exc_info=True)
            self.cleanup() # Use cleanup to release any partially loaded models
            raise


    def _set_eval_mode(self, model_container, container_name: str):
        """Helper to recursively set eval mode on PyTorch modules."""
        if hasattr(model_container, 'eval') and callable(model_container.eval):
            model_container.eval()
            logger.debug(f"Set {container_name} to evaluation mode.")

        # For Hi3DGenPipeline, which has a 'models' dict
        if isinstance(model_container, Hi3DGenPipeline) and hasattr(model_container, 'models') and model_container.models:
            for model_name, model_instance in model_container.models.items():
                if hasattr(model_instance, 'eval') and callable(model_instance.eval):
                    model_instance.eval()
                    logger.debug(f"  Sub-model {model_name} in {container_name} set to eval mode.")
        # For NormalPredictor, which might have an internal 'model' that is a pipeline
        elif hasattr(model_container, 'model') and hasattr(model_container.model, 'modules'): # Check if internal model is nn.Module like
             if hasattr(model_container.model, 'eval') and callable(model_container.model.eval):
                model_container.model.eval()
                logger.debug(f"  Internal model of {container_name} set to eval mode.")


# Global state instance
state = Hi3DGenState()