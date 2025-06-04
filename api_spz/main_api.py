import logging
import os
import sys
import platform
import torch
from contextlib import asynccontextmanager
from pathlib import Path # Use Path for easier path manipulation

# Add the project root directory (parent of api_spz) to sys.path
# This allows imports from api_spz and potentially other top-level modules if needed.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent # C:\...\code
sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Define WEIGHTS_DIR relative to the project root
# It will be C:\...\code\weights
WEIGHTS_DIR = PROJECT_ROOT_DIR / "weights"
TEMP_DIR_API = PROJECT_ROOT_DIR / "temp_api" # API specific temp for logs, etc.

# -------------LOW VRAM TESTING -------------
#
# # only used for debugging, to emulate low-vram graphics cards:
#
# torch.cuda.set_per_process_memory_fraction(0.5)  # Limit to 43% of my available VRAM, for testing.
# And/or set maximum split size (in MB)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import the StableProjectorz API routes
from api_spz.core.state_manage import state # Assuming state_manage.py is in api_spz.core
from api_spz.routes.generation import router as generation_router # Assuming generation.py is in api_spz.routes

# Set up logging and temp directories for the API
TEMP_DIR_API.mkdir(exist_ok=True)
(TEMP_DIR_API / "current_generation").mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(TEMP_DIR_API / "api.log")
    ]
)
logger = logging.getLogger("stable3dgen_api")


print(
    f"\n[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}\n"
)

parser = argparse.ArgumentParser(description="Run Stable3DGen (Hi3DGen) - StableProjectorz API server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=7960, help="Port to bind the server to")
# Default model_path assumes it's an HF ID or a path relative to where the script is run,
# but we will resolve it against WEIGHTS_DIR if it looks like an HF ID.
parser.add_argument("--model_path", type=str, default="Stable-X/trellis-normal-v0-1", help="Hi3DGen pipeline model path (HF ID or local path)")
parser.add_argument("--normal_predictor_repo", type=str, default="hugoycj/StableNormal", help="torch.hub repo for normal predictor.")
parser.add_argument("--normal_predictor_model", type=str, default="StableNormal_turbo", help="Model name for normal predictor.")
parser.add_argument("--normal_predictor_yoso", type=str, default="yoso-normal-v1-8-1", help="YOSO version for normal predictor.")
# --weights_cache_dir is now effectively determined by WEIGHTS_DIR above.
# We keep the arg for potential advanced use but primarily rely on the auto-detected WEIGHTS_DIR.
parser.add_argument("--override_weights_cache_dir", type=str, default=None, help="Advanced: Override auto-detected weights cache directory.")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu'). Auto-detects if None.")

args = parser.parse_args()

# Determine final weights directory
final_weights_dir = Path(args.override_weights_cache_dir) if args.override_weights_cache_dir else WEIGHTS_DIR
final_weights_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Using weights directory: {final_weights_dir.resolve()}")


# Print startup information
print("\n" + "="*50)
print("Stable3DGen (Hi3DGen) - StableProjectorz API Server is starting up:")
print(f"Models will be cached/loaded from: {final_weights_dir.resolve()}")
print("If it's the first time, neural nets will download. Next runs will be faster.")
print("Touching this window will pause it. If it happens, click inside it and press 'Enter' to unpause")
print("="*50 + "\n")


def cache_weights_api(weights_dir_path: Path) -> dict:
    """
    Ensures specified models are cached locally. Adapted for API startup.
    Uses the per-file download approach for better progress visibility.
    """
    # This import is here because hf_transfer might not be a hard dependency for all users
    # and we enable it via environment variable.
    from huggingface_hub import list_repo_files, hf_hub_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Attempt to use hf-transfer

    model_ids = [
        "Stable-X/trellis-normal-v0-1", # For Hi3DGenPipeline
        "Stable-X/yoso-normal-v1-8-1",  # For YOSONormals (used by StableNormal_turbo)
        "ZhengPeng7/BiRefNet",          # For BiRefNet (used by Hi3DGenPipeline)
    ]
    cached_paths_map = {}

    logger.info("Starting model caching process...")
    for repo_id in model_ids:
        # Define the target local directory using Hugging Face's standard cache structure format.
        local_repo_root = weights_dir_path / f"models--{repo_id.replace('/', '--')}"
        logger.info(f"Processing {repo_id} for caching into {local_repo_root}...")
        sys.stdout.flush() # Ensure log appears
        local_repo_root.mkdir(parents=True, exist_ok=True)

        try:
            files_in_repo = list_repo_files(repo_id=repo_id, repo_type="model")
            num_total_files = len(files_in_repo)
            logger.info(f"  Found {num_total_files} files in {repo_id}. Starting download/verification...")
            sys.stdout.flush()

            for i, filename_in_repo in enumerate(files_in_repo):
                # Print progress before each file operation
                logger.info(f"  [{i+1}/{num_total_files}] Processing file: {filename_in_repo}")
                sys.stdout.flush()
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename_in_repo,
                        repo_type="model",
                        local_dir=str(local_repo_root),
                        force_download=False, # Only downloads if not present or outdated
                    )
                except Exception as file_e:
                    # Log individual file errors but continue for other files/repos
                    logger.warning(f"    Skipping {filename_in_repo} due to: {str(file_e)[:100]}...")
                    pass # Continue with the next file
            
            cached_paths_map[repo_id] = str(local_repo_root)
            logger.info(f"  Finished processing {repo_id}.")
        except Exception as repo_e:
            logger.error(f"  ERROR processing repository {repo_id}: {repo_e}")
        sys.stdout.flush()
    logger.info("Model caching process completed.")
    return cached_paths_map


def reduce_uvicorn_logs():
    try:
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        class NoisyRequestsFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                # Add paths you want to silence from access logs
                if "/ping HTTP/1.1" in message or \
                   "/download/spz-ui-layout/generation-3d-panel" in message or \
                   "/info/supported_operations" in message:
                    return False
                return True
        # Check if filter already added to prevent duplicates if lifespan runs multiple times (e.g. reload)
        if not any(isinstance(f, NoisyRequestsFilter) for f in uvicorn_access_logger.filters):
            uvicorn_access_logger.addFilter(NoisyRequestsFilter())
            logger.info("Added filter to reduce Uvicorn access log noise for specific endpoints.")
    except Exception as e:
        logger.warning(f"Could not apply Uvicorn access log filter: {e}")


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app_instance: FastAPI): # Renamed app to app_instance to avoid conflict
    logger.info("Server lifespan startup sequence initiated.")

    cache_weights_api(final_weights_dir) # Use the determined final_weights_dir

    final_device_name = args.device
    if final_device_name is None:
        final_device_name = "cuda" if torch.cuda.is_available() else "cpu"
   
    # Resolve actual Hi3DGen model path (it might be an HF ID or a local path)
    # `cache_weights_api` ensures HF IDs are downloaded to final_weights_dir.
    resolved_hi3dgen_model_path = args.model_path
    if not Path(args.model_path).is_dir(): # If it's not already a local directory path
        # Assume it's an HF ID; construct its expected local cache path
        hf_style_subdir = f"models--{args.model_path.replace('/', '--')}"
        potential_local_path = final_weights_dir / hf_style_subdir
        if potential_local_path.is_dir():
            resolved_hi3dgen_model_path = str(potential_local_path)
            logger.info(f"Resolved Hi3DGen model ID '{args.model_path}' to local cache: {resolved_hi3dgen_model_path}")
        else:
            # This case should ideally not happen if cache_weights_api worked for this model_id.
            # Or, model_path might be a relative local path that's not yet absolute.
            logger.warning(f"Hi3DGen model_path '{args.model_path}' is not an existing directory. "
                           f"If it's an HF ID, its cache path '{potential_local_path}' was not found after caching. "
                           f"Attempting to load '{args.model_path}' directly.")
            # Hi3DGenPipeline.from_pretrained will attempt to load it. If it's an HF ID not locally cached,
            # it might try to download (undesirable if we want all downloads via cache_weights_api).

    state.initialize_pipeline(
        model_path=resolved_hi3dgen_model_path,
        normal_predictor_repo=args.normal_predictor_repo,
        normal_predictor_model=args.normal_predictor_model,
        yoso_version=args.normal_predictor_yoso,
        weights_cache_dir=str(final_weights_dir), # Crucial for torch.hub.load
        device=final_device_name
    )
    logger.info(f"Initialized Hi3DGen pipeline from '{resolved_hi3dgen_model_path}' and normal predictor '{args.normal_predictor_repo}/{args.normal_predictor_model}' on {final_device_name}")
    reduce_uvicorn_logs()
    print("\n" + "="*50)
    print(f"Stable3DGen (Hi3DGen) - StableProjectorz API Server v1.0.0")
    print(f"Server is active and listening on {args.host}:{args.port}")
    logger.info(f"Now in StableProjectorz, enter the 3D mode, click on the connection button and enter {args.host}:{args.port}")
    print("="*50 + "\n")
    
    yield  # This is where the FastAPI application runs
    
    # Shutdown: Clean up resources
    state.cleanup()
    logger.info("Server shutting down, resources cleaned up")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Stable3DGen (Hi3DGen) - StableProjectorz API",
    description="API for Hi3DGen 3D generation, compatible with StableProjectorz", 
    version="1.0.0", 
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the generation routes
app.include_router(generation_router)

# Mount static files for downloads, using the API-specific temp directory
app.mount("/downloads", StaticFiles(directory=TEMP_DIR_API / "current_generation"), name="downloads")


@app.get("/")
async def root():
    return {
        "message": "Stable3DGen (Hi3DGen) - StableProjectorz API is running",
        "status": "ready",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host=args.host, port=args.port, log_level="info", reload=False)
