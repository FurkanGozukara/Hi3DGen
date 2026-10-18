"""Hi3DGen SECourses app (Gradio 6): 3D meshes from one image, several views or a whole folder of images."""
import argparse
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('SPCONV_ALGO', 'native')  # read when spconv is imported; override from the shell to experiment
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
# Every model is read from the weights folder: never wait for (or call) the Hugging Face Hub, set to 0 to allow it
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def configure_console():
    """UTF-8 console output that never raises on characters the Windows code page cannot show."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            try:
                reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
            except (OSError, TypeError, ValueError):
                pass


configure_console()

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

import ui_theme
from ui_theme import btn, status_card
from hi3dgen.pipelines import Hi3DGenPipeline
from logic.auto_save import open_outputs_folder
from logic.batch_processing import BatchSettings, create_batch_processor
from logic.cancellation import (
    ProcessingState, finish_processing, get_cancellation_manager, get_processing_status, request_cancellation,
    should_cancel, start_single_processing,
)
from logic.parameter_info import format_parameter_info_html
from logic.preset_file_manager import initialize_preset_file_system
from logic.preset_manager import (
    delete_preset_from_ui, get_preset_choices_for_ui, get_preset_status_for_ui, initialize_presets_for_ui,
    load_preset_for_ui, save_preset_from_ui, validate_preset_name,
)
from logic.preset_validation import validate_ui_parameters
from logic.processing_core import ProcessingCore, ProcessingParameters

try:
    from logic.system_status import print_system_status
    SYSTEM_STATUS_AVAILABLE = True
except ImportError as e:
    print(f'System status module not available: {e}')
    SYSTEM_STATUS_AVAILABLE = False

APP_VERSION = 'V18'
PATREON_URL = 'https://www.patreon.com/posts/130766890'
MAX_SEED = int(np.iinfo(np.int32).max)
TMP_DIR = str(ROOT / 'tmp')
WEIGHTS_DIR = str(ROOT / 'weights')
GPU = 'gpu'  # every event that uses the GPU shares this concurrency group, so they run one at a time
EXPORT_FORMATS = ['obj', 'glb', 'ply', 'stl']
DISPLAY_MODES = ['solid', 'point_cloud', 'wireframe']
MODEL_LABEL = '3D Model Preview (each model is about 10-40 MB and may take a moment to load)'
# Values of the settings in the order of the presets: seed, stage 1 guidance/steps, stage 2 guidance/steps, polygon
# percentage, xatlas (max cost, normal seam weight, resolution, padding), normal map (resolution, match input), auto-save
DEFAULT_VALUES = (-1, 3.0, 50, 3.0, 6, 0.5, 8.0, 1.0, 1024, 2, 768, True, True, True, True, True)

# Weights the app needs, relative to WEIGHTS_DIR
MODEL_FILES = {
    'models--Stable-X--trellis-normal-v0-1': [
        'pipeline.json', 'ckpts/ss_dec_conv3d_16l8_fp16.safetensors', 'ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors',
        'ckpts/slat_flow_normal_dit_L_64l8p2_fp16.safetensors', 'ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors',
    ],
    'models--Stable-X--yoso-normal-v1-8-1': [
        'model_index.json', 'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/diffusion_pytorch_model.fp16.safetensors',
        'controlnet/controlnetvae.py', 'controlnet/diffusion_pytorch_model.fp16.safetensors',
    ],
    'models--ZhengPeng7--BiRefNet': ['config.json', 'model.safetensors'],
    'github--facebookresearch--dinov2': ['hubconf.py'],
    '.': ['dinov2_vitl14_reg4_pretrain.pth'],
}

# The processing system is created once in main(): one GPU, so one pipeline for every browser session
hi3dgen_pipeline = None
processing_core = None
batch_processor = None
cancellation_manager = None


def verify_weights(weights_dir: str = WEIGHTS_DIR) -> bool:
    """Check that the downloaded models are there. Nothing is downloaded here."""
    print('Verifying the downloaded models...')
    missing = []
    for folder, files in MODEL_FILES.items():
        base = Path(weights_dir) / folder
        lacking = [name for name in files if not (base / name).is_file()]
        label = folder if folder != '.' else 'weights'
        if lacking:
            missing.append(label)
            print(f'  MISSING in {label}: {", ".join(lacking)}')
        else:
            print(f'  OK: {label}')
    if missing:
        print('\nSome models are missing. Run the model download of the installer (Windows_Resume_Model_Download.bat, '
              'or `python DownloadModels.py` on Linux) and start the app again.')
    return not missing


# ----------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f'{seconds:.1f} s'
    if seconds < 60:
        return f'{seconds:.0f} s'
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes} min {rest:02d} s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} h {minutes:02d} min'


def file_paths(files) -> List[str]:
    """Paths of the files of a gr.File (strings, NamedString or dict, depending on the Gradio version)."""
    paths = []
    for item in files or []:
        path = item.get('path') if isinstance(item, dict) else getattr(item, 'name', item)
        if path:
            paths.append(str(path))
    return paths


def clean_path(value) -> str:
    """A folder typed or pasted by the user (Windows 'Copy as path' adds quotes)."""
    return os.path.expanduser(str(value or '').strip().strip('"').strip("'"))


def is_busy() -> bool:
    return get_processing_status().processing_state not in (ProcessingState.IDLE, ProcessingState.CANCELLED)


def int_value(value, default=0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def make_parameters(seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
                    poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
                    normal_map_resolution, normal_match_input_resolution,
                    auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl) -> ProcessingParameters:
    return ProcessingParameters(
        seed=int_value(seed, -1),
        ss_guidance_strength=float(ss_guidance_strength), ss_sampling_steps=int_value(ss_sampling_steps, 50),
        slat_guidance_strength=float(slat_guidance_strength), slat_sampling_steps=int_value(slat_sampling_steps, 6),
        poly_count_pcnt=float(poly_count_pcnt),
        xatlas_max_cost=float(xatlas_max_cost), xatlas_normal_seam_weight=float(xatlas_normal_seam_weight),
        xatlas_resolution=int_value(xatlas_resolution, 1024), xatlas_padding=int_value(xatlas_padding, 2),
        normal_map_resolution=int_value(normal_map_resolution, 768),
        normal_match_input_resolution=bool(normal_match_input_resolution),
        auto_save_obj=bool(auto_save_obj), auto_save_glb=bool(auto_save_glb),
        auto_save_ply=bool(auto_save_ply), auto_save_stl=bool(auto_save_stl),
    )


# ----------------------------------------------------------------------------------------------------
# Live status of a running job
# ----------------------------------------------------------------------------------------------------
# Share of the progress bar of each stage. Most of the time goes into UV unwrapping (xatlas), which reports no progress.
STAGE_RANGES = {
    'Initialization': (0.00, 0.02), 'Preprocessing': (0.02, 0.06), 'Normal Prediction': (0.02, 0.06),
    '3D Generation': (0.06, 0.20), 'Mesh Processing': (0.20, 0.27), 'UV Unwrapping': (0.27, 0.95),
    'File Export': (0.95, 0.97), 'Auto-save': (0.97, 0.99), 'Completed': (1.0, 1.0),
}
SAMPLING_RANGES = {'Sparse structure': (0.06, 0.15), 'Structured latent': (0.15, 0.20)}
# Stages without steps ease towards the end of their range with this time constant (seconds)
EASE_SECONDS = {'Preprocessing': 3.0, 'Normal Prediction': 3.0, 'Mesh Processing': 6.0, 'UV Unwrapping': 60.0}
SAMPLING_STEP = re.compile(r'^(?P<name>.+), step (?P<step>\d+)/(?P<total>\d+)$')


def stage_fraction(stage: str, details: str = '', stage_elapsed: float = 0.0) -> float:
    low, high = STAGE_RANGES.get(stage, (0.0, 0.0))
    if stage == '3D Generation':
        match = SAMPLING_STEP.match(details or '')
        if match:
            step_low, step_high = SAMPLING_RANGES.get(match['name'], (low, high))
            return step_low + (step_high - step_low) * (int(match['step']) - 1) / max(int(match['total']), 1)
        return low
    tau = EASE_SECONDS.get(stage)
    if tau:
        return low + (high - low) * 0.97 * (1.0 - math.exp(-max(stage_elapsed, 0.0) / tau))
    return low


class Run:
    """One generation in a worker thread; the Gradio generator that started it polls it for the status card.

    The work itself never touches Gradio: it reports through `report`, so the page keeps getting live updates, and a cancel
    is just a flag the stages check (see logic/cancellation.py).
    """

    def __init__(self, label: str, first_stage: str = 'Normal map'):
        self.label = label
        self.first_stage = first_stage  # name of what happens before the 3D generation, for the final timings
        self.started = time.monotonic()
        self.stage, self.details, self.stage_started = 'Starting', '', self.started
        self.seed = None
        self.result = None
        self.error: Optional[BaseException] = None
        self.finished = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._preview = None
        self._mesh_preview = None

    # -- called by the worker
    def report(self, stage: str, details: str = ''):
        with self._lock:
            if stage != self.stage:
                self.stage, self.stage_started = stage, time.monotonic()
            self.details = details
            if details.startswith('Using seed '):
                self.seed = details[len('Using seed '):]
        self._wake.set()

    def set_preview(self, image):
        with self._lock:
            self._preview = image
        self._wake.set()

    def set_mesh_preview(self, path):
        with self._lock:
            self._mesh_preview = path
        self._wake.set()

    # -- called by the generator
    def take_preview(self):
        with self._lock:
            image, self._preview = self._preview, None
        return image

    def take_mesh_preview(self):
        with self._lock:
            path, self._mesh_preview = self._mesh_preview, None
        return path

    def snapshot(self):
        with self._lock:
            return self.stage, self.details, self.stage_started, self.seed

    def start(self, work):
        def main():
            try:
                self.result = work(self)
            except BaseException as exc:  # reported on the status card, the thread must not die silently
                self.error = exc
                if isinstance(exc, RuntimeError) and 'already in' in str(exc):
                    print(f'{self.label}: not started, {exc}')  # another job is running: expected, no traceback needed
                else:
                    traceback.print_exc()
            finally:
                self.finished.set()
                self._wake.set()
        threading.Thread(target=main, name='hi3dgen-job', daemon=True).start()
        return self

    def ticks(self, min_gap: float = 0.25, max_gap: float = 1.0):
        """Yield whenever something changed (at most every `min_gap` s) or after `max_gap` s, until the job has ended."""
        last = 0.0
        while True:
            self._wake.wait(timeout=max_gap)
            self._wake.clear()
            pause = min_gap - (time.monotonic() - last)
            if pause > 0 and not self.finished.is_set():
                time.sleep(pause)
            last = time.monotonic()
            yield
            if self.finished.is_set():
                return


def cancellation_requested() -> bool:
    try:
        return bool(should_cancel())
    except Exception:
        return False


def running_card(run: Run) -> str:
    stage, details, stage_started, seed = run.snapshot()
    now = time.monotonic()
    text = f'{stage}: {details}' if details else stage
    metrics = [('Stage', stage), ('Elapsed', format_seconds(now - run.started))]
    if seed is not None:
        metrics.append(('Seed', seed))
    if stage == 'UV Unwrapping':
        metrics.append(('Note', 'xatlas reports no progress, the bar is an estimate'))
    cancelling = cancellation_requested()
    headline = f'Cancelling, finishing the current step - {text}' if cancelling else f'{run.label}: {text}'
    return status_card(headline, stage_fraction(stage, details, now - stage_started), metrics, 'warn' if cancelling else 'busy')


def result_metrics(result, first_stage: str = 'Normal map') -> list:
    metrics = [('Total', format_seconds(result.total_time))]
    if result.normal_prediction_time:
        metrics.append((first_stage, format_seconds(result.normal_prediction_time)))
    metrics += [('3D generation', format_seconds(result.pipeline_time)), ('Mesh processing', format_seconds(result.mesh_processing_time)),
                ('UV unwrapping', format_seconds(result.uv_unwrapping_time)),
                ('Faces', f'{result.original_face_count:,} -> {result.simplified_face_count:,}'),
                ('Vertices', f'{result.final_vertex_count:,}')]
    if result.seed != -1:
        metrics.append(('Seed', result.seed))
    saved = (result.auto_save_result or {}).get('folder_path')
    if saved:
        try:
            saved = str(Path(saved).resolve().relative_to(ROOT))
        except ValueError:
            pass
        metrics.append(('Saved to', saved))
    return metrics


# ----------------------------------------------------------------------------------------------------
# Meshes for the viewer and the download button
# ----------------------------------------------------------------------------------------------------
def prepare_download_file(mesh_path: Optional[str], export_format: str) -> Optional[str]:
    """The mesh as `export_format`, next to the generated GLB (converted once, reused on later requests)."""
    if not mesh_path or not os.path.exists(mesh_path):
        return None
    export_format = (export_format or 'glb').lower()
    job_dir = os.path.dirname(os.path.abspath(mesh_path))
    target = os.path.join(job_dir, f'generated_mesh_{os.path.basename(job_dir)}.{export_format}')
    try:
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            if export_format == 'glb' and mesh_path.lower().endswith('.glb'):
                shutil.copy2(mesh_path, target)
            else:
                mesh = trimesh.load_mesh(mesh_path)
                if not (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None):
                    print(f'  Warning: the mesh {mesh_path} has no UVs before export to {export_format}.')
                mesh.export(target, file_type=export_format)
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                raise RuntimeError(f'Failed to create a valid {export_format} file')
        return target
    except Exception as e:
        print(f'prepare_download_file: could not create the {export_format} file for {mesh_path}: {e}')
        traceback.print_exc()
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass
        return None


def download_button(path: Optional[str], export_format: str):
    if not path:
        return gr.DownloadButton(value=None, interactive=False, label='Download 3D Mesh')
    return gr.DownloadButton(value=path, interactive=True, label=f'Download {export_format.upper()}')


def model_viewer(path: Optional[str], display_mode: str):
    return gr.Model3D(value=path, display_mode=display_mode if display_mode in DISPLAY_MODES else 'solid', label=MODEL_LABEL)


# ----------------------------------------------------------------------------------------------------
# Generation events
# ----------------------------------------------------------------------------------------------------
def preprocess_image(image):
    """Remove the background of an uploaded image (BiRefNet): on the GPU, or on the CPU while a generation uses the GPU."""
    if image is None:
        return None
    if hi3dgen_pipeline is None:
        raise gr.Error('The Hi3DGen pipeline is not loaded.')
    device = 'cpu' if is_busy() else None
    try:
        return hi3dgen_pipeline.preprocess_image(image, resolution=1024, device=device)
    except Exception as e:
        print(f'Background removal failed: {e}')
        traceback.print_exc()
        message = str(e)
        if 'BiRefNet' in message or '401' in message or 'RepositoryNotFoundError' in message:
            print('Resetting the BiRefNet state and trying once more...')
            try:
                if hasattr(hi3dgen_pipeline, 'reset_birefnet_state'):
                    hi3dgen_pipeline.reset_birefnet_state()
                return hi3dgen_pipeline.preprocess_image(image, resolution=1024, device=device)
            except Exception as retry_error:
                print(f'Retry after the BiRefNet reset also failed: {retry_error}')
                raise gr.Error(f'Background removal failed: {retry_error}')
        raise gr.Error(f'Background removal failed: {e}')


def restore_previous_model(previous_model, display_mode: str, export_format: str):
    """(viewer, download button, model path) that show the model of the earlier run again, or nothing when there was none."""
    if previous_model and os.path.exists(previous_model):
        return (model_viewer(previous_model, display_mode), download_button(prepare_download_file(previous_model, export_format), export_format),
                previous_model)
    return model_viewer(None, display_mode), download_button(None, export_format), None


def stream_generation(run: Run, work, display_mode: str, export_format: str, previous_model=None):
    """Run `work` and stream (preview image, 3D model, download button, status card, model path) to the page.

    `previous_model` is the mesh the viewer showed before: it comes back when this run fails or is cancelled after its
    own mesh preview replaced it.
    """
    skip = gr.skip()
    run.start(work)
    preview_sent = False
    mesh_preview_shown = False
    for _ in run.ticks():
        preview = run.take_preview()
        mesh_preview = run.take_mesh_preview()
        preview_sent = preview_sent or preview is not None
        mesh_preview_shown = mesh_preview_shown or bool(mesh_preview)
        # The decimated mesh is the final shape: it is shown while the UVs are computed, the download waits for the final file
        yield (preview if preview is not None else skip,
               model_viewer(mesh_preview, display_mode) if mesh_preview else skip,
               download_button(None, export_format) if mesh_preview else skip,
               running_card(run),
               mesh_preview if mesh_preview else skip)

    result = run.result
    if run.error is not None:
        message = str(run.error)
        if isinstance(run.error, RuntimeError) and 'already in' in message:
            message = 'Another job is still running (for example one started from a page that was closed). Wait for it or cancel it.'
        viewer, download, model_path = restore_previous_model(previous_model, display_mode, export_format) if mesh_preview_shown else (skip, skip, skip)
        yield (skip, viewer, download, status_card(f'{run.label} failed', None, [('Error', message)], 'error'), model_path)
    elif result is None or not result.success or not result.mesh_path:
        message = (result.error_message if result else None) or 'Unknown error'
        if message.startswith('Cancelled'):
            card = status_card(f'{run.label} cancelled', None, [('Elapsed', format_seconds(time.monotonic() - run.started)), ('Where', message)], 'warn')
        else:
            card = status_card(f'{run.label} failed', None, [('Error', message), ('Elapsed', format_seconds(time.monotonic() - run.started))], 'error')
        viewer, download, model_path = restore_previous_model(previous_model, display_mode, export_format) if mesh_preview_shown else (skip, skip, skip)
        yield (result.normal_image if (result and not preview_sent) else skip, viewer, download, card, model_path)
    else:
        yield (skip if preview_sent else result.normal_image, model_viewer(result.mesh_path, display_mode),
               gr.DownloadButton(interactive=False, label='Preparing download...'),
               status_card(f'{run.label} finished', 1.0, result_metrics(result, run.first_stage), 'ok'), result.mesh_path)
        path = prepare_download_file(result.mesh_path, export_format)
        yield (skip, skip, download_button(path, export_format), skip, skip)


def start_job(name: str, work_fn):
    """Wrap `work_fn(run)` in the global cancellation state, which is what the cancel button flips."""
    def work(run: Run):
        start_single_processing(f'Starting {name}')
        try:
            result = work_fn(run)
            return result
        finally:
            finish_processing(f'{name} finished')
    return work


def generate_3d_with_cancellation(image, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength,
                                  slat_sampling_steps, poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight,
                                  xatlas_resolution, xatlas_padding, normal_map_resolution, normal_match_input_resolution,
                                  auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl,
                                  export_format='glb', display_mode='solid', previous_model=None):
    """Single image: normal map, 3D generation, mesh processing, UV unwrapping, export and auto-save."""
    skip = gr.skip()
    if image is None:
        gr.Warning('Upload an image or pick an example first.')
        yield skip, skip, skip, status_card('Upload an image or pick an example first', None, tone='warn'), skip
        return
    if processing_core is None:
        raise gr.Error('The processing system is not initialized, see the console.')
    params = make_parameters(seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
                             poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
                             normal_map_resolution, normal_match_input_resolution,
                             auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl)
    run = Run('Single image')
    work = start_job('single image processing',
                     lambda r: processing_core.process_single_image(image, params, r.report, preview_callback=r.set_preview,
                                                                    mesh_preview_callback=r.set_mesh_preview))
    yield from stream_generation(run, work, display_mode, export_format, previous_model)


def generate_3d_multi_with_cancellation(files, mode, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength,
                                        slat_sampling_steps, poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight,
                                        xatlas_resolution, xatlas_padding, normal_map_resolution,
                                        normal_match_input_resolution, auto_save_obj, auto_save_glb, auto_save_ply,
                                        auto_save_stl, export_format='glb', display_mode='solid', previous_model=None):
    """Several views of one object: background removal per view, one 3D generation, then the same mesh steps."""
    skip = gr.skip()
    images = []
    for path in file_paths(files):
        try:
            with Image.open(path) as img:
                images.append(img.convert('RGB'))
        except Exception as e:
            print(f'Error loading image {path}: {e}')
    if not images:
        gr.Warning('Upload at least one valid image first.')
        yield skip, skip, skip, status_card('Upload the views of your object first', None, tone='warn'), skip
        return
    if processing_core is None:
        raise gr.Error('The processing system is not initialized, see the console.')
    params = make_parameters(seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
                             poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
                             normal_map_resolution, normal_match_input_resolution,
                             auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl)
    print(f'Processing {len(images)} images in {mode} mode...')
    run = Run(f'{len(images)} images', first_stage='Background removal')
    work = start_job('multi-image processing',
                     lambda r: processing_core.process_multi_image(images, mode, params, r.report, preview_callback=r.set_preview,
                                                                   mesh_preview_callback=r.set_mesh_preview))
    yield from stream_generation(run, work, display_mode, export_format, previous_model)


def cancel_processing_ui():
    """Stop the running generation or batch: the stages check the flag, a running mesh worker is killed at once."""
    try:
        if request_cancellation('User requested cancellation'):
            gr.Info('Cancelling... the current step is finished first.')
        else:
            gr.Info('Nothing is running right now.')
    except Exception as e:
        print(f'Error requesting cancellation: {e}')
        gr.Warning(f'Could not cancel: {e}')


# ----------------------------------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------------------------------
def batch_summary_text(progress, running: bool) -> str:
    if progress.total_images == 0 and not progress.errors and not progress.skipped:
        return 'No batch has been started yet.'
    lines = [f'Images to process: {progress.total_images}', f'Processed: {progress.processed_images}',
             f'Skipped (outputs already exist): {len(progress.skipped)}', f'Errors: {len(progress.errors)}']
    if progress.cancelled_images:
        lines.append(f'Cancelled: {progress.cancelled_images}')
    lines.append(f'Elapsed: {format_seconds(progress.elapsed_time)}')
    if running and progress.estimated_time_remaining > 0:
        lines.append(f'Estimated time left: {format_seconds(progress.estimated_time_remaining)}')
    if progress.errors:
        lines += ['', 'Errors:'] + [f'  - {error}' for error in progress.errors[-5:]]
    if progress.skipped:
        shown = progress.skipped[:5]
        lines += ['', 'Skipped:'] + [f'  - {name}' for name in shown]
        if len(progress.skipped) > len(shown):
            lines.append(f'  ... and {len(progress.skipped) - len(shown)} more')
    return '\n'.join(lines)


def batch_card(progress, running: bool, tone: str, headline: str) -> str:
    total = progress.total_images
    active = list(progress.active.items())  # a copy: the batch threads change it while this runs
    fraction = None
    if total:
        partial = sum(stage_fraction(text.split(':')[0]) for _, text in active)
        finished = progress.processed_images + len(progress.errors) + progress.cancelled_images
        fraction = min((finished + min(partial, 0.9 * len(active))) / total, 1.0)
    metrics = [('Done', f'{progress.processed_images}/{total}'), ('Elapsed', format_seconds(progress.elapsed_time))]
    if running and progress.estimated_time_remaining > 0:
        metrics.append(('Time left', format_seconds(progress.estimated_time_remaining)))
    if progress.errors:
        metrics.append(('Errors', len(progress.errors)))
    if progress.skipped:
        metrics.append(('Skipped', len(progress.skipped)))
    if running:
        metrics += active[:4]  # what each image in progress is doing
    return status_card(headline, fraction, metrics, tone)


def start_batch_processing_ui(input_folder, output_folder, skip_existing, use_current_settings,
                              seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
                              poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution,
                              xatlas_padding, normal_map_resolution, normal_match_input_resolution,
                              auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl):
    """Process every image of a folder; streams (status card, results text, start button)."""
    skip = gr.skip()
    if batch_processor is None:
        raise gr.Error('The batch processor is not initialized, see the console.')
    input_folder, output_folder = clean_path(input_folder), clean_path(output_folder)
    problem = None
    if not input_folder or not output_folder:
        problem = 'Please specify both the input and the output folder.'
    elif not os.path.isdir(input_folder):
        problem = f'The input folder does not exist: {input_folder}'
    if problem:
        gr.Warning(problem)
        yield status_card(problem, None, tone='warn'), skip, skip
        return

    values = (seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, poly_count_pcnt,
              xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding, normal_map_resolution,
              normal_match_input_resolution, auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl)
    chosen = make_parameters(*(values if use_current_settings else DEFAULT_VALUES))
    settings = BatchSettings(
        input_folder=input_folder, output_folder=output_folder, skip_existing=bool(skip_existing),
        seed=chosen.seed, ss_guidance_strength=chosen.ss_guidance_strength, ss_sampling_steps=chosen.ss_sampling_steps,
        slat_guidance_strength=chosen.slat_guidance_strength, slat_sampling_steps=chosen.slat_sampling_steps,
        poly_count_pcnt=chosen.poly_count_pcnt, xatlas_max_cost=chosen.xatlas_max_cost,
        xatlas_normal_seam_weight=chosen.xatlas_normal_seam_weight, xatlas_resolution=chosen.xatlas_resolution,
        xatlas_padding=chosen.xatlas_padding, normal_map_resolution=chosen.normal_map_resolution,
        normal_match_input_resolution=chosen.normal_match_input_resolution, auto_save_obj=chosen.auto_save_obj,
        auto_save_glb=chosen.auto_save_glb, auto_save_ply=chosen.auto_save_ply, auto_save_stl=chosen.auto_save_stl,
    )
    print('Starting batch processing:')
    print(f'  Input folder: {input_folder}')
    print(f'  Output folder: {output_folder}')
    print(f'  Skip existing: {skip_existing}')
    print(f'  Use current settings: {use_current_settings}')

    run = Run('Batch')
    run.start(lambda r: batch_processor.start_batch_processing(settings))
    yield status_card('Batch: starting', None, [('Input', input_folder)], 'busy'), 'Starting...', gr.Button(interactive=False)
    for _ in run.ticks(min_gap=0.5, max_gap=1.0):
        progress = batch_processor.progress
        headline = (f'Batch: {progress.processed_images} of {progress.total_images} images done' if progress.total_images
                    else f'Batch: {progress.current_stage or "starting"}')
        cancelling = cancellation_requested()
        if cancelling:
            headline = f'Cancelling, finishing the current images - {headline}'
        yield (batch_card(progress, True, 'warn' if cancelling else 'busy', headline), batch_summary_text(progress, True), skip)

    progress = batch_processor.progress
    if run.error is not None:
        card = status_card('Batch failed', None, [('Error', str(run.error))], 'error')
        gr.Warning(f'Batch failed: {run.error}')
    elif progress.current_stage.startswith('Cancelled'):
        card = batch_card(progress, False, 'warn', f'Batch cancelled after {progress.processed_images} of {progress.total_images} images')
    elif progress.total_images == 0 and progress.skipped:
        card = batch_card(progress, False, 'ok', 'Nothing to do: every image already has its outputs')
    elif progress.total_images == 0:
        card = batch_card(progress, False, 'warn', progress.current_stage or 'No images found in the input folder')
    elif progress.errors:
        card = batch_card(progress, False, 'warn', f'Batch finished: {progress.processed_images} processed, {len(progress.errors)} failed')
    else:
        card = batch_card(progress, False, 'ok', f'Batch finished: {progress.processed_images} images processed')
    yield card, batch_summary_text(progress, False), gr.Button(interactive=True)


# ----------------------------------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------------------------------
def memory_status_text() -> str:
    if processing_core is None:
        return 'Processing system not initialized'
    return processing_core.memory.status()


def set_memory_mode(mode):
    if processing_core is not None:
        if is_busy():
            gr.Warning('The setting applies to the next generation, a job is running now.')
        processing_core.memory.set_mode(mode)
        gr.Info(f'Model memory mode: {processing_core.memory.mode}')
    return memory_status_text()


def free_gpu_memory_ui():
    if processing_core is None:
        return memory_status_text()
    if is_busy():
        gr.Warning('A generation is running. Wait for it to finish or cancel it first.')
        return memory_status_text()
    gr.Info(processing_core.memory.free_gpu_memory())
    return memory_status_text()


def set_mesh_jobs(count):
    if batch_processor is not None:
        gr.Info(f'Batch: up to {batch_processor.set_mesh_jobs(count)} meshes are unwrapped at the same time.')


def open_outputs_folder_ui():
    try:
        open_outputs_folder()
    except Exception as e:
        print(f'Could not open the outputs folder: {e}')
        gr.Warning(f'Could not open the outputs folder: {e}')


# ----------------------------------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------------------------------
def save_preset_ui_handler(preset_name, preset_description, *values):
    """Save the current settings as a preset: (dropdown, message, status, name box, description box)."""
    try:
        is_valid, validation_message = validate_preset_name(preset_name)
        if not is_valid:
            gr.Warning(validation_message)
            return (gr.Dropdown(choices=get_preset_choices_for_ui()), f'❌ {validation_message}', get_preset_status_for_ui(),
                    gr.skip(), gr.skip())
        extra = ''
        try:
            all_valid, _, validation = validate_ui_parameters(
                seed=values[0], ss_guidance_strength=values[1], ss_sampling_steps=values[2],
                slat_guidance_strength=values[3], slat_sampling_steps=values[4], poly_count_pcnt=values[5],
                xatlas_max_cost=values[6], xatlas_normal_seam_weight=values[7], xatlas_resolution=values[8],
                xatlas_padding=values[9], normal_map_resolution=values[10], normal_match_input_resolution=values[11],
                auto_save_obj=values[12], auto_save_glb=values[13], auto_save_ply=values[14], auto_save_stl=values[15])
            if not all_valid:
                print(f'Parameter validation warnings: {validation}')
                extra = f' (parameter adjustments: {validation})'
        except Exception as validation_error:
            print(f'Parameter validation error: {validation_error}')
        success, message, presets, selected = save_preset_from_ui(preset_name, preset_description, *values, overwrite=False)
        if success:
            gr.Info(f'Preset saved: {preset_name}')
            return (gr.Dropdown(choices=presets, value=selected), f'✅ {message}{extra}', get_preset_status_for_ui(),
                    '', '')
        gr.Warning(message)
        return gr.Dropdown(choices=presets, value=selected), f'❌ {message}', get_preset_status_for_ui(), gr.skip(), gr.skip()
    except Exception as e:
        print(f'Error saving preset: {e}')
        traceback.print_exc()
        gr.Warning(f'Error saving preset: {e}')
        return gr.Dropdown(choices=get_preset_choices_for_ui()), f'❌ Error saving preset: {e}', 'Error in the preset system', gr.skip(), gr.skip()


def load_preset_ui_handler(selected_preset):
    """Load a preset into the settings: (dropdown, message, status, 16 setting values)."""
    nothing = [gr.skip()] * len(DEFAULT_VALUES)
    try:
        if not selected_preset or not str(selected_preset).strip():
            return (gr.Dropdown(choices=get_preset_choices_for_ui()), '⚠️ No preset selected', get_preset_status_for_ui(), *nothing)
        success, message, presets, current, ui_values = load_preset_for_ui(selected_preset)
        if success:
            print(f"Loaded preset '{selected_preset}': seed={ui_values[0]}, polygon share={ui_values[5]:.2f}, UV resolution={ui_values[8]}")
            return (gr.Dropdown(choices=presets, value=current), f'✅ {message}', get_preset_status_for_ui(), *ui_values)
        gr.Warning(message)
        return (gr.Dropdown(choices=presets, value=current), f'❌ {message}', get_preset_status_for_ui(), *nothing)
    except Exception as e:
        print(f'Error loading preset: {e}')
        traceback.print_exc()
        gr.Warning(f'Error loading preset: {e}')
        return (gr.Dropdown(choices=get_preset_choices_for_ui()), f'❌ Error loading preset: {e}', 'Error in the preset system', *nothing)


def delete_preset_ui_handler(selected_preset):
    try:
        if not selected_preset or not str(selected_preset).strip():
            return gr.Dropdown(choices=get_preset_choices_for_ui()), '❌ No preset selected for deletion', get_preset_status_for_ui()
        success, message, presets, current = delete_preset_from_ui(selected_preset)
        (gr.Info if success else gr.Warning)(message)
        return gr.Dropdown(choices=presets, value=current), f'{"✅" if success else "❌"} {message}', get_preset_status_for_ui()
    except Exception as e:
        print(f'Error deleting preset: {e}')
        traceback.print_exc()
        return gr.Dropdown(choices=get_preset_choices_for_ui()), f'❌ Error deleting preset: {e}', 'Error in the preset system'


def initialize_preset_ui():
    """Fill the settings from the last used preset when the page opens."""
    try:
        file_success, file_message = initialize_preset_file_system()
        if not file_success:
            print(f'Preset file system initialization failed: {file_message}')
        success, message, presets, selected, ui_values = initialize_presets_for_ui()
        if not success:
            print(f'Preset system initialization failed: {message}')
            presets, selected, ui_values = ['Default'], 'Default', DEFAULT_VALUES
        return (gr.Dropdown(choices=presets, value=selected), 'Preset system ready', get_preset_status_for_ui(), *ui_values)
    except Exception as e:
        print(f'Critical error initializing the preset UI: {e}')
        traceback.print_exc()
        return (gr.Dropdown(choices=['Default'], value='Default'), f'❌ Preset system error: {e}', 'Preset system failed', *DEFAULT_VALUES)


# ----------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------
def example_images() -> List[str]:
    folder = ROOT / 'assets' / 'example_image'
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
    files.sort(key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.name))
    return [str(p) for p in files]


def example_multi_image_sets() -> List[List[str]]:
    folder = ROOT / 'assets' / 'example_multi_image'
    if not folder.is_dir():
        return []
    groups = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            groups.setdefault('_'.join(path.stem.split('_')[:-1]), []).append(str(path))
    return [paths[:3] for paths in groups.values() if len(paths) >= 2]


def show_multi_images(files):
    """Preview of the uploaded views and a hint about how many to use."""
    gallery, names = [], []
    for path in file_paths(files):
        try:
            with Image.open(path):
                pass
            gallery.append(path)
            names.append(Path(path).name)
        except Exception as e:
            print(f'Error loading image {path}: {e}')
    if not gallery:
        return [], status_card('Upload the views of your object', None, tone='idle')
    if len(gallery) < 2:
        return gallery, status_card('1 image loaded: add more views, or use the Single Image tab', None, [('Files', ', '.join(names))], 'warn')
    return gallery, status_card(f'{len(gallery)} images loaded', None, [('Files', ', '.join(names))], 'ok')


def build_ui() -> gr.Blocks:
    idle_card = status_card('Ready', None, [('State', 'Idle')], 'idle')
    startup_memory = memory_status_text()
    single_example_files = example_images()
    multi_example_sets = example_multi_image_sets()

    with gr.Blocks(title=f'Hi3DGen - SECourses App {APP_VERSION}', analytics_enabled=False, delete_cache=(3600, 2 * 86400)) as demo:
        # Header: title and the light/dark switch (a pure client-side toggle, instant even while a job runs)
        with gr.Row(elem_classes=['app-header']):
            gr.Markdown(
                f'# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging\n'
                f'SECourses App {APP_VERSION} with Auto-Save | [Premium release, tutorials and support]({PATREON_URL})',
                container=False)
            theme_button = gr.Button('🌗  Light / dark theme', elem_classes=btn('gray'), scale=0, min_width=230)
        theme_button.click(None, None, None, js=ui_theme.TOGGLE_THEME_JS, api_visibility='private')

        model_state = gr.State(None)  # path of the mesh in the viewer, per browser session

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab('Single Image'):
                        with gr.Row():
                            image_prompt = gr.Image(label='Image Prompt', image_mode='RGBA', type='pil', elem_id='image-prompt',
                                                    buttons=['download', 'fullscreen'])
                            normal_output = gr.Image(label='Normal Bridge', image_mode='RGBA', type='pil', elem_id='normal-output',
                                                     buttons=['download', 'fullscreen'])
                        with gr.Row():
                            gen_shape_btn = gr.Button('Generate 3D Mesh', size='lg', variant='primary', elem_id='generate-btn',
                                                      elem_classes=btn('emerald'))
                            single_cancel_btn = gr.Button('Cancel', size='lg', variant='stop', elem_id='cancel-btn',
                                                          elem_classes=btn('red'), scale=0, min_width=160)
                        single_status = gr.HTML(idle_card, elem_id='single-status', container=False, padding=False)
                        with gr.Row():
                            auto_save_obj_cb = gr.Checkbox(value=True, label='Auto-save OBJ', info='Wavefront OBJ format')
                            auto_save_glb_cb = gr.Checkbox(value=True, label='Auto-save GLB', info='Binary glTF format')
                        with gr.Row():
                            auto_save_ply_cb = gr.Checkbox(value=True, label='Auto-save PLY', info='Stanford Triangle format')
                            auto_save_stl_cb = gr.Checkbox(value=True, label='Auto-save STL', info='Stereolithography format')

                    with gr.Tab('🎭 Multiple Images'):
                        gr.Markdown('### Upload multiple angle views of your object (2-8 images recommended)')
                        with gr.Row():
                            with gr.Column():
                                multi_image_inputs = gr.File(label='Upload Multiple Images', file_count='multiple', file_types=['image'],
                                                             height=200, elem_id='multi-files')
                                multi_mode = gr.Radio(
                                    choices=['multidiffusion', 'stochastic'], value='multidiffusion', label='Processing Mode',
                                    info='MultiDiffusion: Average all views (recommended). Stochastic: Cycle through views.')
                            with gr.Column():
                                multi_preview = gr.Gallery(label='Preview Images', show_label=True, elem_id='multi_preview',
                                                           columns=3, rows=2, height=200, interactive=False)
                                multi_normal_output = gr.Image(label='Preprocessed First Image', image_mode='RGBA', type='pil',
                                                               elem_id='multi-preview-image', buttons=['download', 'fullscreen'])
                        with gr.Row():
                            multi_gen_btn = gr.Button('Generate 3D from Multiple Images', size='lg', variant='primary',
                                                      elem_id='multi-generate-btn', elem_classes=btn('teal'))
                            multi_cancel_btn = gr.Button('Cancel', size='lg', variant='stop', elem_id='multi-cancel-btn',
                                                         elem_classes=btn('red'), scale=0, min_width=160)
                        multi_status = gr.HTML(idle_card, elem_id='multi-status', container=False, padding=False)
                        with gr.Row():
                            multi_auto_save_obj_cb = gr.Checkbox(value=True, label='Auto-save OBJ', info='Wavefront OBJ format')
                            multi_auto_save_glb_cb = gr.Checkbox(value=True, label='Auto-save GLB', info='Binary glTF format')
                        with gr.Row():
                            multi_auto_save_ply_cb = gr.Checkbox(value=True, label='Auto-save PLY', info='Stanford Triangle format')
                            multi_auto_save_stl_cb = gr.Checkbox(value=True, label='Auto-save STL', info='Stereolithography format')

                    with gr.Tab('📦 Batch Processing'):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown('### Input Configuration')
                                batch_input_folder = gr.Textbox(
                                    label='Input Folder', placeholder='Path to folder containing images', elem_id='batch-input',
                                    info='Folder containing images to process (jpg, png, bmp, tiff, webp)')
                                batch_output_folder = gr.Textbox(
                                    label='Output Folder', placeholder='Path to save generated 3D models', elem_id='batch-output',
                                    info='Folder where generated files will be saved')
                                with gr.Row():
                                    batch_skip_existing = gr.Checkbox(
                                        value=True, label='Skip Existing Files',
                                        info='Skip images that already have generated output files')
                                    batch_use_current_settings = gr.Checkbox(
                                        value=True, label='Use Current Settings',
                                        info='Use the advanced settings from the current session')
                            with gr.Column(scale=1):
                                gr.Markdown('### Batch Control')
                                with gr.Row():
                                    batch_start_btn = gr.Button('🚀 Start Batch Processing', size='lg', variant='primary',
                                                                elem_id='batch-start-btn', elem_classes=btn('emerald'))
                                    batch_cancel_btn = gr.Button('⏹️ Cancel', size='lg', variant='stop', elem_id='batch-cancel-btn',
                                                                 elem_classes=btn('red'))
                                batch_status = gr.HTML(status_card('Ready', None, [('State', 'Idle')], 'idle'), elem_id='batch-status',
                                                       container=False, padding=False)
                                batch_results_text = gr.Textbox(label='Results Summary', value='No batch has been started yet.',
                                                                interactive=False, lines=6, elem_id='batch-results')

                    with gr.Tab('📋 Parameter Guide'):
                        gr.HTML(value=format_parameter_info_html(), show_label=False, elem_id='parameter-guide-content')

                with gr.Accordion('Advanced Settings', open=True):
                    seed = gr.Slider(-1, MAX_SEED, label='Seed', value=0, step=1, elem_id='seed',
                                     info='-1 picks a random seed; the seed used is shown after each run.')
                    gr.Markdown('#### Stage 1: Sparse Structure Generation')
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(0.0, 10.0, label='Guidance Strength', value=3, step=0.1)
                        ss_sampling_steps = gr.Slider(1, 50, label='Sampling Steps', value=50, step=1)
                    gr.Markdown('#### Stage 2: Structured Latent Generation')
                    with gr.Row():
                        slat_guidance_strength = gr.Slider(0.0, 10.0, label='Guidance Strength', value=3.0, step=0.1)
                        slat_sampling_steps = gr.Slider(1, 50, label='Sampling Steps', value=6, step=1)

                    gr.Markdown('#### Mesh Simplification Settings')
                    poly_count_slider = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.5, step=0.05, label='Polygon Count Percentage',
                        info='Controls polygon retention post-simplification (e.g., 0.5 = 50%). Higher values = more detail, larger files.')

                    gr.Markdown('#### UV Unwrapping (xatlas) Settings')
                    with gr.Row():
                        xatlas_max_cost_slider = gr.Slider(
                            minimum=1.0, maximum=10.0, value=8.0, step=0.1, label='Max Chart Cost',
                            info='xatlas: Higher values allow more stretch/distortion within a UV chart.')
                        xatlas_normal_seam_weight_slider = gr.Slider(
                            minimum=0.1, maximum=5.0, value=1.0, step=0.1, label='Normal Seam Weight',
                            info='xatlas: Lower values reduce UV seams based on normal changes.')
                    with gr.Row():
                        xatlas_resolution_slider = gr.Slider(
                            minimum=256, maximum=4096, value=1024, step=256, label='UV Atlas Resolution',
                            info='xatlas: Resolution of the UV texture atlas (e.g., 1024 for 1K).')
                        xatlas_padding_slider = gr.Slider(
                            minimum=0, maximum=16, value=2, step=1, label='UV Chart Padding',
                            info='xatlas: Padding in pixels between UV charts in the atlas.')

                    gr.Markdown('#### Normal Map Generation Settings')
                    with gr.Row():
                        normal_map_resolution_slider = gr.Slider(
                            minimum=256, maximum=1024, value=768, step=128, label='Normal Map Resolution',
                            info='Target resolution for the generated normal map.')
                        normal_match_input_res_checkbox = gr.Checkbox(
                            value=True, label='Match Input Resolution for Normals',
                            info="If input image is smaller than 'Normal Map Resolution', use its resolution.")

                with gr.Accordion('Performance', open=False):
                    memory_mode = gr.Radio(
                        choices=['auto', 'keep', 'offload'], value=processing_core.memory.mode if processing_core else 'auto',
                        label='Model memory',
                        info='auto: keep the models on the GPU between generations when it has 20 GB or more (fastest), '
                             'otherwise move them to the CPU after each stage. keep: always on the GPU. offload: always move them away.')
                    memory_status = gr.Textbox(label='GPU memory', value=startup_memory, interactive=False)
                    with gr.Row():
                        free_memory_btn = gr.Button('Free GPU memory', elem_classes=btn('amber'), elem_id='free-gpu-btn')
                        refresh_memory_btn = gr.Button('Refresh', elem_classes=btn('slate'))
                    mesh_jobs_slider = gr.Slider(
                        1, 8, value=batch_processor.mesh_jobs if batch_processor else 2, step=1, label='Batch: meshes unwrapped at the same time',
                        info='While the GPU works on the next image, up to this many finished meshes are decimated and UV-unwrapped '
                             'in separate processes (about 1 GB and 3 CPU cores each).')

            with gr.Column(scale=1):
                with gr.Row():
                    display_mode_dropdown = gr.Dropdown(choices=DISPLAY_MODES, value='solid', label='Display Mode',
                                                        info='Choose how to display the 3D model', scale=2)
                    display_mode_btn = gr.Button('Update Display', elem_classes=btn('sky'), scale=1)
                model_output = gr.Model3D(label=MODEL_LABEL, height=768, elem_id='model-viewer')
                export_format = gr.Dropdown(choices=EXPORT_FORMATS, value='glb', label='File Format')
                download_btn = gr.DownloadButton(label='Download 3D Mesh', interactive=False, elem_classes=btn('blue'),
                                                 elem_id='download-btn')
                open_folder_btn = gr.Button('📁 Open Outputs Folder', elem_classes=btn('indigo'), elem_id='open-outputs-btn')

                gr.Markdown('#### 📋 Preset Management')
                with gr.Row():
                    preset_dropdown = gr.Dropdown(choices=['Default'], value='Default', label='Current Preset',
                                                  info='Select or load a saved preset', interactive=True, elem_id='preset-dropdown')
                    preset_status_text = gr.Textbox(value='📋 Presets: Ready', label='Status', interactive=False, scale=2)
                with gr.Row():
                    with gr.Column(scale=2):
                        preset_name_input = gr.Textbox(label='New Preset Name', placeholder='Enter preset name...',
                                                       info='Name for saving current settings')
                    with gr.Column(scale=3):
                        preset_description_input = gr.Textbox(label='Description (Optional)', placeholder='Describe this preset...',
                                                              info='Optional description for the preset')
                with gr.Row():
                    save_preset_btn = gr.Button('💾 Save Preset', elem_classes=btn('green'), scale=1)
                    load_preset_btn = gr.Button('📂 Load Preset', elem_classes=btn('sky'), scale=1)
                    delete_preset_btn = gr.Button('🗑️ Delete', elem_classes=btn('crimson'), scale=1)
                preset_message_text = gr.Textbox(value='Ready to manage presets', label='Messages', interactive=False, lines=2)

                gr.Examples(examples=single_example_files, inputs=image_prompt, examples_per_page=96, label='Example images')
                if multi_example_sets:
                    gr.Examples(examples=[[paths] for paths in multi_example_sets], inputs=multi_image_inputs,
                                label='Multi-view Example Sets')

        # ---- the settings, in the order of the presets
        settings = [seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, poly_count_slider,
                    xatlas_max_cost_slider, xatlas_normal_seam_weight_slider, xatlas_resolution_slider, xatlas_padding_slider,
                    normal_map_resolution_slider, normal_match_input_res_checkbox]
        single_saves = [auto_save_obj_cb, auto_save_glb_cb, auto_save_ply_cb, auto_save_stl_cb]
        multi_saves = [multi_auto_save_obj_cb, multi_auto_save_glb_cb, multi_auto_save_ply_cb, multi_auto_save_stl_cb]
        preset_values = settings + single_saves
        generation_outputs = [normal_output, model_output, download_btn, single_status, model_state]
        # The status card is where progress and the queue position show; the viewers update silently
        progress_args = dict(show_progress='minimal', concurrency_id=GPU, concurrency_limit=1)

        # ---- generation
        # not in the GPU group: it must work while a generation runs (then on the CPU, see preprocess_image)
        image_prompt.upload(preprocess_image, inputs=[image_prompt], outputs=[image_prompt], api_name='preprocess_image')
        gen_shape_btn.click(
            generate_3d_with_cancellation,
            inputs=[image_prompt] + settings + single_saves + [export_format, display_mode_dropdown, model_state],
            outputs=generation_outputs, api_name='generate_3d_with_cancellation', show_progress_on=[single_status], **progress_args)
        multi_image_inputs.change(show_multi_images, inputs=[multi_image_inputs], outputs=[multi_preview, multi_status],
                                  api_visibility='private', show_progress='hidden')
        multi_gen_btn.click(
            generate_3d_multi_with_cancellation,
            inputs=[multi_image_inputs, multi_mode] + settings + multi_saves + [export_format, display_mode_dropdown, model_state],
            outputs=[multi_normal_output, model_output, download_btn, multi_status, model_state],
            api_name='generate_3d_multi_with_cancellation', show_progress_on=[multi_status], **progress_args)

        # ---- cancel: queued without a limit and outside the GPU group, so it runs at once while the GPU is busy
        # (a queue=False event could not show its toast message)
        for cancel in (single_cancel_btn, multi_cancel_btn, batch_cancel_btn):
            cancel.click(cancel_processing_ui, concurrency_limit=None, show_progress='hidden', trigger_mode='multiple',
                         api_name='cancel_processing_ui' if cancel is batch_cancel_btn else None,
                         api_visibility='public' if cancel is batch_cancel_btn else 'private')

        # ---- batch
        batch_start_btn.click(
            start_batch_processing_ui,
            inputs=[batch_input_folder, batch_output_folder, batch_skip_existing, batch_use_current_settings] + preset_values,
            outputs=[batch_status, batch_results_text, batch_start_btn], api_name='start_batch_processing_ui',
            show_progress_on=[batch_status], **progress_args)

        # ---- viewer and download
        # The viewer applies a display mode only while it loads a model, so a change is two steps: clear it, then load the
        # mesh again. They are two events: a single generator that yields both leaves the viewer empty.
        def clear_viewer(mode):
            return model_viewer(None, mode)

        def reload_viewer(model_path, mode):
            time.sleep(0.1)  # lets the page process the clear first
            return model_viewer(model_path, mode)

        def change_format(model_path, fmt):
            return download_button(prepare_download_file(model_path, fmt), fmt) if model_path else download_button(None, fmt)

        quiet = dict(api_visibility='private', show_progress='hidden', concurrency_limit=None)
        for trigger in (display_mode_btn.click, display_mode_dropdown.input):
            trigger(clear_viewer, inputs=[display_mode_dropdown], outputs=[model_output], **quiet).then(
                reload_viewer, inputs=[model_state, display_mode_dropdown], outputs=[model_output], **quiet)
        export_format.input(change_format, inputs=[model_state, export_format], outputs=[download_btn], api_visibility='private',
                            show_progress='hidden')
        open_folder_btn.click(open_outputs_folder_ui, concurrency_limit=None, api_visibility='private', show_progress='hidden')

        # ---- performance
        memory_mode.input(set_memory_mode, inputs=[memory_mode], outputs=[memory_status], api_visibility='private',
                          show_progress='hidden')
        free_memory_btn.click(free_gpu_memory_ui, outputs=[memory_status], api_visibility='private', show_progress='hidden')
        refresh_memory_btn.click(memory_status_text, outputs=[memory_status], api_visibility='private', show_progress='hidden',
                                 concurrency_limit=None)
        mesh_jobs_slider.release(set_mesh_jobs, inputs=[mesh_jobs_slider], api_visibility='private', show_progress='hidden',
                                 concurrency_limit=None)

        # ---- presets
        preset_outputs = [preset_dropdown, preset_message_text, preset_status_text]
        save_preset_btn.click(save_preset_ui_handler, inputs=[preset_name_input, preset_description_input] + preset_values,
                              outputs=preset_outputs + [preset_name_input, preset_description_input], api_visibility='private',
                              show_progress='hidden')
        load_preset_btn.click(load_preset_ui_handler, inputs=[preset_dropdown], outputs=preset_outputs + preset_values,
                              api_visibility='private', show_progress='hidden')
        delete_preset_btn.click(delete_preset_ui_handler, inputs=[preset_dropdown], outputs=preset_outputs, api_visibility='private',
                                show_progress='hidden')
        preset_dropdown.input(load_preset_ui_handler, inputs=[preset_dropdown], outputs=preset_outputs + preset_values,
                              api_visibility='private', show_progress='hidden')
        demo.load(initialize_preset_ui, outputs=preset_outputs + preset_values, api_visibility='private', show_progress='hidden')
        demo.load(memory_status_text, outputs=[memory_status], api_visibility='private', show_progress='hidden')

    demo.queue(default_concurrency_limit=1, max_size=32)
    return demo


# ----------------------------------------------------------------------------------------------------
# Start
# ----------------------------------------------------------------------------------------------------
def cleanup_processing_system():
    """Stop running work and release the GPU when the app exits."""
    try:
        if cancellation_manager:
            cancellation_manager.request_cancellation('System shutdown')
        if processing_core is not None:
            processing_core.memory.free_gpu_memory(quiet=True)
    except Exception as e:
        print(f'Error during cleanup: {e}')


def initialize_processing_system(model_memory: Optional[str] = None, mesh_jobs: Optional[int] = None):
    """Load the pipeline (to the CPU) and create the processing core, the batch processor and the cancellation manager."""
    global hi3dgen_pipeline, processing_core, batch_processor, cancellation_manager
    started = time.time()
    print('Loading the Hi3DGen pipeline (CPU)...', flush=True)
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(os.path.join(WEIGHTS_DIR, 'models--Stable-X--trellis-normal-v0-1'))
    hi3dgen_pipeline.cpu()
    print(f'Hi3DGen pipeline loaded in {time.time() - started:.1f}s.', flush=True)

    os.makedirs(TMP_DIR, exist_ok=True)
    processing_core = ProcessingCore(hi3dgen_pipeline=hi3dgen_pipeline, weights_dir=WEIGHTS_DIR, tmp_dir=TMP_DIR, max_seed=MAX_SEED)
    if model_memory:
        processing_core.memory.set_mode(model_memory)
    batch_processor = create_batch_processor(processing_core, mesh_jobs)
    cancellation_manager = get_cancellation_manager()
    print(processing_core.memory.status())
    print(f'Meshes unwrapped at the same time in a batch: {batch_processor.mesh_jobs}')
    if SYSTEM_STATUS_AVAILABLE:
        print('\n' + '=' * 60)
        print_system_status(processing_core=processing_core, batch_processor=batch_processor, hi3dgen_pipeline=hi3dgen_pipeline,
                            weights_dir=WEIGHTS_DIR, tmp_dir=TMP_DIR)
        print('=' * 60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run the Hi3DGen Gradio App.')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument('--host', default=None, help='Address to listen on (0.0.0.0 for the local network)')
    parser.add_argument('--port', type=int, default=None, help='Port (default: the first free one from 7860)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser')
    parser.add_argument('--model-memory', choices=['auto', 'keep', 'offload'], default=None,
                        help='GPU memory policy (default: HI3DGEN_MODEL_MEMORY or auto)')
    parser.add_argument('--mesh-jobs', type=int, default=None, help='Batch: meshes unwrapped at the same time')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    started = time.time()
    args = parse_args(argv)
    os.chdir(ROOT)  # the presets and the outputs live next to the app
    print(f'Hi3DGen SECourses App {APP_VERSION}', flush=True)
    verify_weights(WEIGHTS_DIR)

    try:
        initialize_processing_system(args.model_memory, args.mesh_jobs)
    except Exception as e:
        print(f'Error initializing the processing system: {e}')
        traceback.print_exc()
        print('Some features may not work properly.')

    import atexit
    atexit.register(cleanup_processing_system)

    print('Building the interface...', flush=True)
    demo = build_ui()
    address = {}
    if args.host:
        address['server_name'] = args.host
    if args.port:
        address['server_port'] = args.port
    demo.launch(
        **address, share=args.share, inbrowser=not args.no_browser, show_error=True, favicon_path=str(ui_theme.FAVICON_PATH),
        theme=ui_theme.app_theme(), css=ui_theme.APP_CSS, head=ui_theme.APP_HEAD,
        # Gradio's run history writes every call's inputs and outputs to the browser's local storage
        run_history=False, footer_links=['api', 'settings'],
        allowed_paths=[str(ROOT / 'assets'), TMP_DIR, str(ROOT / 'outputs')],
        prevent_thread_lock=True,
    )
    print(f'Hi3DGen is ready at {demo.local_url} (startup {time.time() - started:.1f}s)', flush=True)
    demo.block_thread()
    return 0


if __name__ == '__main__':
    sys.exit(main())
