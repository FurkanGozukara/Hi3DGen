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
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.
import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'SLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint (local only).
    Does NOT download from HuggingFace - all models must be pre-downloaded.

    Args:
        path: The local path to the checkpoint.
              NOTE: config file and model file should be named f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    
    config_file = f"{path}.json"
    model_file = f"{path}.safetensors"
    
    # Check if local files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Model config file not found: {config_file}\n"
            f"Please ensure models are downloaded using DownloadModels.py"
        )
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model weights file not found: {model_file}\n"
            f"Please ensure models are downloaded using DownloadModels.py"
        )

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_vae import SLatEncoder, SLatGaussianDecoder, SLatRadianceFieldDecoder, SLatMeshDecoder
    from .structured_latent_flow import SLatFlowModel
