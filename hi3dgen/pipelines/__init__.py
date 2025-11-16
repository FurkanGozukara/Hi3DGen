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
# SPDX-License-Identifier: MIT
from . import samplers
from .hi3dgen import Hi3DGenPipeline

def from_pretrained(path: str):
    """
    Load a pipeline from a local model folder.
    Does NOT download from HuggingFace - all models must be pre-downloaded.

    Args:
        path: The local path to the model directory containing pipeline.json.
    """
    import os
    import json
    
    config_file = f"{path}/pipeline.json"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Pipeline configuration not found at: {config_file}\n"
            f"Expected a local path with pipeline.json file.\n"
            f"Please ensure models are downloaded using DownloadModels.py"
        )

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
