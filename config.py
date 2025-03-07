import os
from pathlib import Path

# Create cache directory if it doesn't exist
CACHE_DIR = Path.home() / '.cache' / 'huggingface'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_ID = "nitrosocke/Nitro-Diffusion"# Lighter model better suited for CPU
DEFAULT_STEPS = 20
MAX_STEPS = 50