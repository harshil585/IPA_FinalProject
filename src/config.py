"""
config.py
─────────
Environment and directory setup for the SAM-CLIP pipeline.
All paths are relative to the project root so the project is portable.
"""

import os
from pathlib import Path

# ── Project-root-relative directory layout ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRS = {
    "input":   PROJECT_ROOT / "input",
    "outputs": PROJECT_ROOT / "outputs",
    "models":  PROJECT_ROOT / "models",
    "cache":   PROJECT_ROOT / "cache",
    "hf_cache": PROJECT_ROOT / "cache" / "hf",
    "torch_cache": PROJECT_ROOT / "cache" / "torch",
}

# SAM checkpoint URLs keyed by model type
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def setup_environment() -> None:
    """
    Create required directories and configure environment variables for
    HuggingFace Transformers and PyTorch caches to use local paths instead
    of system defaults.
    """
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)

    # Point HuggingFace and PyTorch caches to local project directories
    os.environ["TRANSFORMERS_CACHE"] = str(DIRS["hf_cache"])
    os.environ["HF_HOME"]            = str(DIRS["hf_cache"])
    os.environ["TORCH_HOME"]         = str(DIRS["torch_cache"])

    print("[CONFIG] Environment configured.")
    print(f"         HF cache  -> {DIRS['hf_cache']}")
    print(f"         Torch cache -> {DIRS['torch_cache']}\n")
