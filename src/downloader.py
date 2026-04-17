"""
downloader.py
-------------
Utility to download SAM model weights using Python's urllib
(no wget or shell commands required).
"""

import io
import urllib.request
import sys
from pathlib import Path
from src.config import SAM_CHECKPOINT_URLS


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple CLI progress reporter for urllib.request.urlretrieve (ASCII-safe)."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = f"\r  [{bar}] {percent:5.1f}%  ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)"
        # Write as UTF-8 bytes to avoid cp1252 errors on Windows terminals
        sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.buffer.flush()
        if downloaded >= total_size:
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()


def download_sam_weights(model_type: str = "vit_h", dest: Path | None = None) -> Path:
    """
    Download the SAM checkpoint for *model_type* to *dest*.

    Parameters
    ----------
    model_type : str
        One of ``"vit_h"``, ``"vit_l"``, ``"vit_b"``.
    dest : Path, optional
        Destination file path. Defaults to ``models/<filename>``.

    Returns
    -------
    Path
        Path to the downloaded checkpoint file.

    Raises
    ------
    KeyError
        If *model_type* is not recognised.
    RuntimeError
        If the download fails.
    """
    if model_type not in SAM_CHECKPOINT_URLS:
        raise KeyError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(SAM_CHECKPOINT_URLS)}"
        )

    url = SAM_CHECKPOINT_URLS[model_type]
    filename = url.split("/")[-1]

    if dest is None:
        dest = Path("models") / filename

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[DOWNLOAD] Checkpoint already present: {dest}")
        return dest

    print(f"[DOWNLOAD] Fetching SAM {model_type} weights...")
    print(f"           URL  : {url}")
    print(f"           Dest : {dest}")

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SAM checkpoint from {url}.\n"
            f"Error: {exc}\n"
            "Please download the file manually and place it in the models/ directory."
        ) from exc

    print(f"[DOWNLOAD] SAM weights saved to: {dest}\n")
    return dest
