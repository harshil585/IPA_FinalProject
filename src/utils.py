"""
utils.py
────────
Shared image-processing utilities for the SAM-CLIP pipeline.
"""

import cv2
import numpy as np
from pathlib import Path


# ── Mask merging ──────────────────────────────────────────────────────────────

def merge_similar_masks(
    masks: list[dict],
    min_area: int = 2000,
    overlap_thresh: int = 50,
) -> list[dict]:
    """
    Combine overlapping masks when shared pixels exceed *overlap_thresh*.

    Reduces fragmentation by merging masks with ``area > min_area`` that
    overlap each other by more than *overlap_thresh* pixels.

    Parameters
    ----------
    masks : list[dict]
        SAM mask dictionaries (must contain keys ``'segmentation'`` and
        ``'area'``).
    min_area : int
        Masks with fewer pixels than this are discarded before merging.
    overlap_thresh : int
        Minimum pixel overlap required to merge two masks.

    Returns
    -------
    list[dict]
        Merged mask dictionaries with updated ``'segmentation'`` and
        ``'area'`` keys.
    """
    merged: list[np.ndarray] = []

    for m in masks:
        if m["area"] < min_area:
            continue

        current_mask = m["segmentation"]
        merged_flag = False

        for i in range(len(merged)):
            overlap = np.logical_and(merged[i], current_mask)
            if np.sum(overlap) > overlap_thresh:
                merged[i] = np.logical_or(merged[i], current_mask)
                merged_flag = True
                break

        if not merged_flag:
            merged.append(current_mask)

    return [
        {"segmentation": m, "area": int(np.sum(m))}
        for m in merged
    ]


# ── Image I/O helpers ─────────────────────────────────────────────────────────

def load_image_rgb(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return it as an **RGB** NumPy array.

    Raises
    ------
    ValueError
        If the image cannot be read (file missing or unsupported format).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image_rgb(image_rgb: np.ndarray, output_path: str | Path) -> None:
    """Save an RGB NumPy array to disk (converts to BGR for OpenCV)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


# ── Colour generation ─────────────────────────────────────────────────────────

def generate_colors(n: int) -> list[list[int]]:
    """
    Generate *n* visually distinct colours in HSV space, returned as RGB triples.

    Distributes hues evenly across the HSV colour wheel at full saturation and
    value for maximum colour separation between adjacent segments.
    """
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(rgb.tolist())
    return colors


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(
    image_rgb: np.ndarray,
    save_steps: bool = False,
    step_dir: Path | None = None,
) -> np.ndarray:
    """
    Clean an RGB image for SAM input:

    1. Convert to grayscale.
    2. Stretch contrast (``cv2.normalize``).
    3. Smooth noise with a 5×5 Gaussian blur.
    4. Convert back to 3-channel (required by SAM).

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image in RGB colour space.
    save_steps : bool
        When *True*, each preprocessing step is written to *step_dir*.
    step_dir : Path, optional
        Directory used when *save_steps* is True.

    Returns
    -------
    np.ndarray
        3-channel preprocessed image in RGB colour space.
    """
    if save_steps and step_dir is not None:
        step_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(step_dir / "pp_00_original.png"),
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        )

    # Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if save_steps and step_dir is not None:
        cv2.imwrite(str(step_dir / "pp_01_grayscale.png"), gray)

    # Contrast normalisation
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if save_steps and step_dir is not None:
        cv2.imwrite(str(step_dir / "pp_02_normalized.png"), normalized)

    # Gaussian blur
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    if save_steps and step_dir is not None:
        cv2.imwrite(str(step_dir / "pp_03_blurred.png"), blurred)

    # Back to 3-channel
    final = cv2.merge([blurred, blurred, blurred])
    if save_steps and step_dir is not None:
        cv2.imwrite(
            str(step_dir / "pp_04_final_3channel.png"),
            cv2.cvtColor(final, cv2.COLOR_RGB2BGR),
        )

    return final
