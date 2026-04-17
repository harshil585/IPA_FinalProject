"""
segmenter.py
────────────
SAMSegmenter class — wraps Facebook's Segment Anything Model (SAM) and
adds zero-shot CLIP semantic labeling on top of the generated masks.

All Colab-specific code has been removed.  Paths are relative to the
project root and are resolved at runtime.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path

# segment_anything is imported lazily inside __init__ so the project can
# be imported / configured before the package is installed.

from src.utils import (
    merge_similar_masks,
    generate_colors,
    preprocess_image,
    load_image_rgb,
    save_image_rgb,
)


class SAMSegmenter:
    """
    Main segmentation class that loads SAM and provides a full pipeline:

    1. Image preprocessing  (grayscale -> normalise -> blur -> 3-channel)
    2. Automatic mask generation via SAM
    3. Quality / size filtering and morphological mask refinement
    4. Mask merging to reduce fragmentation
    5. Optional zero-shot CLIP semantic labeling
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = "models/sam_vit_h_4b8939.pth",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.93,
        stability_score_thresh: float = 0.96,
        min_mask_area: int = 800,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM] Using device: {self.device}")

        # Lazy import — only fails here if the package isn't installed yet.
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment_anything is not installed.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}\n"
                "Run: python main.py   (weights are downloaded automatically on first run)."
            )

        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device)
        sam.eval()

        # Automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=0,
            min_mask_region_area=min_mask_area,
        )

        # CLIP is loaded lazily on first call to semantic_label_masks
        self.clip_model = None
        self.clip_processor = None

        print("SAM initialised successfully.\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _create_colored_output(
        self, image: np.ndarray, masks: list[dict]
    ) -> np.ndarray:
        """
        Assign a unique colour to each mask and blend with the original image
        (60 % image + 40 % colour overlay).
        """
        h, w = image.shape[:2]
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colors = generate_colors(len(masks))

        for idx, mask_data in enumerate(masks):
            colored_mask[mask_data["segmentation"]] = colors[idx]

        return cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)

    def _save_intermediate(
        self,
        image_rgb: np.ndarray,
        masks: list[dict],
        step_name: str,
        output_dir: str | Path,
    ) -> str:
        """Save a coloured-mask overlay image for a pipeline step."""
        output = self._create_colored_output(image_rgb, masks)
        step_path = Path(output_dir) / f"step_{step_name}.png"
        save_image_rgb(output, step_path)
        print(f"  -> Step '{step_name}': {len(masks)} masks | saved: {step_path}")
        return str(step_path)

    def _visualize_preprocessing(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        output_dir: str | Path,
    ) -> str:
        """Save a side-by-side comparison of original vs preprocessed image."""
        h, w = original.shape[:2]
        h_pp, w_pp = preprocessed.shape[:2]
        if (h, w) != (h_pp, w_pp):
            preprocessed = cv2.resize(preprocessed, (w, h))
        comparison = np.hstack([original, preprocessed])
        comp_path = Path(output_dir) / "step_01_preprocessing_comparison.png"
        save_image_rgb(comparison, comp_path)
        print(f"  -> Preprocessing comparison saved: {comp_path}")
        return str(comp_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def segment_image(
        self,
        image_path: str,
        output_path: str = "outputs/segmented_output.png",
        show_top_n: int | None = None,           # reserved for future use
        merge_min_area: int = 2000,
        merge_overlap_thresh: int = 50,
        save_intermediate: bool = True,
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """
        Full segmentation pipeline.

        Parameters
        ----------
        image_path : str
            Path to the source image.
        output_path : str
            Destination for the merged-mask visualisation.
        merge_min_area : int
            Minimum mask area used during the merging step.
        merge_overlap_thresh : int
            Pixel-overlap threshold for merging two masks.
        save_intermediate : bool
            When *True*, six intermediate PNG images are written to a
            ``<output_stem>_steps/`` subdirectory.

        Returns
        -------
        merged_masks : list[dict]
        output_image : np.ndarray  (RGB)
        original_rgb : np.ndarray  (RGB)
        """
        # ── Load image ────────────────────────────────────────────────────────
        image_rgb = load_image_rgb(image_path)
        h, w = image_rgb.shape[:2]
        total_pixels = h * w

        print(f"\n{'='*60}")
        print(f"Processing : {image_path}")
        print(f"Image size : {w} × {h}")
        print(f"{'='*60}\n")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        base_name = output_path_obj.stem
        output_dir = output_path_obj.parent / f"{base_name}_steps"

        if save_intermediate:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(output_dir / "step_00_original.png"),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            )
            print(f"Step 00 – Original image saved: {output_dir / 'step_00_original.png'}")

        # ── Resize safety (prevents OOM on large images with vit_h) ──────────
        MAX_SIZE = 1000
        if max(h, w) > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
            h, w = image_rgb.shape[:2]
            total_pixels = h * w

        # ── Preprocessing ─────────────────────────────────────────────────────
        print("\n[PREPROCESSING] Cleaning image …")
        preprocessed = preprocess_image(image_rgb)
        print("Applied grayscale, contrast normalisation, and Gaussian blur")

        if save_intermediate:
            cv2.imwrite(
                str(output_dir / "step_01_preprocessed.png"),
                cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR),
            )
            print(f"Step 01 – Preprocessed image saved: {output_dir / 'step_01_preprocessed.png'}")

        # ── SAM mask generation ───────────────────────────────────────────────
        print("\n[STEP 2] Generating SAM masks …")
        with torch.no_grad():
            masks = self.mask_generator.generate(preprocessed)
        total_before = len(masks)
        print(f"Generated {total_before} raw masks")

        if save_intermediate and total_before > 0:
            self._save_intermediate(preprocessed, masks, "02_raw_SAM_masks", str(output_dir))

        if total_before == 0:
            print("No masks detected.")
            return [], image_rgb, image_rgb

        # ── Filter 1: Remove low-quality masks (dynamic IoU threshold) ────────
        print("\n[STEP 2] Filtering by IoU quality score …")
        mean_iou = np.mean([m["predicted_iou"] for m in masks])
        dynamic_thresh = mean_iou * 0.9
        masks = [m for m in masks if m["predicted_iou"] > dynamic_thresh]
        print(f"Kept {len(masks)} masks (removed {total_before - len(masks)} low-quality)")

        # ── Filter 2: Morphological closing ───────────────────────────────────
        print("\n[STEP 3] Applying morphological refinement …")
        kernel = np.ones((5, 5), np.uint8)
        for m in masks:
            mask_uint8 = m["segmentation"].astype(np.uint8) * 255
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            refined = mask_uint8 > 0
            m["segmentation"] = refined
            m["area"] = int(np.sum(refined))
        print(f"Applied morphological closing to {len(masks)} masks")

        # ── Filter 3: Area-size filter ────────────────────────────────────────
        print("\n[STEP 4] Filtering by area size …")
        before_area = len(masks)
        masks = [m for m in masks if 1000 < m["area"] < 0.6 * total_pixels]
        print(f"Kept {len(masks)} masks (removed {before_area - len(masks)} by size)")

        if save_intermediate and masks:
            self._save_intermediate(image_rgb, masks, "03_after_area_filter", str(output_dir))

        # ── Merge overlapping masks ────────────────────────────────────────────
        print("\n[STEP 5] Merging overlapping masks …")
        merged_masks = merge_similar_masks(
            masks, min_area=merge_min_area, overlap_thresh=merge_overlap_thresh
        )
        total_after = len(merged_masks)
        print(f"Merged from {len(masks)} -> {total_after} masks")

        # ── Analytics ─────────────────────────────────────────────────────────
        reduction_percent = ((1 - total_after / total_before) * 100) if total_before > 0 else 0
        combined_mask = np.zeros((h, w), dtype=bool)
        for m in merged_masks:
            combined_mask |= m["segmentation"]
        covered_pixels = int(np.sum(combined_mask))
        coverage_percent = covered_pixels / total_pixels * 100
        avg_area = covered_pixels / total_after if total_after > 0 else 0
        fragmentation_index = total_after / (total_pixels / 1_000_000)

        # ── Output images ──────────────────────────────────────────────────────
        print("\n[STEP 6] Generating output visualisations …")

        # SAM-only (before merge)
        sam_output_image = self._create_colored_output(image_rgb, masks)
        sam_output_path = str(output_path).replace(".png", "_sam_only.png").replace(".jpg", "_sam_only.png")
        save_image_rgb(sam_output_image, sam_output_path)
        print(f"SAM output (before merge) saved: {sam_output_path}")

        if save_intermediate:
            self._save_intermediate(image_rgb, masks, "04_before_merge_final", str(output_dir))

        # Merged final output
        output_image = self._create_colored_output(image_rgb, merged_masks)
        save_image_rgb(output_image, output_path)
        print(f"Merged output (after merge)  saved: {output_path}")

        if save_intermediate:
            self._save_intermediate(image_rgb, merged_masks, "05_after_merge_final", str(output_dir))

        torch.cuda.empty_cache()

        # ── Print analytics ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("ANALYTICAL METRICS".center(60))
        print("=" * 60)
        print(f"Raw Segments:            {total_before}")
        print(f"Segments After Merge:    {total_after}")
        print(f"Reduction Percentage:    {reduction_percent:.2f}%")
        print(f"Image Coverage:          {coverage_percent:.2f}%")
        print(f"Average Segment Area:    {avg_area:.2f} pixels")
        print(f"Fragmentation Index:     {fragmentation_index:.2f}")
        print("=" * 60)

        if save_intermediate:
            print(f"\nAll 6 intermediate steps saved in: {output_dir}/")
            print("\n  Step Summary:")
            print("  0. Original image")
            print("  1. Preprocessed (grayscale + normalised + blurred)")
            print("  2. Raw SAM masks")
            print("  3. After area size filter")
            print("  4. Before final merging")
            print("  5. After final merging (FINAL OUTPUT)")

        return merged_masks, output_image, image_rgb

    # ── CLIP semantic labeling ────────────────────────────────────────────────

    def semantic_label_masks(
        self,
        image: np.ndarray,
        masks: list[dict],
        labels: list[str] | None = None,
    ) -> tuple[list[dict], np.ndarray]:
        """
        Zero-shot semantic classification of each segmented region using CLIP.

        Each mask's bounding-box crop is compared against *labels* using
        cosine similarity in CLIP's shared vision-language embedding space.
        The best-matching label and its confidence score are stored in the
        mask dict (``mask['label']``, ``mask['confidence']``).

        Parameters
        ----------
        image : np.ndarray
            RGB image from which masks were generated.
        masks : list[dict]
            List of SAM mask dicts (modified in-place).
        labels : list[str], optional
            Candidate semantic labels.  Falls back to a built-in default set.

        Returns
        -------
        masks : list[dict]
            Updated with ``'label'`` and ``'confidence'`` keys.
        labeled_image : np.ndarray  (RGB)
            Visualisation with label text overlaid on each segment centroid.
        """
        if labels is None:
            labels = [
                "person", "car", "tree", "building",
                "road", "sky", "animal", "water", "background",
            ]

        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image as PILImage
        except ImportError:
            print(
                "ERROR: transformers and/or Pillow are not installed.\n"
                "Run: pip install transformers pillow"
            )
            return masks, image

        # Lazy-load CLIP
        if self.clip_model is None:
            print("\n[CLIP] Loading pretrained CLIP model …")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = (
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            )
            self.clip_model.eval()

        print(f"\n[CLIP] Zero-shot classification on {len(masks)} masks | labels: {labels}")

        # Build cropped PIL images for each mask
        crops: list = []
        valid_indices: list[int] = []

        for i, m in enumerate(masks):
            mask_arr = m["segmentation"]
            coords = np.column_stack(np.where(mask_arr > 0))   # (y, x)
            if coords.size == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            crop_img  = image[y_min:y_max + 1, x_min:x_max + 1].copy()
            crop_mask = mask_arr[y_min:y_max + 1, x_min:x_max + 1]
            crop_img[~crop_mask] = 0                            # zero-out background

            crops.append(PILImage.fromarray(crop_img))
            valid_indices.append(i)

        if not crops:
            print("[CLIP] No valid masks to label.")
            return masks, image

        # Inference in batches
        batch_size = 16
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            text_inputs  = self.clip_processor(
                text=labels, return_tensors="pt", padding=True
            ).to(self.device)
            text_outputs = self.clip_model.get_text_features(**text_inputs)

            text_features = (
                text_outputs.pooler_output
                if hasattr(text_outputs, "pooler_output")
                else text_outputs
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for i in range(0, len(crops), batch_size):
                batch = crops[i : i + batch_size]
                image_inputs  = self.clip_processor(images=batch, return_tensors="pt").to(self.device)
                image_outputs = self.clip_model.get_image_features(**image_inputs)

                image_features = (
                    image_outputs.pooler_output
                    if hasattr(image_outputs, "pooler_output")
                    else image_outputs
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                probs  = logits.softmax(dim=1).cpu().numpy()
                all_probs.extend(probs)

        # Build annotated output image
        labeled_image = self._create_colored_output(image, masks)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1

        for idx, prob in zip(valid_indices, all_probs):
            best_idx   = int(np.argmax(prob))
            best_label = labels[best_idx]
            confidence = float(prob[best_idx])

            masks[idx]["label"]      = best_label
            masks[idx]["confidence"] = confidence

            # Draw label at centroid
            coords = np.column_stack(np.where(masks[idx]["segmentation"] > 0))
            if coords.size == 0:
                continue

            cy, cx = coords.mean(axis=0).astype(int)
            text   = f"{best_label} ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            x1 = max(cx - text_w // 2 - 5, 0)
            y1 = max(cy - text_h // 2 - 5, 0)
            x2 = min(cx + text_w // 2 + 5, labeled_image.shape[1] - 1)
            y2 = min(cy + text_h // 2 + 5, labeled_image.shape[0] - 1)

            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.putText(
                labeled_image, text,
                (cx - text_w // 2, cy + text_h // 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

        return masks, labeled_image
