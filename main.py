# ============================================================
# SAM-CLIP Image Segmentation Pipeline
# Entry Point: main.py
# ============================================================

import sys
import argparse
from pathlib import Path
from src.config import setup_environment
from src.segmenter import SAMSegmenter
from src.downloader import download_sam_weights
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM-CLIP Image Segmentation Pipeline — zero-shot semantic labeling"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input image. Defaults to first image found in input/ folder."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/segmented_output.png",
        help="Path for the merged segmentation output image."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant to use."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/sam_vit_h_4b8939.pth",
        help="Path to the SAM model checkpoint file."
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Number of points per side for mask generation grid."
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.94,
        help="IoU prediction quality threshold."
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.96,
        help="Stability score threshold for mask filtering."
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=800,
        help="Minimum mask region area in pixels."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["person", "car", "tree", "building", "road", "sky", "animal", "water", "background"],
        help="Space-separated list of semantic labels for CLIP classification."
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP semantic labeling step (faster, segmentation only)."
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Skip saving intermediate step images."
    )
    return parser.parse_args()


def resolve_input_image(input_arg: str | None) -> str:
    """Resolve the input image path. If not specified, pick first image in input/."""
    if input_arg is not None:
        p = Path(input_arg)
        if not p.exists():
            print(f"[ERROR] Input image not found: {input_arg}")
            sys.exit(1)
        return str(p)

    # Auto-discover from input/ folder
    input_dir = Path("input")
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        found = sorted(input_dir.glob(ext))
        if found:
            print(f"[INFO] No --input specified. Using: {found[0]}")
            return str(found[0])

    print("[ERROR] No input image found in input/ folder. "
          "Place an image there or pass --input <path>.")
    sys.exit(1)


def main():
    args = parse_args()

    # ── 1. Environment setup (cache dirs, env vars) ──────────────────────────
    setup_environment()

    # ── 2. Ensure SAM weights exist ──────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[INFO] SAM checkpoint not found at {checkpoint_path}. Attempting download…")
        download_sam_weights(model_type=args.model_type, dest=checkpoint_path)

    # ── 3. Resolve input image ────────────────────────────────────────────────
    input_image = resolve_input_image(args.input)

    # ── 4. Ensure output directory exists ────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 5. Initialize SAM segmenter ───────────────────────────────────────────
    segmenter = SAMSegmenter(
        model_type=args.model_type,
        checkpoint_path=str(checkpoint_path),
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_area=args.min_mask_area,
    )

    # ── 6. Run segmentation pipeline ──────────────────────────────────────────
    merged_masks, output_image, original_rgb = segmenter.segment_image(
        image_path=input_image,
        output_path=str(output_path),
        merge_min_area=2000,
        merge_overlap_thresh=50,
        save_intermediate=not args.no_intermediate,
    )

    if not merged_masks:
        print("[WARN] No masks produced. Exiting.")
        return

    # ── 7. CLIP semantic labeling ──────────────────────────────────────────────
    if not args.skip_clip:
        labeled_masks, labeled_output = segmenter.semantic_label_masks(
            image=original_rgb,
            masks=merged_masks,
            labels=args.labels,
        )

        labeled_out_path = output_path.parent / (output_path.stem + "_semantic_labeled.png")
        cv2.imwrite(str(labeled_out_path), cv2.cvtColor(labeled_output, cv2.COLOR_RGB2BGR))
        print(f"\n[OUTPUT] Semantic labeled image saved: {labeled_out_path}")

        # Print results table
        print("\n" + "=" * 52)
        print("SEMANTIC LABELING RESULTS".center(52))
        print("=" * 52)
        print(f"{'Mask':<6} {'Label':<14} {'Confidence':<12} {'Area':<10}")
        print("-" * 52)
        for i, m in enumerate(labeled_masks):
            label = m.get("label", "N/A")
            conf  = m.get("confidence", 0.0)
            area  = m.get("area", 0)
            print(f"{i:<6} {label:<14} {conf:<12.2f} {area:<10}")
        print("=" * 52)
    else:
        print("[INFO] CLIP labeling skipped (--skip-clip).")

    print(f"\n[DONE] Segmentation complete. Outputs written to: {output_path.parent}/")


if __name__ == "__main__":
    main()
