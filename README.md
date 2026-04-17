# SAM-CLIP Image Segmentation Pipeline

> Zero-shot semantic image segmentation powered by Facebook's **Segment Anything Model (SAM)** and OpenAI's **CLIP** — running entirely locally in VS Code, no Colab required.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [CLI Arguments](#cli-arguments)
6. [Example Input / Output](#example-input--output)
7. [Troubleshooting](#troubleshooting)

---

## Project Description

This pipeline takes any input image and performs:

| Step | Description |
|------|-------------|
| **Preprocessing** | Grayscale conversion → contrast normalisation → Gaussian blur |
| **SAM mask generation** | Automatic mask generation using SAM (ViT-H, ViT-L, or ViT-B) |
| **Mask filtering** | Dynamic IoU threshold + area-size filter |
| **Morphological refinement** | Morphological closing to smooth mask boundaries |
| **Mask merging** | Overlapping masks are combined to reduce fragmentation |
| **CLIP labeling** | Zero-shot classification assigns semantic labels to each region |

Outputs include coloured segmentation overlays, per-step intermediate images, and a console results table showing `label`, `confidence`, and `area` for every detected segment.

---

## Directory Structure

```
Image_Project/
├── main.py                   # Pipeline entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source modules
│   ├── __init__.py
│   ├── config.py             # Environment & path setup
│   ├── downloader.py         # SAM weights downloader (Python, no wget)
│   ├── segmenter.py          # SAMSegmenter class (SAM + CLIP)
│   └── utils.py              # Shared image-processing helpers
│
├── input/                    # Place your input images here
├── outputs/                  # Segmentation outputs written here
├── models/                   # SAM model checkpoint stored here
└── cache/                    # HuggingFace + PyTorch model caches
    ├── hf/
    └── torch/
```

---

## Installation

### Prerequisites

- Python **3.10 or 3.11** (recommended)
- [Git](https://git-scm.com/) installed and on PATH
- NVIDIA GPU with CUDA ≥ 11.8 (optional but strongly recommended for speed)

### Step-by-step

```powershell
# 1. Open the project folder in VS Code terminal (PowerShell or CMD)
cd d:\Codes\Image_Project

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1      # PowerShell
# OR
.\.venv\Scripts\activate.bat      # CMD

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. (GPU only) Install PyTorch with CUDA support FIRST
#    Visit https://pytorch.org/get-started/locally/ to get the right command.
#    Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** The SAM model weights (~2.5 GB for ViT-H) are downloaded automatically on first run into the `models/` directory.  A stable internet connection is required for the first run only.

---

## How to Run

```powershell
# Make sure the virtual environment is activated first
.\.venv\Scripts\Activate.ps1

# Place your image in the input/ folder, then run:
python main.py

# Or specify a custom input image
python main.py --input input/my_photo.jpg

# Skip CLIP labeling for a faster segmentation-only run
python main.py --skip-clip

# Use a lighter model (faster, less accurate)
python main.py --model-type vit_b --checkpoint models/sam_vit_b_01ec64.pth
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | auto-discover from `input/` | Path to input image |
| `--output`, `-o` | `outputs/segmented_output.png` | Path for merged segmentation output |
| `--model-type` | `vit_h` | SAM model variant: `vit_h`, `vit_l`, `vit_b` |
| `--checkpoint` | `models/sam_vit_h_4b8939.pth` | Path to SAM checkpoint `.pth` file |
| `--points-per-side` | `32` | Grid density for mask generation |
| `--pred-iou-thresh` | `0.94` | IoU quality threshold |
| `--stability-score-thresh` | `0.96` | Stability score threshold |
| `--min-mask-area` | `800` | Minimum mask area in pixels |
| `--labels` | *(built-in set)* | Space-separated list of CLIP target labels |
| `--skip-clip` | `False` | Skip CLIP semantic labeling |
| `--no-intermediate` | `False` | Skip saving intermediate step images |

---

## Example Input / Output

After running `python main.py --input input/street.jpg` you will find:

```
outputs/
├── segmented_output.png              ← Final merged segmentation overlay
├── segmented_output_sam_only.png     ← SAM masks before merging
├── segmented_output_semantic_labeled.png  ← With CLIP label text overlays
└── segmented_output_steps/
    ├── step_00_original.png
    ├── step_01_preprocessed.png
    ├── step_02_raw_SAM_masks.png
    ├── step_03_after_area_filter.png
    ├── step_04_before_merge_final.png
    └── step_05_after_merge_final.png
```

**Console output example:**

```
====================================================
        SEMANTIC LABELING RESULTS
====================================================
Mask   Label          Confidence   Area
----------------------------------------------------
0      person         0.82         18432
1      road           0.71         54210
2      building       0.65         32100
3      sky            0.90         41000
====================================================
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: SAM checkpoint not found` | Internet needed on first run; checkpoint auto-downloads to `models/`. |
| `CUDA out of memory` | Use `--model-type vit_b` or add `--points-per-side 16`. |
| `No masks detected` | Lower `--pred-iou-thresh` (e.g. `0.80`) or `--stability-score-thresh` (e.g. `0.85`). |
| Slow on CPU | Install PyTorch with CUDA support ([guide](https://pytorch.org/get-started/locally/)). |
| `transformers` import error | Run `pip install transformers pillow` inside the activated venv. |
