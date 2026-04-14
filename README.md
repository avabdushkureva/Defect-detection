# Defect Detection System for Packaged Finished Goods

A two-stage YOLOv8-based instance segmentation system for automated quality control in manufacturing logistics.

## About the Project

This system automates quality control of **packaged finished goods** at the final stage of manufacturing. It detects and classifies packaging defects (holes, open boxes, dents) using computer vision, helping manufacturers reduce financial losses associated with damaged cargo before shipment.

Unlike existing solutions focused on distribution centers, this system is designed specifically for **manufacturing logistics** – inspecting goods immediately after they are packed and ready for delivery to customers.

## System Architecture

The system follows a **two-stage instance segmentation approach**:

1. **Stage 1 – Box Segmentation**: Localizes and segments each box in the input image
2. **Stage 2 – Defect Detection**: Detects and classifies defects on cropped box regions

### Key Advantages

| Feature | Benefit |
|---------|---------|
| Background elimination | Defects are analyzed strictly within object boundaries |
| Segmentation masks | More precise defect localization compared to bounding boxes |
| Two-stage design | Improved robustness in cluttered logistics environments |
| Real-time capability | Suitable for high-throughput manufacturing lines |

## Datasets

### Dataset 1: Box Detection

| Aspect | Description |
|--------|-------------|
| **Sources** | TAMPAR, Parcel2D, AISS-CV, Google Images, Yandex.Images, conveyor video frames, self-collected data |
| **Total images** | 6,000 (including 662 negative examples without boxes) |
| **Total objects** | 10,128 boxes |
| **Annotation** | Semi-automatic clickable interactive segmentation |

### Dataset 2: Defect Detection and Classification

| Aspect | Description |
|--------|-------------|
| **Defect types** | `hole`, `open` (open box), `dent` |
| **Total images** | 3,080 (including 112 images without defects) |
| **Total defects** | 4,943 |
| **Class distribution** | Hole: 3,082 / Open: 1,361 / Dent: 500 (imbalanced) |
| **Annotation** | Semi-automatic clickable interactive segmentation |

### Data Format (YOLOv8)

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**Annotation format** (normalized polygon coordinates):

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

## Results

### Stage 1: Box Detection (YOLOv8m)

| Metric | Value |
|--------|-------|
| mAP@50 | 99.49% |
| mAP@50:95 | 98.43% |
| Segmentation loss | 0.1846 |

### Stage 2: Defect Detection (YOLOv8s)

| Metric | Value |
|--------|-------|
| mAP@50 | 99.34% |
| mAP@50:95 | 86.04% |
| Segmentation loss | 0.8766 |
| Classification loss | 0.3204 |

## Key Features

- **Semi-automatic annotation**: Clickable interactive segmentation for efficient mask generation
- **Two-stage architecture**: Isolates objects before defect analysis, reducing background interference
- **YOLOv8 support**: Modern instance segmentation models with multiple architecture sizes (n/s/m/l/x)
- **Data conversion**: Utilities for COCO ↔ YOLOv8 ↔ mask ↔ polygon conversion
- **Video processing**: Frame extraction for custom dataset creation
- **Visualization tools**: Annotation inspection, mask rendering, and result analysis
- **Training monitoring**: Loss curves and mAP metrics across epochs
- **Instance segmentation**: Polygon-based mask support

## Future Development

| Priority | Task |
|----------|------|
| 1 | Expand dataset size and defect variety |
| 2 | Balance class distribution |
| 3 | Add new defect types (e.g., moisture-damaged / wet boxes) |
| 4 | Integrate tracking / barcode recognition for shipment identification |
| 5 | Optimize for edge devices |

## Project Structure

```
scripts/
├── data_preparation/
│   ├── cut_videos.py          # Extract frames from videos
│   ├── coco_to_yolov8.py      # COCO JSON → YOLOv8 format
│   └── mask_to_poly.py        # Binary masks → YOLO polygons
├── visualization/
│   ├── markup.py              # Draw bounding boxes on images
│   ├── segmentation.py        # Draw segmentation polygons
│   ├── show_segm_masks.py     # Display YOLO segmentation masks
│   └── plot_data_desc.py      # Dataset statistics visualization
└── evaluation/
    ├── plot_training_results.py  # Loss and mAP curves
    └── models_args.py             # Model hyperparameter analysis
```

## Scripts Description

### Data Preparation

| Script | Purpose |
|--------|---------|
| `cut_videos.py` | Extract every 10th frame from `.MOV` videos to create image datasets |
| `coco_to_yolov8.py` | Convert COCO JSON annotations to YOLOv8 polygon format |
| `mask_to_poly.py` | Convert binary segmentation masks to normalized YOLO polygons |

### Visualization

| Script | Purpose |
|--------|---------|
| `markup.py` | Draw bounding boxes on images from JSON annotations |
| `segmentation.py` | Draw colored segmentation polygons with transparency overlay |
| `show_segm_masks.py` | Visualize YOLOv8 polygon masks on original images |
| `plot_data_desc.py` | Generate bar charts for dataset statistics and class balance |

### Evaluation

| Script | Purpose |
|--------|---------|
| `plot_training_results.py` | Plot training/validation loss curves and mAP metrics |
| `models_args.py` | Extract and display training hyperparameters (epochs, batch, imgsz) |

## Quick Start

### Installation

```bash
pip install ultralytics opencv-python shapely numpy pandas matplotlib plotly natsort tqdm
```

### Basic Usage Example

```python
# Example: Convert COCO annotations to YOLOv8 format
from coco_to_yolov8 import open_json_markup

# Or visualize training results
from plot_training_results import plot_training_results
plot_training_results(paths_to_res, plot_type='metrics')
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abdushkureva2025quality,
  title={Quality control in manufacturing logistics with convolutional neural networks},
  author={Abdushkureva, Alina and Strimovskaya, Anna},
  journal={Journal of Management Analytics},
  year={2025}
}
```

## License

This project is part of an academic study. For licensing inquiries, please contact the corresponding author.

## Contact

**Alina Abdushkureva** – [avabdushkureva@edu.hse.ru](mailto:avabdushkureva@edu.hse.ru)

**Anna Strimovskaya** – [astrim26@mail.ru](mailto:astrim26@mail.ru)

Project Link: [https://github.com/avabdushkureva/defect-detection](https://github.com/avabdushkureva/defect-detection)
