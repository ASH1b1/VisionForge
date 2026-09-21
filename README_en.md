<div align="center">
  <p>
    <a href="https://github.com/ASH1b1/VisionForge/" target="_blank">
      <img alt="VisionForge" height="200px" src="./logo.png"></a>
  </p>

[简体中文](README.md) | [English](README_en.md)

</div>

<p align="center">
    <a href="./README_en.md"><img src="https://img.shields.io/badge/Version-v3.0.0--dev-blue.svg"></a>
    <a href="./requirements.txt"><img src="https://img.shields.io/badge/Python-3.10+-aff.svg"></a>
    <a href="./README_en.md"><img src="https://img.shields.io/badge/OS-Windows%2010%2F11-pink.svg"></a>
    <a href="https://github.com/ASH1b1/VisionForge/issues"><img src="https://img.shields.io/github/issues/ASH1b1/VisionForge"></a>
</p>

<p align="center">
  <img src="./assets/visionforge-annotated.png" alt="VisionForge main window: text-prompt detection and SAM3 segmentation" width="100%" />
</p>

## 🥳 What's New

- `2026-09-19`: v3.1 capabilities designed: interactive SAM point refinement, custom Ultralytics YOLO pre-labeling, YOLO-Pose / YOLO-OBB import-export, polygon vertex editing, and unified OpenMP environment bootstrap.
- `2026-08-27`: WeChat article series published: "Goodbye LabelImg? Install this AI annotation tool with one command."
- `2026-08-25`: Repository restructured as the standalone VisionForge product repo.

## Introduction

**VisionForge** is a Windows desktop application for AI-assisted image annotation. Built around text-prompt detection, it unifies object detection, instance segmentation, pose estimation, and oriented bounding box workflows in a single interface.

Main modes:

- **Annotate mode**: Generate bounding boxes from text prompts with GroundingDINO, then refine them into polygon masks with SAM3.
- **Detect mode**: Preview detection results non-destructively, adjust confidence thresholds, and promote accepted results into the annotation project.

VisionForge is designed for engineers who need to quickly build or expand training datasets in YOLO, COCO, VOC, and other formats.

## Key Features

<p align="center">
  <img src="./assets/visionforge-unannotated.png" alt="VisionForge unannotated interface" width="100%" />
</p>

- **Text-prompt detection**: Type `person, car` and get confidence-scored bounding boxes within seconds.
- **Interactive SAM3 segmentation**: Convert boxes to polygons, refine with positive/negative clicks, undo single points with Backspace, and commit with Enter.
- **Custom YOLO pre-labeling**: Load a user-exported detection ONNX (YOLOv8 / YOLO11 default `yolo export format=onnx`), preview in Detect mode, then promote to annotations.
- **Multi-format I/O**: YOLO (detect / segment / pose / OBB), COCO, VOC, LabelImg, and PNG masks.
- **Polygon vertex editing**: Drag vertices after closing a polygon, double-click edges to insert vertices, and Delete to remove them.
- **`.gsproj` project files**: Relative paths, undo stack, train/val/test splits, and schema v4 with keypoint metadata.
- **OpenMP environment bootstrap**: Handles `libiomp5md.dll` duplicate loading between PyTorch and SAM3 to prevent `OMP Error #15`.

## Supported Tasks & Formats

| Task | Description | Import | Export |
| :--- | :--- | :--- | :--- |
| 🎯 Object Detection | Axis-aligned bounding boxes | YOLO, COCO, VOC, LabelImg | YOLO, COCO, VOC, LabelImg |
| 🖌️ Instance Segmentation | Polygons / masks | YOLO-seg, COCO polygon, PNG mask | YOLO-seg, COCO polygon, PNG mask |
| 🏃 Pose Estimation | Keypoints | YOLO-Pose, COCO keypoints | YOLO-Pose, COCO keypoints |
| 🔄 Oriented Object Detection | Normalized four-corner OBB | YOLO-OBB | YOLO-OBB |
| 👁️ Detection Preview | Non-destructive Detect mode | — | Export Detect results |

## Built-in Models

| Capability | Model | Notes |
| :--- | :--- | :--- |
| Text-prompt detection | GroundingDINO | Default detector |
| Interactive segmentation | SAM3 | Optional; supports box and point prompts |
| Custom pre-labeling | YOLO detection ONNX | Optional; load a user-exported `.onnx` |

> Model weights are large and not committed to Git. Please place them manually as described below.

## Docs

1. [Quick Start](#quick-start)
2. [Installation & Model Weights](#installation--model-weights)

## Quick Start

### Requirements

- Windows 10 / 11 (x86_64)
- Git
- Anaconda or Miniconda
- 8 GB RAM minimum; NVIDIA GPU optional for faster inference

### Installation

```powershell
git clone https://github.com/ASH1b1/VisionForge.git
cd VisionForge
conda create -n visionforge python=3.10 -y
conda activate visionforge
pip install -r requirements-cpu.txt
```

For NVIDIA GPU:

```powershell
pip install -r requirements-gpu.txt
```

### Model Weights

At minimum, download the GroundingDINO weights:

```powershell
mkdir models\grounding_dino
huggingface-cli download IDEA-Research/grounding-dino-base `
  --local-dir models\grounding_dino\models--IDEA-Research--grounding-dino-base `
  --include "*.safetensors" "*.json" "*.txt"
```

The expected full path is:

```text
models/grounding_dino/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb/model.safetensors
```

SAM3 weights are optional. See "Model files not in GitHub" below.

### Launch

```bat
启动.bat
```

Or:

```powershell
conda run -n visionforge python src/main.py
```

### Your First AI Annotation

1. Open any image.
2. Type `person, car` in the prompt box and press Enter.
3. Switch to the SAM3 tool and click any box to refine it into a polygon.
4. Export → YOLO to get `.txt` labels + `data.yaml`.

### Custom YOLO Pre-labeling

Export a detection ONNX on the training machine (the annotation machine does not need the trainer):

```text
yolo export model=best.pt format=onnx
```

Keep `best.onnx` next to `data.yaml` if you have one. In VisionForge: **Model → Load Custom YOLO → choose .onnx**.

Install the runtime only if needed (not listed in `requirements*.txt`):

```powershell
pip install onnxruntime
# NVIDIA GPU: pip install onnxruntime-gpu
```

Only the default detection export is supported (no in-graph NMS). Do not pass `nms=True`, and do not pre-label with pose / OBB ONNX.

## Model Files Not in GitHub

The following files are too large or come from independent sources and must be downloaded manually:

| Local path | Approx. size | Impact if missing |
| :--- | :--- | :--- |
| `models/grounding_dino/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb/model.safetensors` | ~890 MB | Detection unavailable |
| `models/sam3/sam3/sam3.pt` | ~3.3 GB | SAM3 segmentation unavailable |

GroundingDINO must be loaded offline: `TRANSFORMERS_OFFLINE=1`. The snapshot hash is pinned.

## Contribute

Issues and PRs are welcome. New features should go into the appropriate Controller / Model / Exporter / Importer, not `MainWindow`. Make sure tests pass before submitting:

```powershell
conda run -n visionforge python -m pytest src/tests/ -v --tb=short
```

If you find this project helpful, please consider giving it a ⭐️!

## License

Original VisionForge source is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 张世杰 (Zhang Shijie).

This repository uses a **mixed license**:

- Original VisionForge files (including `src/`): [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
- `sam3_src/`: Meta [SAM License](sam3_src/LICENSE) (not an OSI-approved license). Redistribution must remain under that license and include a copy of it
- Other third-party dependencies keep their original licenses; see [NOTICE](NOTICE)

**SAM3 use restriction:** The SAM License does not permit using SAM Materials for military, warfare, ITAR, or other end uses prohibited by Trade Controls. Apache 2.0 covers original VisionForge code and the GroundingDINO path; it does not remove those restrictions.

Contributions must follow the [Developer Certificate of Origin (DCO)](CONTRIBUTING.md). Commits need a `Signed-off-by` line.

## Acknowledgement

Thanks to the following open-source projects and communities for the inspiration and foundations:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything Model 3 (SAM3)](https://github.com/facebookresearch/segment-anything-3)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [LabelMe](https://github.com/wkentaro/labelme)
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)

## Follow Us

Follow the WeChat public account **Jade的工坊** and reply **VF** for access links, tutorial index, and community group.

<p align="center">
  <img src="./assets/公众号.png" alt="Jade的工坊 WeChat QR code" width="240px" />
</p>

<div align="center"><a href="#top">🔝 Back to Top</a></div>
