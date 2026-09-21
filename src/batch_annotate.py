# GS标注工具 - 批量标注脚本
# 标注 735 张图片，带定期进度汇报
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.runtime_env import bootstrap_runtime_env

bootstrap_runtime_env()

import os
import time
import json
from datetime import datetime
from typing import Any, Iterable, List, Sequence

import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from gui.models.grounding_dino_model import GroundingDINOModel

# ===== 配置 =====
DATASET_DIR = Path('../images/train')
LABEL_DIR = Path('../labels/train')
MODEL_PATH = Path('../local_models/grounding-dino-base')
PROGRESS_FILE = Path('annotation_progress.json')

# 标注参数
PROMPT = 'weed, plant, green plant, leaf'  # 杂草相关类别
BOX_THRESHOLD = 0.30  # 与 groundingdino_video_gui.py 一致
TEXT_THRESHOLD = 0.25

# ===== 进度管理 =====
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': 0, 'failed': 0, 'total': 0, 'start_time': None, 'last_report': None}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def report_progress(progress, force=False):
    """报告进度（每5分钟或强制）"""
    now = time.time()
    if not force and progress['last_report']:
        if now - progress['last_report'] < 300:  # 5分钟 = 300秒
            return
    
    progress['last_report'] = now
    
    elapsed = now - progress['start_time']
    completed = progress['completed']
    total = progress['total']
    failed = progress['failed']
    
    avg_time = elapsed / completed if completed > 0 else 0
    remaining = (total - completed) * avg_time if completed > 0 else 0
    
    report = f"""
{'='*60}
[REPORT] Progress Update - {datetime.now().strftime('%H:%M:%S')}
{'='*60}
Completed: {completed}/{total} ({100*completed/total:.1f}%)
Failed: {failed}
Elapsed: {elapsed/60:.1f} min
Remaining: {remaining/60:.1f} min
Speed: {avg_time:.2f} sec/image
{'='*60}
"""
    print(report)
    save_progress(progress)


def _box_to_xyxy(box: Any) -> Sequence[float]:
    if hasattr(box, "tolist"):
        return box.tolist()
    return box


def _to_python_list(values: Any) -> List[Any]:
    if isinstance(values, torch.Tensor):
        converted = values.detach().cpu().tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(values, list):
        return values
    return list(values)


def _map_detection_labels(
    boxes: List[Sequence[float]],
    labels: Any,
    scores: Any,
    class_names: Sequence[str],
) -> tuple[List[Sequence[float]], List[int], List[float]]:
    """Mirror GroundingDINOModel.infer(): map string labels; pass int ids through."""
    labels_list = _to_python_list(labels)
    scores_list = [float(score) for score in _to_python_list(scores)]

    if labels_list and isinstance(labels_list[0], str):
        return GroundingDINOModel._map_string_labels(
            boxes, labels_list, scores_list, list(class_names)
        )

    out_boxes: List[Sequence[float]] = []
    out_labels: List[int] = []
    out_scores: List[float] = []
    max_class_id = len(class_names) - 1
    for box, label, score in zip(boxes, labels_list, scores_list):
        try:
            class_id = int(label)
        except (TypeError, ValueError):
            continue
        if 0 <= class_id <= max_class_id:
            out_boxes.append(box)
            out_labels.append(class_id)
            out_scores.append(score)
    return out_boxes, out_labels, out_scores


def boxes_to_yolo_lines(
    boxes: Iterable[Any],
    labels: Sequence[Any],
    scores: Sequence[float],
    class_names: Sequence[str],
    img_width: int,
    img_height: int,
) -> List[str]:
    """Map detections to YOLO label lines; skip labels outside the prompt class list."""
    xyxy_boxes = [_box_to_xyxy(box) for box in boxes]
    mapped_boxes, mapped_labels, _ = _map_detection_labels(
        xyxy_boxes, labels, scores, class_names
    )
    lines: List[str] = []
    for box, class_id in zip(mapped_boxes, mapped_labels):
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def main() -> None:
    print("=" * 60)
    print("[START] GS标注工具 批量标注")
    print("=" * 60)
    print(f"类别: {PROMPT}")
    print(f"阈值: box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"图片目录: {DATASET_DIR}")
    print(f"标签目录: {LABEL_DIR}")
    print("=" * 60)

    class_names = GroundingDINOModel._parse_prompt_class_names(PROMPT)

    progress = load_progress()
    progress['start_time'] = time.time()
    progress['total'] = len(list(DATASET_DIR.glob('*.jpg'))) + len(list(DATASET_DIR.glob('*.png')))
    progress['last_report'] = None
    save_progress(progress)

    print("\n[1/3] 加载GS模型...")
    start = time.time()

    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU - inference will be extremely slow!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()

    print(f"模型加载完成: {time.time()-start:.1f}秒")

    images = sorted(list(DATASET_DIR.glob('*.jpg')) + list(DATASET_DIR.glob('*.png')))
    print(f"\n[2/3] 待标注图片: {len(images)} 张")

    print(f"\n[3/3] 开始标注...")
    print("-" * 60)

    start_time = time.time()
    last_check = time.time()

    for i, img_path in enumerate(images):
        try:
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            inputs = processor(images=pil_img, text=PROMPT, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)

            import inspect
            sig_params = inspect.signature(processor.post_process_grounded_object_detection).parameters
            if "box_threshold" in sig_params:
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                    target_sizes=target_sizes
                )[0]
            else:
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                    target_sizes=target_sizes
                )[0]

            label_path = LABEL_DIR / (img_path.stem + '.txt')
            img_w, img_h = pil_img.size
            yolo_lines = boxes_to_yolo_lines(
                results['boxes'],
                results['labels'],
                results['scores'],
                class_names,
                img_w,
                img_h,
            )
            with open(label_path, 'w') as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines) + "\n")

            progress['completed'] += 1

        except Exception as e:
            progress['failed'] += 1
            print(f"[错误] {img_path.name}: {str(e)}")

        if time.time() - last_check >= 60:
            last_check = time.time()
            report_progress(progress)
            save_progress(progress)

        if (i + 1) % 10 == 0 or i == len(images) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"\r进度: {i+1}/{len(images)} ({100*(i+1)/len(images):.1f}%) | {rate:.2f} 张/秒",
                end='',
                flush=True,
            )

    print("\n")
    save_progress(progress)
    report_progress(progress, force=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("[DONE] Batch Annotation Complete!")
    print("=" * 60)
    print(f"Total: {progress['completed']} success, {progress['failed']} failed")
    print(f"Duration: {elapsed/60:.1f} min")
    if progress['completed'] > 0:
        print(f"Avg Speed: {elapsed/progress['completed']:.2f} sec/image")
    print("=" * 60)


if __name__ == "__main__":
    main()
