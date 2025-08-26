# src/tasks/detection_task.py

import json
import os
import logging
from utils.logger import setup_logger

logger = setup_logger(
    name="visdrone_parser",
    log_file="logs/visdrone_parse_log.txt",
    log_level=logging.INFO,
)
from typing import List, Dict, Any


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "visdrone_parse_log.txt")


def parse_visdrone_txt(txt_path: str) -> List[Dict[str, Any]]:
    detections = []
    total_lines = 0
    valid_lines = 0
    LOG_DIR = "logs"
    LOG_FILE = os.path.join(LOG_DIR, "visdrone_parse_log.txt")

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(txt_path, 'r', encoding='utf-8') as f, open(LOG_FILE, 'a', encoding='utf-8') as log_f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                log_f.write(f"[Line {line_num}] ⚠️ Empty line, skipping\n")
                continue
            parts = line.split(',')
            if len(parts) < 6:
                log_f.write(f"[Line {line_num}] ❌ Insufficient fields: only {len(parts)} found. Content: {line}\n")
                continue
            try:
                x1 = float(parts[0])
                y1 = float(parts[1])
                x2 = float(parts[2])
                y2 = float(parts[3])
                category_id = int(parts[5].strip())
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": f"class_{category_id}",
                    "category_id": category_id
                })
                valid_lines += 1
                log_f.write(
                    f"[Line {line_num}] ✅ Parsed successfully → category_id={category_id}, box=[{x1},{y1},{x2},{y2}]\n")
            except ValueError as ve:
                log_f.write(f"[Line {line_num}] ❌ Parsing failed (value error): {line}, Error: {ve}\n")
            except Exception as e:
                log_f.write(f"[Line {line_num}] ❌ Parsing failed (unknown error): {line}, Error: {e}\n")

        log_f.write(
            f"[Summary] File {txt_path}: Total lines={total_lines}, Successfully parsed={valid_lines}, Failed={total_lines - valid_lines}\n\n")

    return detections


def generate_detection_tasks(annotations_folder: str, output_dir: str) -> None:
    all_tasks = []
    for txt_file in os.listdir(annotations_folder):
        if not txt_file.endswith('.txt'):
            continue
        txt_path = os.path.join(annotations_folder, txt_file)
        detections = parse_visdrone_txt(txt_path)

        image_file = txt_file.replace('.txt', '.jpg')
        image_id = hash(txt_file)

        for det in detections:
            all_tasks.append({
                "image_id": image_id,
                "image_file": image_file,
                "task_type": "detection",
                "detections": [det]
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detection_tasks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=2)
    print(f"✅ Detection tasks generated: {output_path}, Total records: {len(all_tasks)}")