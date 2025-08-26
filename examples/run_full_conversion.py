import json
import os
from typing import List, Dict, Any


# ======================
# Step 1: Parse VisDrone .txt files and generate detection_tasks.json
# ======================

def parse_visdrone_txt(txt_path: str):
    detections = []
    total_lines = 0
    valid_lines = 0
    log_file_path = "visdrone_parse_log.txt"

    with open(txt_path, 'r', encoding='utf-8') as f, open(log_file_path, 'a', encoding='utf-8') as log_f:
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
                log_f.write(f"[Line {line_num}] ✅ Parsed successfully → category_id={category_id}, x1={x1}, y1={y1}, x2={x2}, y2={y2}\n")
            except ValueError as ve:
                log_f.write(f"[Line {line_num}] ❌ Parsing failed (value error): {line}, Error: {ve}\n")
            except Exception as e:
                log_f.write(f"[Line {line_num}] ❌ Parsing failed (unknown error): {line}, Error: {e}\n")

        log_f.write(f"[Summary] File {txt_path}: Total lines={total_lines}, Successfully parsed={valid_lines}, Failed={total_lines - valid_lines}\n\n")

    return detections


def generate_detection_tasks(annotations_folder: str, output_dir: str):
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


# ======================
# Step 2: Convert detection_tasks.json to SFT format (human + GPT QA pairs)
# ======================

def convert_detection_tasks_to_sft_llava_debug(
    detection_tasks_json_path: str,
    output_sft_json_path: str,
    images_base_dir: str,
    category_mapping: Dict[int, str] = None
):
    if category_mapping is None:
        category_mapping = {}

    with open(detection_tasks_json_path, 'r', encoding='utf-8') as f:
        tasks_data = json.load(f)

    sft_records = []
    skipped_detections = []
    total_images = 0
    processed_images = 0

    for record in tasks_data:
        image_id = record.get("image_id")
        image_file = record.get("image_file")
        detections = record.get("detections", [])

        total_images += 1

        image_path = os.path.join(images_base_dir, image_file)
        if not os.path.exists(image_path):
            skipped_detections.append({
                "image_file": image_file,
                "reason": "Image file does not exist",
                "detections_skipped": len(detections)
            })
            continue

        valid_detections = []
        for idx, det in enumerate(detections):
            x1 = det.get("x1")
            y1 = det.get("y1")
            x2 = det.get("x2")
            y2 = det.get("y2")
            label = det.get("label", "unknown")
            category_id = det.get("category_id", -1)

            if None in [x1, y1, x2, y2]:
                skipped_detections.append({
                    "image_file": image_file,
                    "detection_idx": idx,
                    "reason": "Coordinates contain None values",
                    "category_id": category_id,
                    "label": label
                })
                continue

            if category_id not in category_mapping:
                category_mapping[category_id] = label
            mapped_name = category_mapping.get(category_id, "unknown")

            valid_detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": mapped_name, "category_id": category_id
            })

        if not valid_detections:
            skipped_detections.append({
                "image_file": image_file,
                "reason": "No valid detections",
                "detections_skipped": len(detections)
            })
            continue

        detected_categories = {det["label"] for det in valid_detections}
        human_question = f"Does the image contain the following objects? {', '.join(detected_categories)}. If yes, please mark their locations."
        detections_text = []
        for det in valid_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = det["label"]
            category_id = det["category_id"]
            detections_text.append(f"Object {label} (ID: {category_id}), Location: [{x1}, {y1}, {x2}, {y2}]")
        gpt_answer = "Yes, the image contains the following objects: " + "；".join(detections_text) + "."

        sft_records.append({
            "image_path": image_path,
            "conversation": [
                {"from": "human", "value": human_question},
                {"from": "gpt", "value": gpt_answer}
            ]
        })
        processed_images += 1

    os.makedirs(os.path.dirname(output_sft_json_path), exist_ok=True)
    with open(output_sft_json_path, 'w', encoding='utf-8') as f:
        json.dump(sft_records, f, ensure_ascii=False, indent=2)

    print("\n=== Processing Summary ===")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Skipped: {len(skipped_detections)}")

    if skipped_detections:
        print("\n🔍 First 10 skipped records:")
        for item in skipped_detections[:10]:
            print(f"- {item.get('image_file')} | Reason: {item.get('reason')}")

    print(f"\n✅ SFT data saved to: {output_sft_json_path}")
    print("📝 Category ID to Name Mapping:")
    for k, v in sorted(category_mapping.items()):
        print(f"  {k}: {v}")


# ======================
# Step 3: Main Execution (Update the paths below!)
# ======================

if __name__ == "__main__":
    # ===============================
    # ✅ Update these 4 paths according to your system!
    # ===============================

    # 1. Folder containing VisDrone 2019-DET .txt annotation files
    annotations_folder = r"D:\datasets\VisDrone\VisDrone2019-DET-train\annotations"

    # 2. Output folder for detection_tasks.json
    output_detection_tasks_dir = r"C:\Users\38487\Desktop\VisDrone-Detection\outputs\tasks"

    # 3. Folder where your VisDrone JPG images are stored
    images_base_dir = r"D:\datasets\VisDrone\VisDrone2019-DET-train\images"

    # 4. Output path for the final SFT-format JSON (QA pairs)
    output_sft_json_path = r"C:\Users\38487\Desktop\VisDrone-Detection\outputs\sft\sft_detection_qa.json"

    # ===============================
    # Step 1: Generate detection_tasks.json
    # ===============================
    generate_detection_tasks(annotations_folder, output_detection_tasks_dir)

    # ===============================
    # Step 2: Convert to SFT format with real class names
    # ===============================
    # ✅ Official VisDrone2019-DET category mapping (ID -> English name)
    category_mapping = {
        0: "pedestrian",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning_tricycle",
        8: "bus",
        9: "motorcycle",
        10: "others",
        11: "ignored"  # or 'sign', depending on your dataset version
    }

    detection_tasks_json_path = os.path.join(output_detection_tasks_dir, "detection_tasks.json")
    convert_detection_tasks_to_sft_llava_debug(
        detection_tasks_json_path,
        output_sft_json_path,
        images_base_dir,
        category_mapping=category_mapping
    )