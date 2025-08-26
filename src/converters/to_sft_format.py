# src/sft_converter.py

import json
import os
from typing import List, Dict, Any

def convert_detection_tasks_to_sft_llava_debug(
    detection_tasks_json_path: str,
    output_sft_json_path: str,
    images_base_dir: str,
    category_mapping: Dict[int, str] = None
) -> None:
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