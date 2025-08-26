# src/main.py

import os

# === Import task modules (responsible for task construction logic) ===
from tasks.detection_task import generate_detection_tasks

# === Import conversion modules (e.g., converting to SFT format) ===
from converters.to_sft_format import convert_detection_tasks_to_sft_llava_debug

# === Import utility modules (e.g., category mapping, path utilities) ===
from utils.category_mapping import VISDRONE_CATEGORY_MAPPING
from utils.path_utils import ensure_dir_exists


def main():
    # ===============================
    # ✅ You can modify these paths according to your environment!
    # ===============================
    # Path to the original dataset (VisDrone .txt annotation files and images)
    annotations_folder = r"C:\Users\38487\Desktop\Low-Altitude-Intelligence\datasets\VisDrone\VisDrone2019-DET-train\annotations"
    images_base_dir = r"C:\Users\38487\Desktop\Low-Altitude-Intelligence\datasets\VisDrone\VisDrone2019-DET-train\images"

    # Directory where the generated detection tasks will be saved (detection_tasks.json)
    output_detection_tasks_dir = r"C:\Users\38487\Desktop\Low-Altitude-Intelligence\outputs\tasks"

    # Final output path for the SFT format data (human + gpt QA pairs)
    output_sft_json_path = r"C:\Users\38487\Desktop\Low-Altitude-Intelligence\outputs\sft\sft_detection_qa.json"

    # ===============================
    # Step 1: Call the task module to generate detection tasks (detection_tasks.json)
    # ===============================
    # The generated tasks will be saved in the file: detection_tasks.json inside output_detection_tasks_dir
    generate_detection_tasks(annotations_folder, output_detection_tasks_dir)

    # ===============================
    # Step 2: Call the conversion module to convert detection_tasks.json to SFT format (QA pairs)
    # ===============================
    detection_tasks_json_path = os.path.join(output_detection_tasks_dir, "detection_tasks.json")
    ensure_dir_exists(os.path.dirname(output_sft_json_path))  # Ensure the SFT output directory exists

    convert_detection_tasks_to_sft_llava_debug(
        detection_tasks_json_path,
        output_sft_json_path,
        images_base_dir,
        category_mapping=VISDRONE_CATEGORY_MAPPING
    )

    print("✅ All tasks executed successfully!")


if __name__ == "__main__":
    main()