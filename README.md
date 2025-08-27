# SFT Format Conversion Tool for Low-Altitude Economy Datasets (Current Implementation: Detection Task / VisDrone Dataset)

## 🎯 Project Objective

This project aims to build a **generalized SFT (Supervised Fine-Tuning) data conversion tool for low-altitude economic scenarios**, which converts annotation data from various low-altitute datasets (such as UAV-captured images/videos) into **Question-and-Answer (QA) pair format suitable for supervised fine-tuning of large language models (LLMs) or vision-language models (VLMs)**.

---

## 🧩 Project Background

In low-altitute economic applications—such as UAV logistics, urban inspection, security monitoring, and agricultural protection—a large volume of aerial imagery or video data is typically collected and annotated for AI model training. Common task types include:

1. **Detection**: Detecting objects in images (e.g., drones, vehicles, pedestrians, buildings, etc.) and outputting bounding boxes.
2. **Classification**: Categorizing images, regions, or objects (e.g., determining whether something is a violation, congestion, or a specific target).
3. **Segmentation**: Pixel-level segmentation of targets or regions (e.g., roads, buildings, vegetation, etc.).
4. **Counting**: Counting the number of specific targets (e.g., number of vehicles, crowds, etc.).
5. **Trajectory / Tracking**: Tracking the movement trajectory of targets (e.g., drones, vehicles, animals, etc.).
6. **Event / Anomaly**: Detecting abnormal events (e.g., illegal flying, fire, collision, congestion, etc.).

> ✅ This project serves as a **starting point for developing a general-purpose script tool**, with the goal of supporting the conversion of annotation data for the above **six major task types** into **SFT-format QA pairs**.
> 
> **Currently, only the first type — Detection (Detection) — has been implemented**, using the **VisDrone 2019 dataset** as an example. The other five task types (Classification, Segmentation, Counting, Trajectory, and Event) will be extended in the future.

---

## 🗂️ Project Directory Structure (Tree View)

```text
Low-Altitute-Intelligence/

├── src/

│ ├── main.py

│ ├── tasks/

│ │ └── detection_task.py

│ ├── converters/

│ │ └── to_sft_format.py

│ ├── utils/

│ │ ├── init.py

│ │ ├── logger.py

│ │ ├── category_mapping.py

│ │ └── path_utils.py

│ └── templates/

│ │ └── detection_template.py

│

├── outputs/

│ ├── tasks/

│ └── sft/

│

├── logs/

│ └── visdrone_parse_log.txt

├── examples/

│ └── run_full_conversion.py

├── datasets/

│

└── ...
```
> 💡 **Notes:**
> - Currently, the project takes the **VisDrone Detection Dataset** as its first use case, and implements the full pipeline from `.txt annotation → detection task → SFT QA pair`.
> - The project adopts a **modular design**, making it easy to extend to the other five task types (Classification, Segmentation, Counting, Trajectory, Event) by simply adding corresponding task modules.

---

## ✨ Currently Implemented Functional Modules

### 1. 📦 Detection Task Generation Module (`src/tasks/detection_task.py`)
- Reads VisDrone `.txt` annotation files (each line formatted as: `x1,y1,x2,y2,confidence,category_id`)
- Parses and constructs structured detection tasks, including:
  - Image filename (e.g., `000001.jpg`)
  - Bounding box coordinates (`x1, y1, x2, y2`)
  - Category ID and label (e.g., `class_0`)
- Output: `outputs/tasks/detection_tasks.json`

### 2. 🔁 SFT Format Conversion Module (`src/converters/to_sft_format.py`)
- Converts the detection tasks into **“Human Question + GPT Answer”-style QA pairs**
- Output: `outputs/sft/sft_detection_qa.json`, suitable for fine-tuning large models

### 3. 🛠️ Utility Modules
- **`logger.py`**: Wraps Python’s built-in `logging` module to support output to both console and log file
- **`category_mapping.py`**: Mapping table for VisDrone category IDs and their corresponding names
- **`path_utils.py`**: Path-related utilities, such as ensuring output directories exist

### 4. ▶️ Outputs
- Detection tasks JSON: `outputs/tasks/detection_tasks.json`
- SFT-format QA pairs: `outputs/sft/sft_detection_qa.json`
- Log file: `logs/visdrone_parse_log.txt`
---

## 📂 Currently Used Dataset

### ✅ VisDrone 2019 Detection Dataset (Current Example Only)
- Annotation format: `.txt` files, each line represents a detected object with fields: `x1,y1,x2,y2,confidence,category_id`
- Image format: `.jpg`
- Task type: Object Detection (Bounding Box)

Due to GitHub's file size limitations, the ​​VisDrone dataset has not been uploaded to this repository​​.

To avoid uploading large files, this project ​​does not include the datasets/directory by default​​.

You need to ​​manually create the datasets/folder (if it does not already exist)​​ at the same level as the src/directory,

then ​​download the dataset and place it in the datasets/folder​​ so that the program can correctly read the annotation and image files.

🔗 ​​Download link:​​ https://github.com/VisDrone/VisDrone-Dataset

> 📌 **Note:** Currently, only the VisDrone Detection Task is implemented as an example. In the future, you can gradually add other datasets (e.g., COCO, custom UAV data, etc.) by providing the corresponding annotation formats and category mappings, then creating new task modules as needed.

🧪 Standalone Full Script
If you prefer to run the ​​entire process in a single script​​ (without using the modular structure), you can use a standalone script that integrates all steps:

examples/run_full_conversion.py

---

## ⚠️ Notes & Future Extensions

### ✅ Current Limitations
- Supports only the **VisDrone Detection Task**
- Supports only **Object Detection (Bounding Box)**
- Data paths are currently hard-coded (can be improved later via config files or CLI arguments)

### 🔧 Future Extensions
1. **Support for the other 5 low-altitude tasks**: Classification, Segmentation, Counting, Trajectory, and Event/Anomaly
2. **Support for more datasets**: such as COCO, custom data, multi-source UAV data
3. **Support for command-line arguments / configuration files**: for flexible specification of task type, data path, output directory, etc.
4. **Support for multilingual / customizable QA templates**: e.g., customized human/GPT dialog templates, multilingual labels
5. **Support for a unified task abstraction layer**: to provide a generic interface for different task types, improving extensibility and maintainability

---

## 🙌 Summary

Starting from the **SFT-format conversion needs of low-altitute economic datasets**, and using the **VisDrone Detection Dataset** as the first implementation example, this project has achieved:
- The full pipeline from **annotation parsing → task construction → SFT QA pair generation**
- A **modular, extensible code structure** that lays the foundation for the other five task types (Classification, Segmentation, Counting, Trajectory, Event)

---