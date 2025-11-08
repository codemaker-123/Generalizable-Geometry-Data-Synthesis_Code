# Generalizable-Geometry-Data-Synthesis_Code
# dependency
conda env create -f environment.yml
# generate data
python generate_refine.py


# Geometry Dataset Generation

This repository provides a Python script for **automatically generating geometric reasoning datasets**.
It creates problem statements, renders corresponding geometric diagrams, and saves structured data (including images and clause annotations) in JSON format.

---

## ⚙️ Main Configurable Variables

You can adjust the following variables near the top of the script to control the behavior of data generation:

| Variable       | Description                                                                                              | Default                          |
| -------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `need_proof`   | Whether to include proof-related information.                                                            | `False`                          |
| `augment`      | Enable image augmentation (e.g., occlusion).                                                             | `False`                          |
| `test`         | Use testing mode (smaller dataset for debugging).                                                        | `True`                           |
| `generate_num` | Total number of data samples to generate. Automatically set to `1000` if `test=True`, otherwise `10000`. | `1000`                           |
| `process_num`  | Number of parallel processes. Default is all CPU cores minus one.                                        | Depends on CPU                   |
| `batch_size`   | Number of samples written to file per batch.                                                             | `20`                             |
| `save_dir`     | Directory to save generated data and images. Automatically set based on `test` mode.                     | `./dataset_test_hard_small_new/` |
| `defs_path`    | Path to the definitions file.                                                                            | `./defs.txt`                     |
| `rules_path`   | Path to the rules file.                                                                                  | `./rules.txt`                    |
| `max_retries`  | Maximum number of retries per sample if generation fails.                                                | `5`                              |

---

## 📂 Output Structure

```
save_dir/
├── data.json              # Main dataset file with all generated entries
├── sorted_data.json       # Clause frequency analysis
├── failed_ids.json        # List of failed generation IDs
├── img/                   # Folder containing generated images
└── temp/                  # Temporary files (auto-deleted after use)
```

Each JSON entry contains:

```json
{
  "id": "0",
  "image": "img/0.jpg",
  "conversations": [
    {"from": "human", "value": "Render a clear and concise description of an image about geometric shapes.\n<image>"},
    {"from": "gpt", "value": "Generated natural language description..."}
  ],
  "clause": ["clause_1", "clause_2", "..."]
}
```

---

## 🧩 Notes

* Image generation and graph building each have a **20-second timeout**.
* If a process fails multiple times, its ID is logged in `failed_ids.json`.
* Use `test=True` for quick debugging before full dataset generation.
* All logs are saved in `generation.log`.

---

## 🕹️ Example Quick Edit

To generate a full 10K dataset using all CPU cores:

```python
test = False
generate_num = 10000
```

To reduce the number of parallel processes:

```python
process_num = 4
```
