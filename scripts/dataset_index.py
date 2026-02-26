"""
Build Dataset Index for Processed PHEME Dataset
(5802 threads version)

Creates:
    data/processed/pheme_dataset_index.csv
"""

import json
import csv
from pathlib import Path


############################################################
# PROJECT ROOT (AUTO-DETECT)
############################################################

# scripts/ → project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "pheme_dataset"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pheme_dataset_index.csv"


############################################################
# BUILD INDEX
############################################################

def build_index():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n{DATASET_DIR}\n"
            "\nRun pheme_processed.py first."
        )

    rows = []

    print("Scanning processed dataset...\n")

    for event_dir in sorted(DATASET_DIR.iterdir()):

        if not event_dir.is_dir():
            continue

        event_name = event_dir.name

        for json_file in sorted(event_dir.glob("*.json")):

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                rows.append({
                    "thread_id": data["thread_id"],
                    "event": event_name,
                    "label": data["label"],
                    "file_path": str(json_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                })

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

    ########################################################
    # SAVE CSV
    ########################################################

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "thread_id",
                "event",
                "label",
                "file_path"
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ Dataset index created successfully")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total threads indexed: {len(rows)}")


############################################################
# MAIN
############################################################

if __name__ == "__main__":
    build_index()