
import json
from pathlib import Path

# Input / output paths
input_path = Path("PerGAZE.json")
output_dir = input_path.parent

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Split by dataset
air_data = [
    record for record in data
    if record.get("dataset") == "AiR"
]

coco_data = [
    record for record in data
    if record.get("dataset") == "COCO-Search18"
]

# Save JSON files
for filename, records in [
    ("data/split/AiR/AiR.json", air_data),
    ("data/split/COCO_Search18/COCOSearch-18.json", coco_data),
]:
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"{filename}: {len(records):,} samples")

# Verify no samples lost
assert len(data) == len(air_data) + len(coco_data), \
    "Some records have unknown dataset values!"

print(f"Total: {len(data):,} samples")
