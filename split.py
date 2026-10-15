import json
from pathlib import Path

input_path = Path("PerGAZE.json")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

air = [x for x in data if x["dataset"] == "AiR"]
coco = [x for x in data if x["dataset"] == "COCO-Search18"]

for filename, records in [
    ("PerGAZE_AiR.json", air),
    ("PerGAZE_COCO-Search18.json", coco),
]:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

print(f"AiR: {len(air):,}")
print(f"COCO-Search18: {len(coco):,}")
print(f"Total: {len(air) + len(coco):,}")
