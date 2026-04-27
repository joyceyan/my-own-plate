"""
Prepare Nutrition5k dataset for vision-language model fine-tuning.

Reads dish metadata CSVs, normalizes ingredient names, filters invalid
samples, and writes a single nutrition5k_all.jsonl with all valid records.

Train/val/test splitting is handled downstream by convert_dataset.py.
"""

import csv
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
NUTRITION5K_ROOT = Path.home() / "src" / "Nutrition5k"
METADATA_DIR = NUTRITION5K_ROOT / "metadata"
IMAGERY_DIR = NUTRITION5K_ROOT / "imagery"
OUTPUT_DIR = Path(__file__).resolve().parent

CAFE1_CSV = METADATA_DIR / "dish_metadata_cafe1.csv"
CAFE2_CSV = METADATA_DIR / "dish_metadata_cafe2.csv"

SEED = 42
MAX_INGREDIENTS = 5

PROMPT = (
    "Identify the ingredients in this image and estimate calories (kcal), protein (g), "
    "fat (g), and carbohydrates (g). Respond as JSON with keys: ingredients, "
    "calories, protein, fat, carbs."
)

# ── Ingredient normalization ───────────────────────────────────────────────────

STRIP_QUALIFIERS = re.compile(
    r"\b(?:boneless|skinless|cooked|raw|canned|low-fat|low fat|nonfat|non-fat"
    r"|drained|chopped|sliced|diced|frozen|thawed|prepared)\b",
    re.IGNORECASE,
)

RENAME_MAP = {
    "beverage water": "water",
    "bread white": "white bread",
    "sauce tomato": "tomato sauce",
}


def normalize_ingredient(name: str) -> str:
    """Strip qualifiers, apply renames, title-case, and collapse whitespace."""
    name = STRIP_QUALIFIERS.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    lowered = name.lower()
    if lowered in RENAME_MAP:
        name = RENAME_MAP[lowered]
    return name.strip().title()


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_dish_row(row: list[str]) -> dict | None:
    """Parse a single CSV row into a structured dish record.

    The CSV has no header. Columns are:
      dish_id, total_calories, total_mass, total_fat, total_carb, total_protein,
      num_ingrs, then repeating groups of 7 per ingredient:
      (ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein)
    """
    if len(row) < 6:
        return None

    dish_id = row[0].strip()
    try:
        total_calories = float(row[1])
        total_mass = float(row[2])
        total_fat = float(row[3])
        total_carb = float(row[4])
        total_protein = float(row[5])
    except (ValueError, IndexError):
        return None

    # Extract ingredient names from repeating 7-field groups starting at index 6
    # Each group: ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein
    ingredients = []
    ingr_start = 6
    num_ingrs = (len(row) - ingr_start) // 7
    for i in range(num_ingrs):
        name_idx = ingr_start + i * 7 + 1  # ingr_name is the 2nd field in each group
        if name_idx < len(row):
            name = normalize_ingredient(row[name_idx].strip())
            if name:
                ingredients.append(name)

    image_path = (
        NUTRITION5K_ROOT / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    )

    return {
        "dish_id": dish_id,
        "image_path": image_path,
        "calories": total_calories,
        "total_fat": total_fat,
        "total_carb": total_carb,
        "total_protein": total_protein,
        "total_mass": total_mass,
        "ingredients": ingredients,
    }


def load_metadata(csv_path: Path) -> list[dict]:
    """Load and parse a dish metadata CSV (no header row)."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            record = parse_dish_row(row)
            if record is not None:
                records.append(record)
    return records


# ── Completion formatting ──────────────────────────────────────────────────────

def format_completion(record: dict) -> str:
    """Build a structured JSON completion string from a dish record."""
    obj = {
        "ingredients": ", ".join(record["ingredients"][:MAX_INGREDIENTS]),
        "calories": record["calories"],
        "protein": record["total_protein"],
        "fat": record["total_fat"],
        "carbs": record["total_carb"],
    }
    return json.dumps(obj)


# ── Main ───────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file with prompt/completion format."""
    with open(path, "w") as f:
        for r in records:
            entry = {
                "image_path": str(r["image_path"]),
                "prompt": PROMPT,
                "completion": format_completion(r),
            }
            f.write(json.dumps(entry) + "\n")


def main():
    # 1. Load both cafeteria CSVs
    raw_records = []
    for csv_path in [CAFE1_CSV, CAFE2_CSV]:
        if csv_path.exists():
            records = load_metadata(csv_path)
            print(f"Loaded {len(records)} dishes from {csv_path.name}")
            raw_records.extend(records)
        else:
            print(f"WARNING: {csv_path} not found, skipping.")

    total_raw = len(raw_records)
    if total_raw == 0:
        print("No records loaded. Ensure the metadata CSVs are present.")
        return

    # 2. Filter: missing images and empty ingredients
    skipped_image = 0
    skipped_ingredients = 0
    valid_records = []

    for r in raw_records:
        if not r["image_path"].exists():
            logger.warning("Missing image for %s, skipping.", r["dish_id"])
            skipped_image += 1
            continue
        if not r["ingredients"]:
            logger.warning("No ingredients for %s, skipping.", r["dish_id"])
            skipped_ingredients += 1
            continue
        valid_records.append(r)

    # 3. Shuffle (deterministic) for reproducibility — splitting handled downstream
    random.seed(SEED)
    random.shuffle(valid_records)

    # 4. Write single JSONL with all valid records
    all_jsonl_path = OUTPUT_DIR / "nutrition5k_all.jsonl"
    write_jsonl(valid_records, all_jsonl_path)

    # 5. Print summary
    total_skipped = skipped_image + skipped_ingredients
    print(f"\n{'='*60}")
    print(f"Raw records loaded:     {total_raw}")
    print(f"Skipped (no image):     {skipped_image}")
    print(f"Skipped (no ingr.):     {skipped_ingredients}")
    print(f"Total skipped:          {total_skipped}")
    print(f"Valid samples:          {len(valid_records)}")
    print(f"\nOutput: {all_jsonl_path}")
    print(f"{'='*60}")
    print(f"\nNext step: cd training && python convert_dataset.py")

    print("\nExample completions:")
    for i, r in enumerate(valid_records[:3], 1):
        print(f"\n  [{i}] {r['dish_id']}")
        print(f"      {format_completion(r)}")


if __name__ == "__main__":
    main()
