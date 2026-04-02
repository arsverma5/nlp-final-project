# augment recipes.csv with Very Long (>120 min) recipes from RAW_recipes.csv
# Output: data/raw/recipes_augmented.csv

import pandas as pd
import ast

RAW_PATH = "../data/raw/RAW_recipes.csv"
RECIPES_PATH = "../data/raw/recipes.csv"
OUT_PATH = "../data/raw/recipes_augmented.csv"

TARGET_BUCKET_MIN = 120
# how many Very Long recipes to add
TARGET_COUNT = 300


def convert_ingredients(raw_str: str) -> str:
    """
    Convert Food.com list format "['flour', 'eggs']"
    -> comma string "flour, eggs" to match recipes.csv format
    """
    try:
        items = ast.literal_eval(raw_str)
        return ", ".join(items)
    except Exception:
        return str(raw_str)


def convert_steps(raw_str: str) -> str:
    """
    Convert Food.com steps list -> single string joined by spaces
    """
    try:
        steps = ast.literal_eval(raw_str)
        return " ".join(steps)
    except Exception:
        return str(raw_str)


def build_timing_string(total_minutes: int) -> str:
    """
    Construct a timing string that parse_total_time() in preprocessing.py
    can parse ex: "Total Time: 2 hrs 30 mins"
    """
    hours = int(total_minutes) // 60
    mins  = int(total_minutes) % 60
    parts = []
    if hours:
        parts.append(f"{hours} hrs")
    if mins:
        parts.append(f"{mins} mins")
    time_str = " ".join(parts) if parts else "0 mins"
    return f"Total Time: {time_str}"

def main():
    raw = pd.read_csv(RAW_PATH)
    existing = pd.read_csv(RECIPES_PATH, index_col=0)

    print(f"RAW_recipes: {len(raw):,} rows")
    print(f"Existing recipes.csv: {len(existing):,} rows")

    from preprocessing import RecipeProcessor
    processor = RecipeProcessor()

    existing_copy = existing.copy()
    existing_copy["total_minutes"] = existing_copy["timing"].apply(processor.parse_total_time)
    existing_clean = existing_copy.dropna(subset=["total_minutes"])
    existing_clean = existing_clean[existing_clean["total_minutes"] <= 4320]
    existing_clean["time_bucket"] = processor.bucketize_time(existing_clean["total_minutes"])

    print("\nCurrent bucket distribution:")
    print(existing_clean["time_bucket"].value_counts().reindex([
        "Quick (<=30 min)", "Medium (31-60 min)",
        "Long (61-120 min)", "Very Long (>120 min)"
    ]))

    very_long_raw = raw[
        (raw["minutes"] > TARGET_BUCKET_MIN) &
        (raw["minutes"] <= 4320)
    ].copy()

    print(f"\nAvailable Very Long recipes in RAW: {len(very_long_raw):,}")

    sample = very_long_raw.sample(
        n=min(TARGET_COUNT, len(very_long_raw)),
        random_state=42
    )

    new_rows = pd.DataFrame()
    new_rows["recipe_name"] = sample["name"].values
    new_rows["ingredients"] = sample["ingredients"].apply(convert_ingredients).values
    new_rows["directions"] = sample["steps"].apply(convert_steps).values
    new_rows["timing"] = sample["minutes"].apply(build_timing_string).values
    new_rows["prep_time"] = ""
    new_rows["cook_time"] = ""
    new_rows["total_time"] = sample["minutes"].astype(str).values + " mins"
    new_rows["servings"] = ""
    new_rows["yield"] = ""
    new_rows["rating"] = None
    new_rows["url"] = ""
    new_rows["cuisine_path"] = ""
    new_rows["nutrition"] = ""
    new_rows["img_src"] = ""

    new_rows = new_rows[existing.columns]

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved augmented dataset -> {OUT_PATH}")
    print(f"Total rows: {len(combined):,}")

    combined_copy = combined.copy()
    combined_copy["total_minutes"] = combined_copy["timing"].apply(processor.parse_total_time)
    combined_clean = combined_copy.dropna(subset=["total_minutes"])
    combined_clean = combined_clean[combined_clean["total_minutes"] <= 4320]
    combined_clean["time_bucket"] = processor.bucketize_time(combined_clean["total_minutes"])

    print("\nNew bucket distribution:")
    print(combined_clean["time_bucket"].value_counts().reindex([
        "Quick (<=30 min)", "Medium (31-60 min)",
        "Long (61-120 min)", "Very Long (>120 min)"
    ]))

if __name__ == "__main__":
    main()