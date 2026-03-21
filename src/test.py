# test.py

from preprocessing import RecipeProcessor
import pandas as pd

def main():
    repro = RecipeProcessor()

    recipe_df = pd.read_csv("data/raw/recipes.csv")
    
    # print(recipe_df["ingredients"][0])

    ingredients = repro.parse_quantities(recipe_df["ingredients"][0])
    print(ingredients)

    print(repro.extract_ing_features(ingredients))

if __name__ == "__main__":
    main()