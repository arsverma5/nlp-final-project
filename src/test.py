# test.py

from preprocessing import RecipeProcessor
import pandas as pd

def main():
    repro = RecipeProcessor()

    recipe_df = pd.read_csv("data/raw/recipes.csv")
    
    # print(recipe_df["ingredients"][0])

    print(repro.parse_quantities(recipe_df["ingredients"][0]))

if __name__ == "__main__":
    main()