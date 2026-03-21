# preprocessing.py

# pip install pint
from pint import UnitRegistry
# pip install transformers
from transformers import pipeline 
# python -m pip install ingredient_parser_nlp
from ingredient_parser import parse_ingredient

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import re



class RecipeProcessor:
    def __init__(self, n_components=100):
        self.scaler = StandardScaler()
        # self.embedder = pipeline('feature-extraction', model="alexdseo/RecipeBERT")
        self.pca = PCA(n_components=n_components)
        self.ureg = UnitRegistry()

        # mapping instances of quantities for pint library
        self._unit_map = {
            "cup": "cup", "cups": "cup",
            "tbsp": "tablespoon", "tablespoon": "tablespoon", "tablespoons": "tablespoon",
            "tsp": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
            "oz": "ounce", "ounces": "ounce",
            "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
            "g": "gram", "gram": "grams",
            "kg": "kilogram",
            "ml": "milliliter", "mL": "milliliter",
            "l": "liter", "L": "liter",
            "fl oz": "fluid_ounce",
        }
        self.base_units = {
            # "VOLUME_BASE": self.ureg.milliliter,
            "WEIGHT_BASE": self.ureg.gram,
            "TEMP_BASE": self.ureg.degC,
        }
        

    def normalize_quantities(self, amount, unit_str):
        """
        Using pint library, normalize the quantities.
        Handling the weight and volume of ingredients.
        """
        unit_str = unit_str.lower().strip()
        pint_unit = self._unit_map.get(unit_str)
        
        # if pint unit not in map, skip it
        if pint_unit is None:
            return None 
        
        try:
            qty = amount * self.ureg.parse_expression(pint_unit)
            
            # convert to base unit depending on dimensionality
            if qty.dimensionality == self.ureg.milliliter.dimensionality:
                return qty.to(self.base_units["VOLUME_BASE"]).magnitude
            elif qty.dimensionality == self.ureg.gram.dimensionality:
                return qty.to(self.base_units["WEIGHT_BASE"]).magnitude
            else:
                return qty.magnitude  # already dimensionless or unknown
        except Exception:
            return None
        
    def parse_quantities(self, text, time_label=None, unit_map=None):
        """
        Using the pint library, parse the ing and
        return the quantities as feature vectors.
        """
        parsed = parse_ingredient(text)

        item_count   = 0

        for ingredient in parsed:
            amount = ingredient.amount  
            unit   = ingredient.unit

            if amount is None:
                item_count += 1
                continue

            normalized = self.normalize_quantities(amount, unit)

            if unit and unit.lower() in ("cup", "cups", "tbsp", "tsp", "ml", "l", "fl oz"):
                total_volume += normalized or 0.0
            elif unit and unit.lower() in ("g", "kg", "oz", "lb", "lbs"):
                total_weight += normalized or 0.0
            else:
                item_count += 1

        return [total_volume, total_weight, item_count]

    def filter_measurements(self, text):
        """
        Removes units from the raw text and returns
        the cleaned string for embedding. This 
        function is primarily for embedding 
        building.
        """
        # remove patterns like "2 cups", "1.5 lbs", "350°F", "45 min"
        text = re.sub(r'\d+\.?\d*\s*(cups?|tbsp|tsp|oz|lbs?|g|kg|ml|°[FCfc]|min|minutes?|hours?)', '', text)
        text = re.sub(r'\b\d+\.?\d*\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fit_transform(self, texts):
        X = self.build_features(texts)
        return self.scaler.fit_transform(X)  # call on train set

    def transform(self, texts):
        X = self.build_features(texts)
        return self.scaler.transform(X)
