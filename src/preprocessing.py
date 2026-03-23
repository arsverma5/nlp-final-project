# pip install pint ingredient_parser_nlp scikit-learn scipy
from pint import UnitRegistry
from ingredient_parser import parse_ingredient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

import numpy as np
import re
import pandas as pd



# Time bucket definitions
TIME_BINS   = [0, 30, 60, 120, float("inf")]
TIME_LABELS = ["Quick (<=30 min)", "Medium (31-60 min)", "Long (61-120 min)", "Very Long (>120 min)"]


class RecipeProcessor:
    def __init__(self, n_components=100, tfidf_max_features=5000):
        self.scaler = StandardScaler(with_mean=False)  
        self.pca = PCA(n_components=n_components)
        self.vectorizer  = TfidfVectorizer(max_features=tfidf_max_features)
        self.label_encoder = LabelEncoder()
        self.ureg = UnitRegistry()

        # map units to stem for text processing
        self._unit_map = {
            "cup": "cup", "cups": "cup",
            "tbsp": "tablespoon", "tablespoon": "tablespoon", "tablespoons": "tablespoon",
            "tsp": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
            "oz": "ounce", "ounces": "ounce",
            "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
            "g": "gram", "gram": "gram", "grams": "gram",
            "kg": "kilogram",
            "ml": "milliliter", "mL": "milliliter",
            "l": "liter", "L": "liter",
            "fl oz": "fluid_ounce",
        }
        self._base_units = {
            "VOLUME": self.ureg.milliliter,
            "WEIGHT": self.ureg.gram,
        }

    # Quantity parsing + normalization
    def normalize_quantity(self, amount: float, unit_str: str):
        """
        Convert an ingredient amount + unit string into a base unit value.
        Volume -> milliliters, weight -> grams. Returns None if unrecognized.
        """
        pint_unit = self._unit_map.get(unit_str.lower().strip())
        if pint_unit is None:
            return None

        try:
            qty = amount * self.ureg.parse_expression(pint_unit)
            if qty.dimensionality == self.ureg.milliliter.dimensionality:
                return qty.to(self._base_units["VOLUME"]).magnitude
            elif qty.dimensionality == self.ureg.gram.dimensionality:
                return qty.to(self._base_units["WEIGHT"]).magnitude
            return qty.magnitude
        except Exception:
            return None

    def parse_ingredient_text(self, text: str) -> list[dict]:
        """
        Parse a raw ingredient string into a list of dicts with keys:
        name, quantity, unit.
        """
        parsed = parse_ingredient(text)
        ingredients = []

        for i, name_obj in enumerate(parsed.name):
            name = name_obj.text if name_obj else None
            quantity, unit = None, None

            if i < len(parsed.amount):
                amt = parsed.amount[i]
                if amt.quantity:
                    quantity = float(amt.quantity)
                if amt.unit:
                    unit = str(amt.unit)

            ingredients.append({"name": name, "quantity": quantity, "unit": unit})

        return ingredients

    def extract_ingredient_features(self, ingredients: list[dict]) -> dict:
        """
        Reduce a parsed ingredient list to feature-ready structure:
        ingredient names (lowercased), count, and normalized quantities.
        """
        names, quantities = [], []

        for ing in ingredients:
            if ing["name"]:
                names.append(ing["name"].lower())
            if ing["quantity"] and ing["unit"]:
                normalized = self.normalize_quantity(ing["quantity"], ing["unit"])
                if normalized is not None:
                    quantities.append(normalized)

        return {
            "ingredients": names,
            "ingredient_count": len(names),
            "quantities": quantities,
        }

    def process_ingredients_column(self, ingredient_series) -> list[dict]:
        """
        Apply full ingredient parsing pipeline to a pandas Series of raw strings.
        default empty features on parse errors.
        """
        results = []
        for text in ingredient_series:
            try:
                parsed = self.parse_ingredient_text(text)
                results.append(self.extract_ingredient_features(parsed))
            except Exception:
                results.append({"ingredients": [], "ingredient_count": 0, "quantities": []})
        return results

    # Text cleaning

    # Match explicit time expressions, ex: "bake for 30 minutes"
    _TIME_EXPR = re.compile(
        r'(?:'
        # numeric: "for 30 minutes", "1-2 hours", "3 days"
        r'\b(?:for\s+)?\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\d+)?\s*'
        r'(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?)\b'
        r'|'
        # standalone time words w/ no leading number
        r'\b(?:(?:for\s+)?(?:a\s+)|all\s+)?(?:overnight|day|night)\b'
        r')',
        re.IGNORECASE,
    )

    def count_time_mentions(self, text: str) -> int:
        """
        Count how many explicit time expressions appear in a directions string.
        Call this BEFORE clean_directions, becomes scalar feature. 
        """
        if not isinstance(text, str):
            return 0
        return len(self._TIME_EXPR.findall(text))

    def clean_directions(self, text: str) -> str:
        """
        Remove explicit time expressions from directions text so the model
        learns from semantic data rather than directly reading the answer.
        Also strips standalone numbers + normalises whitespace.
        """
        if not isinstance(text, str):
            return ""
        text = self._TIME_EXPR.sub("", text)
        text = re.sub(r'\b\d+\.?\d*\b', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def filter_measurements(self, text: str) -> str:
        """
        Strip numeric quantities and common unit tokens from an ingredient
        string, leaving only descriptive terms for embedding
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(
            r'\d+\.?\d*\s*(cups?|tbsp|tsp|oz|lbs?|g|kg|ml|degrees?|°[FCfc]|min|minutes?|hours?)',
            '', text,
        )
        text = re.sub(r'\b\d+\.?\d*\b', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def build_corpus(self, ingredients_series, directions_series) -> list[str]:
        """
        Concat cleaned ingredients + cleaned directions into a single
        document string per recipe for TF-IDF vectorization.
        """
        corpus = []
        for ing, direc in zip(ingredients_series, directions_series):
            clean_ing   = self.filter_measurements(str(ing))
            clean_direc = self.clean_directions(str(direc))
            corpus.append(f"{clean_ing} {clean_direc}".strip())
        return corpus

    # Time parsing + bucketing
    def parse_total_time(self, text: str):
        """
        Parse the default recipe timing string like:
          "Prep Time: 30 mins, Cook Time: 1 hrs, Total Time: 1 hrs 30 mins"
        into a total time integer
        Returns None if no time information is found or input is not a string.
        """
        if not isinstance(text, str):
            return None

        text_lower = text.lower()
        total_match = re.search(r'total time[:\s]+([^,]+)', text_lower)
        segment = total_match.group(1) if total_match else text_lower

        hours = re.search(r'(\d+)\s*hr', segment)
        minutes = re.search(r'(\d+)\s*min', segment)

        total = 0
        if hours:
            total += int(hours.group(1)) * 60
        if minutes:
            total += int(minutes.group(1))

        return total if total > 0 else None

    @staticmethod
    def bucketize_time(minutes_series):
        """
        Convert a numeric minutes Series into ordered time-bucket labels.

        Bins:
            Quick : <= 30 min
            Medium : 31 - 60 min
            Long : 61 - 120 min
            Very Long: > 120 min
        """
        return pd.cut(
            minutes_series,
            bins=TIME_BINS,
            labels=TIME_LABELS,
            right=True,
        )

    # TF-IDF vectorization
    def fit_tfidf(self, texts):
        """Fit the TF-IDF vectorizer on texts and return the document-term matrix."""
        return self.vectorizer.fit_transform(texts)

    def transform_tfidf(self, texts):
        """Transform texts with the already-fitted TF-IDF vectorizer."""
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return TF-IDF vocabulary terms (useful for feature importance plots)."""
        return self.vectorizer.get_feature_names_out().tolist()

    # Scalar feature helpers
    @staticmethod
    def build_scalar_features(df) -> np.ndarray:
        """
        Assemble numeric scalar features from the dataframe:
          - time_mention_count : explicit time expressions found in directions
          - ingredient_count   : number of distinct ingredients parsed

        Both columns must already exist on df before calling this.
        Returns a (n_samples, n_scalar_features) numpy array.
        """
        cols = [c for c in ["time_mention_count", "ingredient_count"] if c in df.columns]
        return df[cols].fillna(0).to_numpy(dtype=float)

    @staticmethod
    def hstack_features(tfidf_matrix, scalar_array):
        """
        Horizontally stack a sparse TF-IDF matrix with a dense scalar array
        into one sparse feature matrix ready for sklearn models.
        """
        return hstack([tfidf_matrix, csr_matrix(scalar_array)])