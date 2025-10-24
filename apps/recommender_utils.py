import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Load all model assets from /models folder
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

VECTORIZER_PATH = MODELS_DIR / "tf_vectorizer.pkl"
VECTORS_PATH = MODELS_DIR / "tf_ingredient_vectors.pkl"
META_PATH = MODELS_DIR / "recipes_meta.csv"

# --- Load models and metadata ---
vectorizer = joblib.load(VECTORIZER_PATH)
ingredient_vectors = joblib.load(VECTORS_PATH)
recipes_meta = pd.read_csv(META_PATH)

# --- Core Recommendation Function ---
def recommend_unique_titles(ingredients, top_n=5):
    """
    Recommend top recipes given an ingredient list.
    Ensures unique, most relevant titles are returned.
    """
    if not ingredients or not isinstance(ingredients, str):
        return ["Please enter valid ingredients"]

    # Vectorize user input
    input_vec = vectorizer.transform([ingredients])

    # Compute cosine similarity
    sim_scores = cosine_similarity(input_vec, ingredient_vectors).flatten()

    # Sort by similarity (descending)
    top_indices = np.argsort(sim_scores)[::-1]

    # Get top recipes while ensuring uniqueness
    seen_titles = set()
    top_recipes = []
    for idx in top_indices:
        title = recipes_meta.iloc[idx]["recipe_title"]
        if title not in seen_titles:
            seen_titles.add(title)
            top_recipes.append(title)
        if len(top_recipes) >= top_n:
            break

    if not top_recipes:
        return ["No similar recipes found. Try different ingredients."]

    return top_recipes
