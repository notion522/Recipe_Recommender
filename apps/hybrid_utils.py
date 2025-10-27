import joblib
import faiss
import pandas as pd
import numpy as np
import re

def load_models():
    vectorizer = joblib.load("../models/hybrid_vectorizer.pkl")
    svd = joblib.load("../models/hybrid_svd.pkl")
    index = faiss.read_index("../models/hybrid_faiss_index.faiss")
    df = pd.read_csv("../models/recipes_meta.csv")
    return vectorizer, svd, index, df

units = ['cup', 'cups', 'pound', 'pounds', 'tablespoon', 'tablespoons',
         'teaspoon', 'teaspoons', 'oz', 'ounce', 'ounces', 'gram', 'grams',
         'kg', 'ml', 'liter', 'liters', 'pinch', 'dash']

def clean_ingredient_list(ingredient_list):
    cleaned = []
    for ing in ingredient_list:
        ing = ing.lower()
        ing = re.sub(r'[^a-zA-Z ]', '', ing)
        words = ing.split()
        filtered = [word for word in words if word not in units]
        cleaned.append(' '.join(filtered))
    return ' '.join(cleaned)


def recommend_recipes(vectorizer, svd, index, recipes_df, input_ingredients, input_directions=None, top_k=5):
    # Clean and combine input
    input_clean = clean_ingredient_list(input_ingredients)
    input_text = input_clean
    if input_directions:
        input_text += ' ' + ' '.join(input_directions)

    input_set = set(input_clean.split())

    # Vectorize + reduce + normalize
    input_vector = vectorizer.transform([input_text]).astype(np.float32)
    input_reduced = svd.transform(input_vector)
    faiss.normalize_L2(input_reduced)

    # FAISS search
    distances, indices = index.search(input_reduced, top_k)

    results = []
    for idx, sim in zip(indices[0], distances[0]):
        recipe = recipes_df.iloc[idx]
        recipe_ing = recipe['clean_ingredients']
        recipe_set = set(str(recipe_ing).split())

        missing = recipe_set - input_set
        matched_count = len(recipe_set & input_set)
        total_count = len(recipe_set)

        results.append({
            "Recipe Title": recipe['recipe_title'],
            "Similarity": round(float(sim), 3),
            "Matched Ingredients": matched_count,
            "Total Ingredients": total_count,
            "Missing Ingredients": ', '.join(missing),
            "Directions": recipe.get('directions_clean', '')
        })

    return pd.DataFrame(results).sort_values(by='Similarity', ascending=False)
