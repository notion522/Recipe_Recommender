import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load Pickle
# ---------------------------
pickle_path = "C:/Users/a.nimse/Desktop/Recipe_Recommender-1/models/recipe_recommender.pkl"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

vectorizer = data["vectorizer"]
ingredient_vectors = data["ingredient_vectors"]
df = data["df"]

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend_recipes(user_ingredients, top_n=5):
    user_input = ' '.join(user_ingredients).lower()
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, ingredient_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_scores = similarities[top_indices]
    
    recommendations = df.iloc[top_indices][[
        'recipe_title', 'cleaned_ingredients', 'description', 'directions'
    ]].copy()
    
    recommendations['similarity'] = top_scores.round(3)
    recommendations = recommendations[recommendations['similarity'] > 0.1]
    return recommendations

def recommend_with_missing(user_ingredients, top_n=3):
    recs = recommend_recipes(user_ingredients, top_n=top_n)
    if recs.empty:
        return []
    
    recommendations_with_missing = []
    user_ingredients_set = set(user_ingredients)
    
    for _, row in recs.iterrows():
        recipe_ingredients = set(row['cleaned_ingredients'])
        missing = list(recipe_ingredients - user_ingredients_set)
        recommendations_with_missing.append({
            "recipe_title": row['recipe_title'],
            "similarity": row['similarity'],
            "missing_ingredients": missing,
            "description": row['description'],
            "directions": row['directions']
        })
    return recommendations_with_missing

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🍳 Recipe Recommender", layout="wide")
st.title("🍽️ Smart Recipe Recommender")
st.write("Enter the ingredients you have, and we’ll recommend the best matching recipes!")

user_input = st.text_input("Enter ingredients (comma separated):", "chicken, garlic, soy sauce")

if st.button("Find Recipes"):
    user_ingredients = [i.strip().lower() for i in user_input.split(",") if i.strip()]
    
    if not user_ingredients:
        st.warning("Please enter at least one ingredient.")
    else:
        results = recommend_with_missing(user_ingredients, top_n=3)
        
        if not results:
            st.error("No recipes found. Try adding more ingredients!")
        else:
            for i, r in enumerate(results, 1):
                st.subheader(f"{i}. {r['recipe_title']}  ({r['similarity']*100:.1f}% match)")
                
                if r['missing_ingredients']:
                    st.warning(f"Missing ingredients: {', '.join(r['missing_ingredients'])}")
                else:
                    st.success("✅ You have all ingredients!")
                
                st.write(f"**Description:** {r['description']}")
                st.write(f"**Directions:** {r['directions']}")
                st.markdown("---")
