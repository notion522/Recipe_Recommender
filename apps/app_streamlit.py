import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from recommender_utils import recommend_unique_titles

# Load saved data
vectorizer = joblib.load("../models/tf_vectorizer.pkl")
ingredient_vectors = joblib.load("../models/tf_ingredient_vectors.pkl")
recipes_df = pd.read_csv("../models/recipes_meta.csv")

# Streamlit UI
st.title("🥣 Recipe Finder by Ingredients")
st.write("Enter ingredients separated by commas or spaces to find the most similar recipes.")

ingredients = st.text_input("Enter ingredients (comma-separated):")
if st.button("Find Recipes"):
    results = recommend_unique_titles(ingredients)
    st.subheader("🔍 Top Matching Recipes:")
    for r in results:
        st.write(f"- {r}")