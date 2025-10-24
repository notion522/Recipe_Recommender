import gradio as gr
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from recommender_utils import recommend_unique_titles
import gradio as gr

# Load data
vectorizer = joblib.load("../models/tf_vectorizer.pkl")
ingredient_vectors = joblib.load("../models/tf_ingredient_vectors.pkl")
recipes_df = pd.read_csv("../models/recipes_meta.csv")


def recommend_fn(ingredients):
    return "\n".join(recommend_unique_titles(ingredients))

iface = gr.Interface(fn=recommend_fn, inputs="text", outputs="text", title="🥣 Recipe Recommender")
iface.launch()
