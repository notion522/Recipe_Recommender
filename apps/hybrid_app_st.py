import streamlit as st
from hybrid_utils import load_models, recommend_recipes

@st.cache_resource
def get_models():
    return load_models()

vectorizer, svd, index, df = get_models()

st.title("🍲 Hybrid Recipe Recommender")
st.markdown("Find recipes using your available ingredients and optional cooking steps.")

ingredients_input = st.text_area(
    "Enter ingredients (comma-separated):",
    placeholder="e.g., chicken, soy sauce, honey"
)

directions_input = st.text_area(
    "Optional: cooking directions or actions (comma-separated):",
    placeholder="e.g., grill, marinate, bake"
)


top_k = 20

if st.button("🔍 Recommend Recipes"):
    if ingredients_input.strip():
        input_ingredients = [x.strip() for x in ingredients_input.split(",") if x.strip()]
        input_directions = [x.strip() for x in directions_input.split(",") if x.strip()] if directions_input else None

        results_df = recommend_recipes(vectorizer, svd, index, df, input_ingredients, input_directions, top_k)
        results_df = results_df.drop_duplicates(subset=["Recipe Title"]).head(top_k).reset_index(drop=True)


        # Display results as cards
        st.write("### 🥗 Top Recommended Recipes")
        for _, row in results_df.iterrows():
            with st.expander(f"🍴 {row['Recipe Title']} — Similarity: {row['Similarity']}"):
                st.markdown(f"**Matched Ingredients:** {row['Matched Ingredients']} / {row['Total Ingredients']}")
                st.markdown(f"**Missing Ingredients:** {row['Missing Ingredients'] or 'None'}")
                st.markdown("**Directions:**")
                st.write(row['Directions'])
    else:
        st.warning("⚠️ Please enter at least one ingredient.")