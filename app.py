# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess data
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, sheet_name='English_version', header=None)
    df.columns = df.iloc[1]
    df = df.drop(index=[0, 1]).reset_index(drop=True)
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    
    numeric_cols = [
        'price', 'cliped_count', 'cooking_time', 'ghg_total', 'ghg_production', 
        'ghg_disposal', 'ghg_cooking',
        'energy_(g)', 'fat_(g)', 'carbohydrates_(g)', 'zinc_(mg)', 'folic_acid_(μg)', 
        'protein_(g)', 'total_fiber_(g)', 'vitamin_a_(μg)', 'vitamin_c_(mg)', 
        'vitamin_e_(mg)', 'calcium_(mg)', 'iron_(mg)', 'potassium_(mg)', 
        'magnesium_(mg)', 'saturated_fat_(g)', 'cholesterol_(g)', 'salt_equivalent_(g)'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
    def safe_eval(x):
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            return x
        except (ValueError, SyntaxError):
            return {} if isinstance(x, str) else x
    
    for col in ['nutrition', 'keywords', 'ingredients', 'modified_ingredients', 'ingredients_category', 'cooking_method','disposal_amount']:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval)
    
    scaler = MinMaxScaler()
    df[['protein_score', 'fiber_score']] = scaler.fit_transform(df[['protein_(g)', 'total_fiber_(g)']])
    df[['fat_score', 'salt_score']] = 1 - scaler.fit_transform(df[['fat_(g)', 'salt_equivalent_(g)']])
    
    df['nutrient_score'] = (
        0.4 * df['protein_score'] +
        0.3 * df['fiber_score'] +
        0.2 * df['fat_score'] +
        0.1 * df['salt_score']
    )
    
    df['combined_text'] = (
        df['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('') + ' ' +
        df['recipe_description'].fillna('')
    ).str.lower()
    
    df = df.dropna(subset=['recipe_name', 'price']).reset_index(drop=True)
    return df

# Score recipes function
def score_recipes(df, weights, max_ghg, max_price, min_protein):
    df = df.copy()
    for col in ['ghg_total', 'price', 'protein_(g)']:
        if df[col].max() > df[col].min():
            df[f'normalized_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[f'normalized_{col}'] = 0.0
    
    df['score'] = (
        -weights['ghg'] * df['normalized_ghg_total'] +
        -weights['price'] * df['normalized_price'] +
        weights['protein'] * df['normalized_protein_(g)'] +
        weights['nutrient'] * df['nutrient_score']
    )
    
    df['score'] = df['score'].where(
        (df['ghg_total'] <= max_ghg) & 
        (df['price'] <= max_price) & 
        (df['protein_(g)'] >= min_protein),
        -np.inf
    )
    
    return df.sort_values(by='score', ascending=False)

# Recommendation function
def recommend_recipes(
    df,
    max_ghg,
    max_price,
    min_protein,
    max_energy,
    cuisine,
    max_cooking_time,
    selected_nutrients,
    keywords,
    cooking_method,
    dish_type,
    weights,
    top_n
):
    filtered_df = df.copy()
    
    if max_ghg:
        filtered_df = filtered_df[filtered_df['ghg_total'] <= max_ghg]
    if max_price:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]
    if min_protein:
        filtered_df = filtered_df[filtered_df['protein_(g)'] >= min_protein]
    if max_energy:
        filtered_df = filtered_df[filtered_df['energy_(g)'] <= max_energy]
    if cuisine and cuisine != "All":
        filtered_df = filtered_df[filtered_df['recipe_cuisine'] == cuisine]
    if max_cooking_time:
        filtered_df = filtered_df[filtered_df['cooking_time'] <= max_cooking_time]
    if cooking_method and cooking_method != "All":
        filtered_df = filtered_df[filtered_df['cooking_method'].apply(
            lambda x: cooking_method in x if isinstance(x, list) else cooking_method == x)]
    if dish_type and dish_type != "All":
        filtered_df = filtered_df[filtered_df['dish'] == dish_type]
    
    for nutrient, min_value in selected_nutrients.items():
        if min_value:
            filtered_df = filtered_df[filtered_df[nutrient] >= min_value]
    
    if keywords:
        for kw in keywords:
            filtered_df = filtered_df[filtered_df['combined_text'].str.contains(kw, case=False, na=False)]
    
    if not filtered_df.empty:
        filtered_df = score_recipes(filtered_df, weights, max_ghg, max_price, min_protein)
    
    return filtered_df.head(top_n)

# Streamlit UI
st.title("🌿 Nutrient & Price-Conscious Recipe Recommender")

df = load_data('/content/recipe_data_with_Eng_name.xlsx')

st.sidebar.header("Filter Recipes")

max_ghg = st.sidebar.slider("Maximum GHG Total (kg CO2e)", 0.0, float(df['ghg_total'].max()), 1000.0, step=10.0)
max_price = st.sidebar.slider("Maximum Price (JPY)", 0.0, float(df['price'].max()), 500.0, step=10.0)
min_protein = st.sidebar.slider("Minimum Protein (g)", 0.0, float(df['protein_(g)'].max()), 0.0, step=1.0)
max_energy = st.sidebar.slider("Maximum Energy (g)", 0.0, float(df['energy_(g)'].max()), 500.0, step=10.0)

st.sidebar.subheader("Additional Nutrient Filters")
nutrient_options = [
    'total_fiber_(g)', 'vitamin_c_(mg)', 'calcium_(mg)', 'iron_(mg)', 
    'potassium_(mg)', 'magnesium_(mg)'
]
selected_nutrients = {}
for nutrient in nutrient_options:
    min_value = st.sidebar.slider(f"Minimum {nutrient.replace('_', ' ').title()}", 
                                  0.0, float(df[nutrient].max()), 0.0, step=1.0)
    selected_nutrients[nutrient] = min_value

cuisines = ['All'] + sorted(df['recipe_cuisine'].dropna().unique().tolist())
cuisine = st.sidebar.selectbox("Cuisine Type", cuisines)

max_cooking_time = st.sidebar.slider("Maximum Cooking Time (minutes)", 
                                     0, int(df['cooking_time'].max()), 30, step=5)

cooking_methods_list = df['cooking_method'].dropna().apply(
    lambda x: x if isinstance(x, list) else [x]
)
cooking_methods = sorted(set([method for sublist in cooking_methods_list for method in sublist if isinstance(method, str)]))
cooking_method = st.sidebar.selectbox("Cooking Method", ['All'] + cooking_methods)

dish_types = ['All'] + sorted(df['dish'].dropna().unique().tolist())
dish_type = st.sidebar.selectbox("Dish Type", dish_types)

user_keywords = st.text_input("Enter recipe keywords (e.g., 'vegan curry spicy'):", "").strip()
keywords = [kw.strip().lower() for kw in user_keywords.split()] if user_keywords else []

st.sidebar.subheader("Preference Weights")
weight_ghg = st.sidebar.slider("Importance of Low GHG", 0.0, 1.0, 0.3, step=0.1)
weight_price = st.sidebar.slider("Importance of Low Price", 0.0, 1.0, 0.3, step=0.1)
weight_protein = st.sidebar.slider("Importance of High Protein", 0.0, 1.0, 0.3, step=0.1)
weight_nutrient = st.sidebar.slider("Importance of Nutrient Score", 0.0, 1.0, 0.1, step=0.1)
weights = {'ghg': weight_ghg, 'price': weight_price, 'protein': weight_protein, 'nutrient': weight_nutrient}

top_n = st.slider("Number of recommendations to show", 1, 20, 5)

if st.button("Get Recommendations"):
    if not any([max_ghg, max_price, min_protein, max_energy, cuisine != "All", 
                max_cooking_time, any(v > 0 for v in selected_nutrients.values()), keywords, cooking_method != "All", dish_type != "All"]):
        st.warning("Please provide at least one filter or keyword.")
    else:
        results = recommend_recipes(
            df, max_ghg, max_price, min_protein, max_energy, cuisine,
            max_cooking_time, selected_nutrients, keywords, cooking_method,
            dish_type, weights, top_n
        )
        if results.empty:
            st.warning("No recipes match your criteria.")
        else:
            st.success(f"Top {len(results)} recommended recipes:")
            for _, row in results.iterrows():
                st.markdown(f"### {row['recipe_name']}")
                st.write(f"**Cuisine:** {row['recipe_cuisine']}")
                st.write(f"**Dish Type:** {row['dish']}")
                st.write(f"**Price:** ¥{row['price']:.2f}")
                st.write(f"**GHG Total:** {row['ghg_total']:.2f} kg CO2e")
                st.write(f"**Protein:** {row['protein_(g)']:.2f} g")
                st.write(f"**Cooking Time:**{row['cooking_time']} min")
                st.write(f"Description: {row['recipe_description']}")
                st.markdown("---")



