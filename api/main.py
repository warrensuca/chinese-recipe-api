from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import ast
import joblib 
from typing import Optional
import os

app = FastAPI(title="Recipe Recommendation API")

#load data and models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


csv_path = os.path.join(BASE_DIR, "../clustering/clustered_recipes.csv")

scaler_path = os.path.join(BASE_DIR, "../clustering/scaler.joblib")
kmeans_path = os.path.join(BASE_DIR, "../clustering/kmeans.joblib")


df = pd.read_csv(csv_path)

scaler = joblib.load(scaler_path) 
kmeans = joblib.load(kmeans_path)

nutrition_features = [
    "Calories", "Carbohydrates", "Protein", "Fat", 
    "Saturated Fat", "Sodium", "Sugar"
]

scaled_features = [
    "Scaled_Calories", "Scaled_Carbohydrates", "Scaled_Protein", 
    "Scaled_Fat", "Scaled_Saturated Fat", "Scaled_Sodium", "Scaled_Sugar"
]

scaled_nutrition = df[scaled_features].to_numpy()


dataset_medians = df[nutrition_features].median().to_numpy()


#helper functions

async def parse_ingredients(value):
    try:
        ingredients = ast.literal_eval(value)
        if isinstance(ingredients, list):
            return set(str(x).strip().lower() for x in ingredients)
        return set()
    except:
        return set()

async def jaccard_similarity(set_a, set_b):
    union = set_a | set_b
    if len(union) == 0:
        return 0
    return len(set_a & set_b) / len(union)

async def nutrition_similarity(index_a, index_b):
    distance = np.linalg.norm(scaled_nutrition[index_a] - scaled_nutrition[index_b])
    return 1 / (1 + distance)

async def recipe_similarity(index_a, index_b):
    nutrition_score = nutrition_similarity(index_a, index_b)
    ingredients_a = parse_ingredients(df.iloc[index_a]["Ingredients_Names"])
    ingredients_b = parse_ingredients(df.iloc[index_b]["Ingredients_Names"])
    ingredient_score = jaccard_similarity(ingredients_a, ingredients_b)
    
    return (0.8 * nutrition_score) + (0.2 * ingredient_score)


# endpoints


@app.get("/")
async def root():
    return {"message": "Recipe Recommendation API"}


@app.get("/clusters")
async def get_clusters():
    clusters = (
        df.groupby(["Cluster", "Cluster_Name"])
        .size()
        .reset_index(name="Recipe_Count")
    )
    return clusters.to_dict("records")


@app.get("/clusters/{cluster_id}")
async def get_cluster_recipes(cluster_id: int):
    recipes = df[df["Cluster"] == cluster_id]
    if len(recipes) == 0:
        raise HTTPException(status_code=404, detail="Cluster not found")

    columns_to_return = ["Name", "Ingredients_List", "Procedure", "Cluster_Name", "Cluster"] + nutrition_features
    return recipes[columns_to_return].to_dict("records")


@app.get("/recipe/{recipe_name}")
async def get_recipe(recipe_name: str):
    recipe = df[df["Name"] == recipe_name]
    if len(recipe) == 0:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe.iloc[0].to_dict()


@app.get("/recommend/{recipe_name}")
async def recommend(recipe_name: str):
    matches = df[df["Name"] == recipe_name]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail="Recipe not found")

    recipe_index = matches.index[0]
    cluster_id = df.iloc[recipe_index]["Cluster"]
    cluster_indices = df[df["Cluster"] == cluster_id].index

    recommendations = []
    for idx in cluster_indices:
        if idx == recipe_index:
            continue
        score = recipe_similarity(recipe_index, idx)
        recommendations.append({
            "Name": df.iloc[idx]["Name"],
            "Cluster": int(df.iloc[idx]["Cluster"]),
            "Similarity": float(score)
        })

    recommendations = sorted(recommendations, key=lambda x: x["Similarity"], reverse=True)
    return recommendations[:10]


@app.get("/nearest-cluster/{recipe_name}")
async def nearest_cluster(recipe_name: str):
    recipe = df[df["Name"] == recipe_name]

    if len(recipe) == 0:
        raise HTTPException(status_code=404, detail="Recipe not found")

    recipe = recipe.iloc[0]
    recipe_vector = np.array([
        recipe["Scaled_Calories"], recipe["Scaled_Carbohydrates"],
        recipe["Scaled_Protein"], recipe["Scaled_Fat"],
        recipe["Scaled_Saturated Fat"], recipe["Scaled_Sodium"], recipe["Scaled_Sugar"]
    ]).reshape(1, -1)

    
    best_cluster = int(kmeans.predict(recipe_vector)[0])

    cluster_name = df[df["Cluster"] == best_cluster].iloc[0]["Cluster_Name"]

    return {
        "cluster": best_cluster,
        "cluster_name": cluster_name
    }


@app.get("/recommend-by-nutrition")
async def recommend_by_nutrition(
    calories: Optional[float] = None,
    carbohydrates: Optional[float] = None,
    protein: Optional[float] = None,
    fat: Optional[float] = None,
    saturated_fat: Optional[float] = None,
    sodium: Optional[float] = None,
    sugar: Optional[float] = None,
):
    user_values = {
        "Calories": calories, "Carbohydrates": carbohydrates,
        "Protein": protein, "Fat": fat,
        "Saturated Fat": saturated_fat, "Sodium": sodium, "Sugar": sugar,
    }

    #track which features the user actually provided
    provided_mask = []
    user_array = dataset_medians.copy() # start with median placeholders

    for i, feature in enumerate(nutrition_features):
        if user_values[feature] is not None:
            user_array[i] = user_values[feature]
            provided_mask.append(True)
        else:
            provided_mask.append(False)

    if not any(provided_mask):
        raise HTTPException(status_code=400, detail="Provide at least one nutrition value.")

    
    scaled_user_input = scaler.transform(user_array.reshape(1, -1))[0]

    # vectorized distacne calculation
    
    dataset_relevant_features = scaled_nutrition[:, provided_mask]
    user_relevant_features = scaled_user_input[provided_mask]

    
    diff = dataset_relevant_features - user_relevant_features
    distances = np.linalg.norm(diff, axis=1)

    
    # np.argsort returns the row indices sorted by closest distance
    top_indices = np.argsort(distances)[:10] 

    recommendations = []
    for idx in top_indices:
        row = df.iloc[idx]
        recommendations.append({
            "Name": row["Name"],
            "Cluster": int(row["Cluster"]),
            "Cluster_Name": row["Cluster_Name"],
            "Distance": float(distances[idx])
        })

    return recommendations