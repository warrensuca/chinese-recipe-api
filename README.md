# Chinese Recipe Recommendation API

A machine learning-powered REST API that clusters and recommends Chinese recipes based on their nutritional profiles and ingredient similarities.

**Live URL:** [https://chinese-recipe-api.vercel.app/](https://chinese-recipe-api.vercel.app/)

## Features

* **Nutritional Clustering:** Recipes are categorized into distinct clusters based on 7 nutritional features.
* **Hybrid Recommendation Engine:** Recommends similar recipes by blending nutritional distance (Euclidean) and ingredient overlap (Jaccard similarity).
* **Targeted Search:** Input custom macro/micro nutritional values (Calories, Protein, Fat, etc.) to get nearest-neighbor recipe recommendations.


## Tech Stack

* **Framework:** FastAPI
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Deployment:** Vercel

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API Root / Health Check |
| `GET` | `/clusters` | Returns a summary of all recipe clusters and their counts |
| `GET` | `/clusters/{cluster_id}` | Returns all recipes belonging to a specific cluster |
| `GET` | `/recipe/{recipe_name}` | Fetches detailed information for a specific recipe |
| `GET` | `/recommend/{recipe_name}` | Recommends top 10 similar recipes based on a given recipe |
| `GET` | `/nearest-cluster/{recipe_name}` | Predicts and returns the closest cluster for a recipe |
| `GET` | `/recommend-by-nutrition` | Recommends recipes closest to user-provided nutritional values |
