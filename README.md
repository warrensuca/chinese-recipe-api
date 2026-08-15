# Chinese Recipe Recommendation API

A machine learning-centred REST API that clusters and recommends Chinese recipes scraped from [Omnivore's Kitchen](https://omnivorescookbook.com/category/recipe/) and [The Woks of Life](https://thewoksoflife.com/category/recipes/) based on their nutritional profiles and ingredient similarities.

**Live URL:** [https://jian-api.onrender.com/](https://jian-api.onrender.com/)

## Features
* **Nutritional Clustering:** Recipes are categorized into distinct clusters based on 7 nutritional features.
* **Hybrid Recommendation Engine:** Recommends similar recipes by blending nutritional distance (Euclidean) and ingredient overlap (Jaccard similarity).
* **Targeted Search:** Input custom macro/micro nutritional values (Calories, Protein, Fat, etc.) to get nearest-neighbor recipe recommendations.
* **Healthier Ingredient Alternatives:** Recommends healthier ingredient substitutes (higher protein, lower calorie, lower carb) using sentence transformer embeddings and a novel Health-Similarity Score.


## Tech Stack

* **Framework:** FastAPI
* **Data Acquisition:** Beautiful Soup
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning & Embeddings:** Scikit-Learn, SentenceTransformers (`all-MiniLM-L6-v2`), Joblib
* **Deployment:** Render

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API Root / Health Check |
| `GET` | `/clusters` | Returns a summary of all recipe clusters and their counts |
| `GET` | `/clusters/{cluster_id}` | Returns all recipes belonging to a specific cluster |
| `GET` | `/recipes/{recipe_name}` | Fetches detailed information for a specific recipe |
| `GET` | `/recommend/{recipe_name}` | Recommends top 10 similar recipes based on a given recipe |
| `GET` | `/nearest-cluster/{recipe_name}` | Predicts and returns the closest cluster for a recipe |
| `GET` | `/recommend-by-nutrition` | Recommends recipes closest to user-provided nutritional values |
| `GET` | `/recommend-by-weighted_nutrition` | Weighted nutritional distance recommendation |
| `GET` | `/healthier/higher-protein/{ingredient_name}` | Recommends higher protein alternative ingredients |
| `GET` | `/healthier/lower-calorie/{ingredient_name}` | Recommends lower calorie alternative ingredients |
| `GET` | `/healthier/lower-carb/{ingredient_name}` | Recommends lower carbohydrate alternative ingredients |

### Healthier Ingredient Recommendation Query Parameters

All `/healthier/*` endpoints support the following optional query parameters:
* `similarity_weight` *(float, default: 0.5)*: Weight $w \in [0, 1]$ given to semantic embedding similarity versus nutritional improvement.
* `top_k` *(int, default: 10)*: Number of top recommendations to return.

### Novel Health-Similarity Score (HSS) Formula

To balance semantic and culinary similarity with nutritional improvement, recommendations are scored using a novel **Health-Similarity Score ($HSS$)**:

1. **Embedding Cosine Similarity ($Sim$)**:
   $$\text{Sim}(T, j) = \frac{v_T \cdot v_j}{\|v_T\|_2 \|v_j\|_2}$$
   where $v_T, v_j \in \mathbb{R}^{384}$ are 384-dimensional sentence transformer vectors (`all-MiniLM-L6-v2`) for target ingredient $T$ and candidate alternative $j$.

2. **Sigmoid-Normalized Health Score ($\text{HealthScore}_H$)**:
   $$\Delta_H(T, j) = \frac{\text{Metric}_H(j) - \text{Metric}_H(T)}{\sigma_H}$$
   $$\text{HealthScore}_H(T, j) = \frac{1}{1 + e^{-\Delta_H(T, j)}}$$
   where $\sigma_H$ is the dataset standard deviation for health objective $H \in \{\text{higher\_protein}, \text{lower\_calorie}, \text{lower\_carb}\}$.

3. **Combined Balanced Score ($HSS_H$)**:
   $$HSS_H(T, j; w) = w \cdot \max(0, \text{Sim}(T, j)) + (1 - w) \cdot \text{HealthScore}_H(T, j)$$

