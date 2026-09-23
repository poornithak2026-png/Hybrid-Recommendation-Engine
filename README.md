# Hybrid-Recommendation-Engine
A dynamic hybrid recommendation engine blending collaborative and content-based filtering to solve the cold-start problem.
# Hybrid Recommendation Engine with Cold-Start Handling[cite: 6]

**AWS Student Builder Group - AI/ML Project**[cite: 4, 6]

## Scenario & Objective
A streaming platform needs to recommend content to users, but naive collaborative filtering fails when new users and new items have no interaction history[cite: 6]. The objective of this project is to build a hybrid recommender system that blends collaborative filtering with content-based features[cite: 6]. Instead of breaking on new users, the system degrades gracefully for cold-start cases by dynamically shifting weights based on user interaction history[cite: 6].

## Dataset
This project utilizes the **MovieLens 1M** dataset for faster iteration and model training. 

## Key Features & Approach
*   **Simulated Cold-Start Slice:** Since all MovieLens 1M users naturally have at least 20 ratings, a cold-start slice was explicitly constructed by restricting 10% of users to fewer than 5 interactions (<5 ratings) during training[cite: 2, 5].
*   **Collaborative Filtering (CF):** Implemented a matrix factorization base model using Singular Value Decomposition (SVD) via the `scikit-surprise` library[cite: 2, 5].
*   **Content-Based Filtering (CB):** Built an item-to-item similarity matrix by applying TF-IDF vectorization and Cosine Similarity to movie metadata (genres) using `scikit-learn`[cite: 2, 5].
*   **Dynamic Blending Strategy:** The engine abandons a fixed 50/50 mix[cite: 5]. Instead, it calculates the user's history length and dynamically shifts the scoring weight entirely toward the content-based component for cold users, gradually introducing the collaborative filtering scores as their interaction history grows[cite: 3, 5].

## Repository Structure
*   `Hybrid_Recommendation_Engine.ipynb`: The primary Google Colab notebook containing the complete pipeline (data loading, preprocessing, model training, evaluation, and write-up)[cite: 2, 4].
*   `README.md`: Project documentation and setup instructions.

## Setup & Dependency Installation
This project was developed and tested in Google Colab to avoid local environment and policy restrictions. To run this locally or in a fresh Colab environment, install the following dependencies:

```bash
pip install pandas numpy scikit-learn scikit-surprise
