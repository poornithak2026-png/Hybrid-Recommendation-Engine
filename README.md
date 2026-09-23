# Hybrid Recommendation Engine

A dynamic hybrid recommendation system that combines collaborative filtering and content-based filtering to handle the cold-start problem in movie recommendations.

> **AWS Student Builder Group - AI/ML Project**

## Overview

Recommendation systems often struggle with new users who have little or no interaction history. Collaborative filtering depends heavily on user-item interactions, so its performance can be limited for cold-start users.

This project addresses the problem by combining:

- **Collaborative Filtering** using Singular Value Decomposition (SVD)
- **Content-Based Filtering** using TF-IDF and Cosine Similarity
- **Dynamic Weighting** based on the user's interaction history

Instead of using a fixed combination of both approaches, the system adjusts the contribution of each model according to how much information is available about the user.

## Objective

Build a recommendation engine that:

1. Provides personalized movie recommendations.
2. Combines collaborative and content-based approaches.
3. Handles users with limited interaction history.
4. Dynamically adjusts recommendation weights as more user interactions become available.

## Dataset

This project uses the **MovieLens 1M dataset**, which contains movie ratings and movie metadata.

The dataset is used to train the collaborative filtering model and to build content-based movie similarities.

### Cold-Start Simulation

The MovieLens 1M dataset normally contains users with sufficient rating history.

To simulate a cold-start scenario, **10% of users are restricted to fewer than 5 ratings during training**.

This allows the hybrid recommendation system to be evaluated under limited user-history conditions.

## Approach

### 1. Collaborative Filtering

Collaborative filtering learns user preferences from historical user-item interactions.

This project uses:

**Singular Value Decomposition (SVD)**

implemented using the `scikit-surprise` library.

The model learns latent relationships between users and movies based on rating patterns.

### 2. Content-Based Filtering

Content-based filtering recommends movies based on their characteristics.

For this project:

- Movie genres are converted into TF-IDF vectors.
- Cosine similarity is used to measure similarity between movies.
- Movies similar to those previously interacted with by a user can then be recommended.

### 3. Dynamic Hybrid Recommendation

The two recommendation approaches are combined dynamically.

The system considers the user's interaction history:

- **Cold-start users:** Content-based recommendations receive greater importance because there is limited collaborative information.
- **Users with more history:** Collaborative filtering gradually receives more influence as more interaction data becomes available.

This avoids relying on a fixed 50/50 combination for every user.

## System Workflow

```text
                MovieLens 1M Dataset
                         |
              +----------+----------+
              |                     |
              v                     v
      Collaborative           Content-Based
        Filtering                Filtering
              |                     |
             SVD              TF-IDF + Cosine
              |                     |
              +----------+----------+
                         |
                         v
                 Dynamic Weighting
                         |
                         v
              Hybrid Recommendations
