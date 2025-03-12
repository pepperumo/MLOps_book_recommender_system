# Book Recommender System Architecture

## Overview
This document outlines the simplified architecture of our book recommender system, which now focuses exclusively on collaborative filtering.

## System Components

### 1. Data Processing
- **`process_data.py`**: Handles the cleaning and processing of raw book and rating data
- **Inputs**: Raw book metadata and user ratings
- **Outputs**: Clean, processed datasets ready for feature extraction

### 2. Feature Engineering
- **`build_features.py`**: Creates the necessary features for the collaborative filtering model
- **Key Features**:
  - User-item matrix: Maps users to books they've rated
  - Book similarity matrix: Calculates similarity between books based on content features
  - Book ID and user ID mappings: Maps between original IDs and encoded IDs used by the model

### 3. Model Training
- **`train_model.py`**: Trains the collaborative filtering model
- **`train_model_collaborative.py`**: Contains the CollaborativeRecommender class implementation
- **`train_model_evaluate.py`**: Evaluates the collaborative model's performance
- **Training Process**:
  - Matrix factorization to learn latent factors for users and items
  - Hyperparameter tuning for optimal performance
  - Model validation using evaluation metrics

### 4. Prediction
- **`predict_model.py`**: Handles generating recommendations using the trained collaborative model
- **Recommendation Types**:
  - User recommendations: Personalized book suggestions for a specific user
  - Similar book recommendations: Finding books similar to a given book
  - Cold-start handling: Approach for new users with no rating history

### 5. Testing
- **`test_prediction.py`**: Tests the recommendation functionality and performance
- **`test_collaborative_model.ps1`**: PowerShell script to test the entire pipeline

## Data Flow
1. Raw data is processed by `process_data.py`
2. Features are built using `build_features.py`
3. The collaborative model is trained with `train_model.py`
4. Recommendations are generated using `predict_model.py`

## Docker Components
The system includes Docker containers for:
- **data-retrieval**: Fetches raw book data
- **data-ingestion**: Processes raw data into usable format
- **model-training**: Trains the collaborative filtering model
- **prediction-api**: Serves recommendations via an API

## Why Collaborative Filtering Only?
We simplified the architecture to focus solely on collaborative filtering because:
1. It provides high-quality recommendations based on user behavior patterns
2. It's more efficient to maintain a single model type
3. Performance testing showed sufficient accuracy with collaborative filtering alone
4. Simpler architecture leads to easier deployment and maintenance

## Next Steps
Potential future enhancements:
1. Improved cold-start handling
2. Advanced collaborative filtering techniques
3. Adding real-time learning capabilities
4. Scaling to larger datasets
