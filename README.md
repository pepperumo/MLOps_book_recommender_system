Project Name
==============================

This project is a Book Recommender System using Collaborative Filtering. It's designed with MLOps principles to provide a streamlined, production-ready recommendation system.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── process_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    │   │   └── train_model_collaborative.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

## Architecture Overview

This recommender system uses collaborative filtering to provide personalized book recommendations. Collaborative filtering works by analyzing user-item interactions (in this case, book ratings) to identify patterns and similarities between users and/or items.

Key components:
- **Data Processing**: Raw book and rating data is cleaned and processed
- **Feature Engineering**: User-item matrices and book similarity calculations
- **Collaborative Filtering**: Implementation using matrix factorization techniques
- **Evaluation**: Performance metrics for recommendation quality
- **Prediction API**: Endpoints for generating recommendations

## Steps to follow 

Convention : All python scripts must be run from the root specifying the relative file path.

### 1- Create a virtual environment using Virtualenv.

    `python -m venv venv`

###   Activate it 

    `.\venv\Scripts\activate`

###   Install the packages from requirements.txt  (You can ignore the warning with "setup.py")

    `pip install -r .\requirements.txt`

### 2- Execute import_raw_data.py to import the datasets (say yes when it asks you to create a new folder)

    `python .\src\data\import_raw_data.py` 

### 3- Execute process_data.py initializing `./data/raw` as input file path and `./data/processed` as output file path.

    `python .\src\data\process_data.py`

### 4- Execute build_features.py to preprocess the data (this can take a while)

    `python .\src\features\build_features.py`

### 5- Execute train_model.py to train the collaborative filtering model

    `python .\src\models\train_model.py`

### 6- Finally, execute predict_model.py file to make the predictions (by default you will be printed predictions for the first 5 users of the dataset). 

    `python .\src\models\predict_model.py`

### 7- Run the API to access the recommendation system

    `uvicorn src.api.api:app --reload`

    This will start a FastAPI server on http://127.0.0.1:8000 with the following endpoints:
    - `/docs`: Interactive API documentation
    - `/recommend/user/{user_id}`: Get personalized book recommendations for a user
    - `/similar-books/{book_id}`: Get similar books to a specific book

### Running the Full Pipeline

To run the entire data processing, model training, and testing pipeline in sequence:

    `python run_full_pipeline.py`

This script executes all components sequentially:
1. Data retrieval
2. Data processing
3. Feature building
4. Model training
5. Model evaluation
6. Model testing
7. Sample predictions
8. API tests

### Test the complete pipeline

To test the entire pipeline with a single command, you can use the provided PowerShell script:

    `.\test_collaborative_model.ps1`

## Data Management

This project manages book and user rating data through a structured pipeline:

1. **Data Retrieval**: Raw data is fetched from the external API
2. **Data Processing**: Raw data is cleaned and processed using `process_data.py`
3. **Feature Building**: Processed data is converted into feature matrices for modeling
4. **Model Training**: Features are used to train the collaborative filtering model

All data is organized according to the project structure, with separate directories for raw, processed, and feature data.

## Docker Deployment

This project supports deployment using Docker and Docker Compose for containerized execution.

### Build and Run with Docker Compose

1. **Build the Docker images**
   ```bash
   docker-compose build
   ```

2. **Start the system**
   ```bash
   docker-compose up
   ```

This will start four services:
- `data-retrieval`: Retrieves book data from the API
- `data-ingestion`: Processes the data
- `model-training`: Trains the recommendation models
- `prediction-api`: Serves the recommendation API on port 8000

You can access the API at http://localhost:8000 once it's running.

3. **Stop the system**
   ```bash
   docker-compose down
   ```

### Docker Compatibility

The system uses a custom unpickler to ensure model compatibility between different environments. This resolves class definition differences when loading models in Docker containers.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
