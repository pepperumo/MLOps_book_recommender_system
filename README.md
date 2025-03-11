Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

## Steps to follow 

Convention : All python scripts must be run from the root specifying the relative file path.

### 1- Create a virtual environment using Virtualenv.

    `python -m venv my_env`

###   Activate it 

    `./my_env/Scripts/activate`

###   Install the packages from requirements.txt  (You can ignore the warning with "setup.py")

    `pip install -r .\requirements.txt`

### 2- Execute import_raw_data.py to import the 4 datasets (say yes when it asks you to create a new folder)

    `python .\src\data\import_raw_data.py` 

### 3- Execute make_dataset.py initializing `./data/raw` as input file path and `./data/processed` as output file path.

    `python .\src\data\make_dataset.py`

### 4- Execute build_features.py to preprocess the data (this can take a while)

    `python .\src\features\build_features.py`

### 5- Execute train_model.py to train the model

    `python .\src\models\train_model.py`

### 5- Finally, execute predict_model.py file to make the predictions (by default you will be printed predictions for the first 5 users of the dataset). 

    `python .\src\models\predict_model.py`

### Note that we have 10 recommandations per user

## Using Data Version Control (DVC)

This project uses DVC to manage large model and data files separately from Git.

### Setting up DVC

1. **Make sure your virtual environment is activated**
   ```bash
   ./my_env/Scripts/activate
   ```

2. **Pull model and data files** (after cloning the repository)
   ```bash
   # Make sure your virtual environment is activated
   dvc pull
   ```

3. **After training new models** (if you modify any models)
   ```bash
   # Make sure your virtual environment is activated
   dvc add models
   git add models.dvc
   git commit -m "Update model files"
   dvc push
   ```

4. **After adding or modifying data** (if you add new datasets)
   ```bash
   # Make sure your virtual environment is activated
   dvc add data
   git add data.dvc
   git commit -m "Update data files"
   dvc push
   ```

For more detailed DVC instructions, please see the [API README](./src/api/README.md#using-dvc-for-model-and-data-files).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
