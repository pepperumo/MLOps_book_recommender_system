import streamlit as st
# Set page configuration must come before any other streamlit commands
st.set_page_config(
    page_title="MLOps Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import os
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import sys, platform, datetime, time

# Only show version info after page config
st.write(f"🕒 Rendered at {time.time()} with Streamlit {st.__version__}")

# Function to read and convert mermaid diagrams to HTML
def render_mermaid(mermaid_file_path):
    with open(mermaid_file_path, 'r') as file:
        mermaid_code = file.read()
    
    # Remove the filepath comment if present
    if (mermaid_code.startswith('//')):
        mermaid_code = '\n'.join(mermaid_code.split('\n')[1:])
    
    # Create HTML for the mermaid diagram
    mermaid_html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            fontSize: 14
        }});
    </script>
    """
    return mermaid_html

# Function to read markdown files
def read_markdown_file(markdown_file_path):
    with open(markdown_file_path, 'r') as file:
        markdown_text = file.read()
    return markdown_text

# Root directory
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")  # New assets directory in streamlit folder
API_URL = os.environ.get("API_URL", "http://localhost:8000")


# Title and introduction
st.title("📚 MLOps Book Recommender System")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "Project Overview",
    "System Architecture",
    "Data Pipeline & Model Development",
    "API & UI Deployment",
    "Monitoring Stack",
    "DVC Pipeline",
    "Airflow Pipeline",
    "Future Improvements"
]
selected_page = st.sidebar.radio("Go to", pages)

# Add team information to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
            <div style='text-align: center; color: #666;'>
            
            <div style='margin: 10px 0;'>
            This app is maintained by:<br>
            <a href="https://www.linkedin.com/in/giuseppe-rumore-b2599961" target="_blank">Giuseppe Rumore</a> |
            <a href="https://www.linkedin.com/in/fbarulli" target="_blank">Fabian Barulli</a> |
            <a href="https://www.linkedin.com/in/allaeldene-ilou" target="_blank">Allaeldene Ilou</a>
            </div>
            </a>
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg" 
             width="80" 
             alt="LinkedIn"
             style="margin-top: 10px;">
            </a>
            <br>
            <a href="https://github.com/pepperumo/MLOps_book_recommender_system" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
             width="40"
             alt="GitHub"
             style="margin-top: 10px; border-radius: 50%;">
            </a>
            <div style='text-align: center; color: #666;'>
            <small>Version 1.0.0</small><br>
            <small>©2025 Book Recommender System</small><br>
            </div>
        """, unsafe_allow_html=True)

# Define page functions for each section
def show_project_overview():
    st.markdown("""
    This interactive app provides an overview of the MLOps Book Recommender System project structure and architecture.
    The actual recommendation system is implemented with a React frontend, while this Streamlit app serves purely as 
    documentation to help understand the project.
    """)
    st.header("Project Overview")
      # Display frontend overview image
    st.image(os.path.join(ASSETS_DIR, "Frontend_book.png"), caption="Book Recommender UI", width=800)
    
    # Overview section
    st.markdown("""
    ## About this Project
    
    The MLOps Book Recommender System is a comprehensive machine learning project that demonstrates MLOps best practices through 
    a book recommendation engine. It uses collaborative filtering to provide personalized book recommendations to users.
    """)
    # Two-column layout for features and rationale
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Features")
        st.markdown("""
        * Data versioning with DVC
        * CI/CD with GitHub Actions
        * Containerized components with Docker
        * API service with FastAPI
        * Frontend with React
        * Workflow orchestration with Airflow
        * Monitoring with Prometheus and Grafana
        """)
    with col2:
        st.subheader("Why Collaborative Filtering?")
        st.markdown("""
        1. It provides high-quality recommendations based on user behavior patterns
        2. It's more efficient to maintain a single model type
        3. Performance testing showed sufficient accuracy with collaborative filtering alone
        4. Simpler architecture leads to easier deployment and maintenance
        """)
    
    # Project components visualization
    st.subheader("Main Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### Data Components
        - **Data Retrieval**: Fetches raw book data
        - **Data Processing**: Cleans and formats data
        - **Feature Engineering**: Creates model features
        """)
    with col2:
        st.markdown("""
        #### Model Components
        - **Model Training**: Trains the collaborative model
        - **Model Evaluation**: Measures model performance
        - **Model Serving**: Serves recommendations via API
        """) 
    with col3:
        st.markdown("""
        #### Infrastructure Components
        - **CI/CD Pipeline**: Automates testing and deployment
        - **Monitoring Stack**: Tracks system health and performance
        - **Docker Containers**: Isolates and packages components
        """) 
    
    # Service links
    st.subheader("Service Access")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### API & Frontend
        - **FastAPI**: [http://localhost:8000](http://localhost:8000)
        - **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
        - **React Frontend**: [http://localhost:4000](http://localhost:4000)
        """)
    with col2:
        st.markdown("""
        #### Airflow
        - **Airflow Webserver**: [http://localhost:8080](http://localhost:8080)
        - **Airflow API**: [http://localhost:8080/api](http://localhost:8080/api)
        """)
    with col3:
        st.markdown("""
        #### Monitoring
        - **Prometheus**: [http://localhost:9090](http://localhost:9090)
        - **Grafana**: [http://localhost:3000](http://localhost:3000)
        - **PushGateway**: [http://localhost:9091](http://localhost:9091)
        """)


def show_system_architecture():
    st.header("System Architecture")
    
    st.markdown("""
    The system follows a modular architecture with distinct components that handle specific responsibilities.
    Below is the high-level architecture diagram that shows how these components interact.
    """)
    
    # Display the architecture diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_architecture.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    st.markdown("### System Components")
    
    # Create 3 columns layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data & Model Layers")
        st.markdown("""
        **Data Layer**
        - Ingests raw book data from multiple sources
        - Processes and cleans data for consistency
        - Maintains data versioning with DVC
        - Stores data in structured formats for model use
        
        **Model Layer**
        - Transforms raw data into model features
        - Trains collaborative filtering recommendation models
        - Tracks experiments with metrics and parameters
        - Registers and versions trained models
        """)
    
    with col2:
        st.subheader("Application Layers")
        st.markdown("""
        **API Layer**
        - Serves model predictions via REST endpoints
        - Handles user authentication and request validation
        - Implements caching for performance optimization
        - Provides swagger documentation for API users
        
        **UI Layer**
        - Delivers responsive React-based interface
        - Displays personalized book recommendations
        - Allows users to search and browse book catalog
        - Provides rating functionality to improve future recommendations
        
        **Airflow Layer**
        - Orchestrates data and model pipeline execution
        - Schedules periodic retraining and evaluation
        - Monitors pipeline status and handles failures
        - Manages dependencies between workflow tasks
        """)
    
    with col3:
        st.subheader("Infrastructure Layers")
        st.markdown("""
        **CI/CD Layer**
        - Automates testing on code commits
        - Validates model performance before deployment
        - Builds and publishes Docker images
        - Deploys components to production environment
        
        **Docker Containers**
        - Packages components with dependencies
        - Ensures consistent execution across environments
        - Simplifies deployment and scaling
        - Isolates services for better resource management
        
        **Monitoring Layer**
        - Collects system and application metrics
        - Visualizes performance through Grafana dashboards
        - Alerts on anomalies or performance degradation
        - Tracks model drift and data quality issues
        """)


def show_data_pipeline():
    st.header("Data Pipeline & Model Development")
    
    st.markdown("""
    The data and model pipeline includes the processes for retrieving, processing, 
    and transforming data, as well as training and evaluating the recommendation model.
    """)
    
    # Display the data pipeline diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_data_model_pipeline.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    # Create two columns for Data Flow and Key Files
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Flow")
        st.markdown("""
        1. **Raw Data Retrieval**: 
           - Fetches book metadata and user ratings from Google Books API
           - Collects author information, genres, and publication details
           - Stores raw data in versioned storage with DVC
        
        2. **Data Processing**:
           - Removes duplicate entries and inconsistent records
           - Normalizes text fields and standardizes formats
           - Handles missing values with appropriate strategies
        
        3. **Feature Extraction**:
           - Creates user-item interaction matrices
           - Generates embedding vectors for books and users
           - Builds similarity metrics between items
        
        4. **Model Training**:
           - Trains collaborative filtering recommendation model
           - Optimizes hyperparameters for best performance
           - Saves model checkpoints during training
        
        5. **Model Evaluation**:
           - Calculates precision, recall, and F1 metrics
           - Measures recommendation relevance and diversity
           - Performs A/B testing against baseline models
        
        6. **Model Registration**:
           - Registers production-ready model with metadata
           - Makes model available for serving via API
           - Archives previous model versions for rollback
        """)
    
    with col2:
        st.markdown("### Key Files")
        st.markdown("""
        - **`retrieve_data.py`**:
          - Implements API clients for external data sources
          - Handles rate limiting and pagination
          - Supports incremental data fetching
        
        - **`process_data.py`**:
          - Applies data cleaning rules and transformations
          - Performs deduplication and normalization
          - Creates consistent data schema for modeling
        
        - **`build_features.py`**:
          - Extracts numerical and categorical features
          - Implements feature selection algorithms
          - Creates embedding representations for items
        
        - **`train_model.py`**:
          - Implements collaborative filtering algorithms
          - Supports multiple training strategies
          - Handles distributed training configuration
        
        - **`predict_model.py`**:
          - Generates personalized book recommendations
          - Handles both user-based and item-based predictions
          - Optimizes recommendation serving for low latency
        
        - **`test_model.py`**:
          - Implements unit and integration tests for models
          - Validates model behavior with test datasets
          - Ensures compatibility with API interfaces
        
        - **`model_utils.py`**:
          - Provides helper functions for model operations
          - Implements common transformations and utilities
          - Facilitates model loading and preprocessing
        
        - **`mlflow_utils.py`**:
          - Handles model versioning and tracking with MLflow
          - Logs model parameters, metrics, and artifacts
          - Supports model registry integration
          - **`evaluate_model.py`**:
          - Calculates precision@k and recall@k at k=5, 10, 20
          - Pushes evaluation metrics to Prometheus for monitoring
          - Compares model versions for improvement
        """)
    
    st.markdown("""
    ### Data Pipeline Setup
    
    The data pipeline can be run using Docker Compose:
    
    ```bash
    # Run the data retrieval and processing pipeline
    docker-compose -f docker-compose.data-pipeline.yml up
    
    # Run the model training pipeline
    docker-compose -f docker-compose.train.yml up
    ```
    """)


def show_dvc_pipeline():
    st.header("DVC Pipeline")
    
    st.markdown("""
    ### Data Version Control Pipeline
    
    The Book Recommender System uses [DVC](https://dagshub.com/pepperumo/MLOps_book_recommender_system) (Data Version Control) to manage the ML pipeline,
    ensuring reproducibility and tracking of data and models throughout the development process.
    """)
      # Display the DVC pipeline image
    st.image(os.path.join(ASSETS_DIR, "dvc.png"), caption="DVC Pipeline Graph", width=800)
    
    # DVC Commands
    st.subheader("Running the Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Pipeline Benefits
        - **Reproducibility**: Exact recreation of models and results
        - **Version Control**: Track changes to data and models over time
        - **Automation**: Simple execution of complex ML workflows
        - **Dependency Management**: Automatic handling of stage dependencies
        """)
    
    with col2:
        st.markdown("""
        #### Pipeline Monitoring
        - Track metrics across experiments
        - Compare model versions
        - Visualize pipeline execution with `dvc dag`
        - Detect changes in data and code dependencies
        """)
    
    # With Docker
    st.markdown("""
    ### Running with Docker
    
    The DVC pipeline can also be run using Docker for consistent environments:
    
    ```bash
    # Run the DVC pipeline in Docker
    docker-compose -f docker-compose.dvc.yml up
    ```
    """)


def show_airflow_pipeline():
    st.header("Airflow Pipeline")
    st.markdown("""
    ### Book Recommender System ML Pipeline
    
    This Airflow DAG orchestrates the full ML pipeline for the book recommender system. It automates the 
    end-to-end workflow from data retrieval to model evaluation and API testing.
    """)
    
    # Create a visualization of the DAG workflow
    st.subheader("Pipeline Workflow")
    # Display the Airflow pipeline diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "airflow_chart.mmd"))
    st.components.v1.html(mermaid_html, height=300, scrolling=True)
    
    # Access Info
    st.success("""
    **Access Airflow UI**: [http://localhost:8080](http://localhost:8080)  
    Username: admin | Password: admin
    """)
    
    # DAG details
    st.subheader("DAG Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### Schedule & Settings
        
        - **Schedule**: Weekly
        - **Start Date**: April 16, 2025
        - **Retries**: 1 (with 5-minute delay)
        - **Tags**: mlops, recommender, books
        """)
        
    with col2:
        st.markdown("""
        #### Execution Flow
        
        1. Linear flow from data retrieval to model training
        2. After training, two parallel paths:
           - Model evaluation
           - API testing sequence (start → wait → test → stop)
        """)
    with col3:
        st.markdown("""
        #### Task Categories:
        
        - **Data Tasks** (Green)
            - Data Retrieval
            - Data Processing
        
        - **Feature Tasks** (Purple)
            - Feature Building
        
        - **Model Tasks** (Red)
            - Model Training
            - Model Evaluation
        
        - **API Test Tasks** (Blue)
            - Start API
            - Wait for API
            - Run API Tests
            - Stop API
        """)
    with col4:
        st.markdown("""
        #### Task Details:
        
        - **Data Retrieval**: Fetches raw book data from sources and stores it
        - **Data Processing**: Cleans, transforms, and prepares the data for feature building
        - **Feature Building**: Creates features needed for the recommendation model
        - **Model Training**: Trains the collaborative filtering model using the features
        - **Model Evaluation**: Evaluates model performance with metrics like precision and recall
        - **API Testing Flow**: Starts the API, waits for it to initialize, runs tests, then shuts it down
        """)
    
    st.markdown("""
    ### Airflow Setup
    
    The Airflow environment can be run using Docker Compose:
    
    ```bash
    # Start the Airflow environment
    docker-compose -f docker-compose.airflow.yml up
    
    # Access the Airflow webserver at http://localhost:8080
    # Username: admin | Password: admin
    ```
    """)


def show_api_ui_deployment():
    st.header("API & UI Deployment")
    
    st.markdown("""
    The recommendation system's user-facing components consist of a FastAPI backend 
    and a React frontend. These components are containerized and can be deployed together.
    """)    
    # Display the API & UI deployment diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_api_ui_deployment.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=False)
    
    # Access Info
    st.success("""
    **Access API**: [http://localhost:8000](http://localhost:8000)  
    **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)  
    **React Frontend**: [http://localhost:4000](http://localhost:4000)
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### API Components")
        st.markdown("""
        The API service is built with FastAPI and provides the following endpoints:
        
        - **GET** `/recommend/user/{user_id}`: Get book recommendations for a specific user
        - **GET** `/similar-books/{book_id}`: Get similar books to a given book
        - **GET** `/books`: Get a list of books with their metadata
        - **GET** `/books/{book_id}`: Get details for a specific book
        - **GET** `/users`: Get a list of users
        - **GET** `/users/{user_id}`: Get a specific user's profile
        - **POST** `/ratings`: Submit a new book rating
        - **GET** `/health`: Health check endpoint
        - **GET** `/docs`: API documentation (Swagger UI)
        - **GET** `/redoc`: Alternative API documentation (ReDoc)
        """)
    
    with col2:
        st.markdown("### Frontend Components")
        st.markdown("""
        The React frontend provides an interactive user interface with:
        
        - Dashboard with popular books
        - User recommendation page
        - Similar books search
        - Book browsing and filtering
        """)
    
    st.markdown("""
    ### API & UI Deployment Setup
    
    The API and frontend components can be deployed together using Docker Compose:
    
    ```bash
    # Deploy the API and frontend components
    docker-compose -f docker-compose.deploy-local.yml up
    
    # Access the API at http://localhost:8000
    # Access the API docs at http://localhost:8000/docs
    # Access the frontend at http://localhost:4000
    ```
    """)


def show_monitoring_stack():
    st.header("Monitoring & Observability")
    
    st.markdown("""
    The monitoring stack tracks system health, performance, and model metrics 
    to ensure the recommendation system operates optimally.
    """)    
    # Display the monitoring diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_monitoring.mmd"))
    st.components.v1.html(mermaid_html, height=700, scrolling=False)
      # Display Grafana dashboard screenshot
    st.subheader("Grafana Dashboard")
    st.image(os.path.join(ASSETS_DIR, "Grafana_monitoring.png"), caption="Book Recommender System Metrics Dashboard in Grafana", width=800)
    
    # Access Info
    st.success("""
    **Access Prometheus**: [http://localhost:9090](http://localhost:9090) (No authentication required)  
    **Access Grafana**: [http://localhost:3000](http://localhost:3000) (Username: admin | Password: admin)  
    **Access PushGateway**: [http://localhost:9091](http://localhost:9091)
    """)
    
    st.markdown("""
    #### Prometheus
    Collects and stores metrics from various system components.
    
    #### Grafana
    Visualizes metrics with customizable dashboards.
    
    #### Pushgateway
    Allows batch jobs like model training to push metrics.
    
    ### Key Metrics
    
    - **Model Performance Metrics**:
      - Precision@k (k=5, 10, 20)
      - Recall@k (k=5, 10, 20)
      - Model load time
    
    - **API Metrics**:
      - Recommendation count
      - API health check status
      - Request latency
    
    ### Monitoring Setup
    
    The monitoring stack can be run standalone or alongside the main application:
    
    ```bash
    # Standalone monitoring
    docker-compose -f docker-compose.monitoring.yml up
    
    # With deployment
    docker-compose -f docker-compose.deploy-local.yml -f docker-compose.monitoring.yml up
    ```
    """)


def show_future_improvements():
    st.header("Future Improvements")
    
    st.markdown("""
    While the current MLOps Book Recommender System provides a solid foundation, several enhancements 
    could further improve the system's functionality, performance, and user experience.
    """)
    
    st.markdown("""
    ### Technical Improvements
    
    - **Cloud Deployment**: Migrate from local Docker setup to a production cloud platform (AWS, GCP, Azure) for better scalability and reliability
    
    - **TypeScript Migration**: Convert React frontend from JavaScript to TypeScript for improved type safety and maintainability
    
    ### ML & Recommendation Enhancements
    
    - **Content-Based Filtering**: Analyze book content (descriptions, genres, authors) to recommend similar items, addressing the cold-start problem
    
    - **Hybrid Filtering**: Combine collaborative and content-based approaches for more robust recommendations across all user types
    
    - **Contextual Recommendations**: Incorporate user preferences, reading patterns, and seasonal trends for more personalized recommendations
    """)

# Dictionary to map page names to their respective functions
page_functions = {
    "Project Overview": show_project_overview,
    "System Architecture": show_system_architecture,
    "Data Pipeline & Model Development": show_data_pipeline,
    "API & UI Deployment": show_api_ui_deployment,
    "Monitoring Stack": show_monitoring_stack,
    "DVC Pipeline": show_dvc_pipeline,
    "Airflow Pipeline": show_airflow_pipeline,
    "Future Improvements": show_future_improvements
}

# Display the selected page
page_functions[selected_page]()

