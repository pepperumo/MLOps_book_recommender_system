import streamlit as st
import pandas as pd
import os
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="MLOps Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to read and convert mermaid diagrams to HTML
def render_mermaid(mermaid_file_path):
    with open(mermaid_file_path, 'r') as file:
        mermaid_code = file.read()
    
    # Remove the filepath comment if present
    if mermaid_code.startswith('//'):
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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAPHS_DIR = os.path.join(ROOT_DIR, "graphs")  # Original graphs directory
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")  # New assets directory in streamlit folder
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Title and introduction
st.title("📚 MLOps Book Recommender System")
st.markdown("""
This interactive app provides an overview of the MLOps Book Recommender System project structure and architecture.
The actual recommendation system is implemented with a React frontend, while this Streamlit app serves purely as 
documentation to help understand the project.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "Project Overview",
    "System Architecture",
    "Data Pipeline",
    "API & UI Deployment",
    "Monitoring Stack",
    "Docker Components",
    "Project Structure"
]
selected_page = st.sidebar.radio("Go to", pages)

# Project Overview Page
if selected_page == "Project Overview":
    st.header("Project Overview")
    
    st.markdown("""
    ## About this Project
    
    The MLOps Book Recommender System is a comprehensive machine learning project that demonstrates MLOps best practices through 
    a book recommendation engine. It uses collaborative filtering to provide personalized book recommendations to users.
    
    ### Key Features
    
    * Data versioning with DVC
    * CI/CD with GitHub Actions
    * Containerized components with Docker
    * API service with FastAPI
    * Frontend with React
    * Workflow orchestration with Airflow
    * Monitoring with Prometheus and Grafana
    
    ### Why Collaborative Filtering?
    
    The system focuses solely on collaborative filtering because:
    1. It provides high-quality recommendations based on user behavior patterns
    2. It's more efficient to maintain a single model type
    3. Performance testing showed sufficient accuracy with collaborative filtering alone
    4. Simpler architecture leads to easier deployment and maintenance
    """)
    
    # Project components visualization
    st.subheader("Main Components")
    
    col1, col2 = st.columns(2)
    
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
    
    st.markdown("""
    #### Infrastructure Components
    - **CI/CD Pipeline**: Automates testing and deployment
    - **Monitoring Stack**: Tracks system health and performance
    - **Docker Containers**: Isolates and packages components
    """)

# System Architecture Page
elif selected_page == "System Architecture":
    st.header("System Architecture")
    
    st.markdown("""
    The system follows a modular architecture with distinct components that handle specific responsibilities.
    Below is the high-level architecture diagram that shows how these components interact.
    """)
    
    # Display the architecture diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_architecture.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    st.markdown("""
    ### Component Description
    
    #### Data Layer
    Handles data acquisition, processing, versioning, and storage.
    
    #### Model Layer
    Manages feature engineering, model training, experiment tracking, and model registry.
    
    #### API Layer
    Provides endpoints for recommendations and book data access.
    
    #### UI Layer
    Delivers an interactive user interface for end users.
    
    #### CI/CD Layer
    Automates testing and deployment workflows.
    
    #### Airflow Layer
    Orchestrates data and model pipelines.
    
    #### Docker Containers
    Packages components for consistent deployment.
    
    #### Monitoring Layer
    Tracks system health and performance metrics.
    """)

# Data Pipeline Page
elif selected_page == "Data Pipeline":
    st.header("Data Pipeline & Model Development")
    
    st.markdown("""
    The data and model pipeline includes the processes for retrieving, processing, 
    and transforming data, as well as training and evaluating the recommendation model.
    """)
    
    # Display the data pipeline diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_data_model_pipeline.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    st.markdown("""
    ### Data Flow
    1. Raw data is retrieved from external sources
    2. Data is processed and cleaned
    3. Features are extracted for model training
    4. The model is trained using collaborative filtering
    5. Model performance is evaluated
    6. The trained model is registered for serving
    
    ### Key Files
    - `process_data.py`: Cleans and processes raw data
    - `build_features.py`: Creates features for the model
    - `train_model.py`: Trains the collaborative filtering model
    - `evaluate_model.py`: Evaluates model performance
    """)
    
    # Data processing steps visualization
    st.subheader("Data Processing Steps")
    
    steps = [
        "Raw Data Collection",
        "Data Cleaning",
        "Feature Extraction",
        "Model Training",
        "Model Evaluation",
        "Model Serving"
    ]
    
    step_descriptions = [
        "Collect book metadata and user ratings",
        "Remove duplicates, handle missing values, normalize formats",
        "Create user-item matrices and similarity features",
        "Train collaborative filtering model",
        "Measure recommendation quality with metrics",
        "Deploy model for API access"
    ]
    
    # Create a DataFrame for the steps
    df = pd.DataFrame({
        "Step": steps,
        "Description": step_descriptions,
        "Step Number": range(1, len(steps) + 1)
    })
    
    # Create a chart
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('Step Number:O', axis=alt.Axis(title=None)),
        y=alt.Y('Step:N', axis=alt.Axis(title=None)),
        color=alt.Color('Step:N', legend=None),
        tooltip=['Step', 'Description']
    ).properties(
        width=700,
        height=300
    )
    
    lines = alt.Chart(df).mark_line(color='gray').encode(
        x='Step Number:O',
        y='Step:N'
    )
    
    st.altair_chart(chart + lines, use_container_width=True)

# API & UI Deployment Page
elif selected_page == "API & UI Deployment":
    st.header("API & UI Deployment")
    
    st.markdown("""
    The recommendation system's user-facing components consist of a FastAPI backend 
    and a React frontend. These components are containerized and can be deployed together.
    """)
    
    # Display the API & UI deployment diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_api_ui_deployment.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    st.markdown("""
    ### API Components
    
    The API service is built with FastAPI and provides the following endpoints:
    
    - `/recommend/user/{user_id}`: Get book recommendations for a user
    - `/similar-books/{book_id}`: Get similar books to a given book
    - `/books`: Get a list of books with their metadata
    - `/users`: Get a list of users
    - Health check and documentation endpoints
    
    ### Frontend Components
    
    The React frontend provides an interactive user interface with:
    
    - Dashboard with popular books
    - User recommendation page
    - Similar books search
    - Book browsing and filtering
    
    ### Deployment Options
    
    The system can be deployed in multiple ways:
    
    1. **Full Pipeline**: Includes data processing, model training, API, and frontend
    2. **Deployment Only**: Just the API and frontend components
    3. **With Monitoring**: All components plus monitoring stack
    """)
    
    # Show deployment methods comparison
    st.subheader("Deployment Methods")
    
    deployment_data = {
        "Method": ["Full Pipeline", "API & Frontend Only", "With Monitoring"],
        "Use Case": ["Development & Testing", "Production", "Production with Observability"],
        "Components": ["All", "API & Frontend", "All + Monitoring"],
        "Docker Compose File": ["docker-compose.train.yml", "docker-compose.deploy-local.yml", "docker-compose.monitoring.yml"]
    }
    
    df_deployment = pd.DataFrame(deployment_data)
    st.table(df_deployment)

# Monitoring Stack Page
elif selected_page == "Monitoring Stack":
    st.header("Monitoring & Observability")
    
    st.markdown("""
    The monitoring stack tracks system health, performance, and model metrics 
    to ensure the recommendation system operates optimally.
    """)
    
    # Display the monitoring diagram using mermaid from assets directory
    mermaid_html = render_mermaid(os.path.join(ASSETS_DIR, "mlops_monitoring.mmd"))
    st.components.v1.html(mermaid_html, height=600, scrolling=True)
    
    st.markdown("""
    ### Monitoring Components
    
    #### Prometheus
    Collects and stores metrics from various system components.
    
    #### Grafana
    Visualizes metrics with customizable dashboards.
    
    #### Pushgateway
    Allows batch jobs like model training to push metrics.
    
    ### Key Metrics
    
    - **System Metrics**: CPU, memory, and network usage
    - **API Metrics**: Request count, latency, and error rates
    - **Model Metrics**: Recommendation quality, training time, and prediction latency
    
    ### Monitoring Setup
    
    The monitoring stack can be run standalone or alongside the main application:
    
    ```bash
    # Standalone monitoring
    docker-compose -f docker-compose.monitoring.yml up
    
    # With deployment
    docker-compose -f docker-compose.deploy.yml -f docker-compose.monitoring.yml up
    ```
    """)
    
    # Create a metrics visualization
    st.subheader("Sample Metrics Dashboard")
    
    # Generate fake data for visualization
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    api_requests = np.random.randint(100, 500, size=30) + np.arange(30) * 5
    response_time = 50 + 10 * np.sin(np.arange(30)/5) + np.random.randint(0, 20, size=30)
    error_rate = np.random.rand(30) * 2
    
    metrics_df = pd.DataFrame({
        'Date': dates,
        'API Requests': api_requests,
        'Avg Response Time (ms)': response_time,
        'Error Rate (%)': error_rate
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(metrics_df.set_index('Date')['API Requests'])
        st.caption("API Requests Over Time")
    
    with col2:
        st.line_chart(metrics_df.set_index('Date')['Avg Response Time (ms)'])
        st.caption("Average Response Time (ms)")

# Docker Components Page
elif selected_page == "Docker Components":
    st.header("Docker Components")
    
    st.markdown("""
    The system uses Docker and Docker Compose to containerize and orchestrate 
    the various components, making deployment consistent and reproducible.
    """)
    
    st.subheader("Container Architecture")
    
    # Create visualization for Docker containers
    docker_components = {
        "Container": [
            "data-retrieval",
            "data-ingestion", 
            "model-training", 
            "prediction-api", 
            "frontend", 
            "prometheus", 
            "grafana", 
            "pushgateway"
        ],
        "Purpose": [
            "Fetches book data from external sources",
            "Processes raw data into usable format",
            "Trains the recommendation models",
            "Serves recommendations via API",
            "Provides user interface",
            "Collects and stores metrics",
            "Visualizes monitoring data",
            "Collects metrics from batch jobs"
        ],
        "Category": [
            "Data", "Data", "Model", "API", "UI", "Monitoring", "Monitoring", "Monitoring"
        ]
    }
    
    docker_df = pd.DataFrame(docker_components)
    
    # Color mapping
    color_scale = alt.Scale(
        domain=["Data", "Model", "API", "UI", "Monitoring"],
        range=["#2a9d2a", "#e6550d", "#6a51a3", "#3182bd", "#d62728"]
    )
    
    # Create chart
    docker_chart = alt.Chart(docker_df).mark_bar().encode(
        y=alt.Y('Container:N', sort=None),
        x=alt.X('Purpose:N'),
        color=alt.Color('Category:N', scale=color_scale),
        tooltip=['Container', 'Purpose', 'Category']
    ).properties(
        width=700,
        height=400
    )
    
    st.altair_chart(docker_chart, use_container_width=True)
    
    st.markdown("""
    ### Docker Compose Files
    
    The project includes several Docker Compose files for different purposes:
    
    - **docker-compose.train.yml**: Runs the full training pipeline
    - **docker-compose.deploy-local.yml**: Deploys the API and frontend locally
    - **docker-compose.monitoring.yml**: Sets up the monitoring stack
    - **docker-compose.airflow.yml**: Runs Airflow for workflow orchestration
    
    ### Example: Starting the Deployment
    
    ```bash
    # Start the API and frontend
    docker-compose -f docker-compose.deploy-local.yml up
    
    # Access the frontend at http://localhost:4000
    # Access the API at http://localhost:8000
    ```
    """)

# Project Structure Page
elif selected_page == "Project Structure":
    st.header("Project Structure")
    
    st.markdown("""
    The project follows a well-organized structure based on the 
    [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) template,
    with additional directories for MLOps components.
    """)
    
    st.code("""
    MLOps_book_recommender_system/
    ├── LICENSE
    ├── README.md
    ├── data/
    │   ├── external/      # Data from third party sources
    │   ├── interim/       # Intermediate data that has been transformed
    │   ├── processed/     # Final, canonical data sets for modeling
    │   └── raw/           # Original, immutable data
    │
    ├── docs/
    │   └── architecture.md
    │
    ├── models/            # Trained and serialized models
    │
    ├── notebooks/         # Jupyter notebooks for exploration
    │
    ├── src/
    │   ├── data/          # Scripts to download or generate data
    │   ├── features/      # Scripts for feature engineering
    │   ├── models/        # Scripts for training and prediction
    │   ├── fastAPI/       # API implementation
    │   └── visualization/ # Scripts for visualizations
    │
    ├── frontend/          # React frontend application
    │
    ├── flask/             # Alternative Flask implementation
    │   ├── backend/
    │   └── frontend/
    │
    ├── docker/            # Docker configuration files
    │
    ├── airflow/           # Airflow DAGs and plugins
    │   ├── dags/
    │   └── plugins/
    │
    ├── graphs/            # Architecture diagrams
    │
    ├── streamlit/         # Documentation app (this app)
    │
    └── docker-compose.*.yml  # Docker Compose files for different configurations
    """, language="bash")
    
    st.subheader("Key Files")
    
    files_list = [
        ("src/data/process_data.py", "Processes raw data into clean format"),
        ("src/features/build_features.py", "Creates features for modeling"),
        ("src/models/train_model.py", "Trains the collaborative filtering model"),
        ("src/models/predict_model.py", "Generates recommendations"),
        ("src/fastAPI/api.py", "Implements the recommendation API"),
        ("frontend/src/App.js", "Main React application component"),
        ("docker-compose.deploy-local.yml", "Deployment configuration"),
        ("dvc.yaml", "Data version control pipeline definition")
    ]
    
    files_df = pd.DataFrame(files_list, columns=["File", "Purpose"])
    st.table(files_df)

# Footer
st.markdown("---")
st.markdown("""
**MLOps Book Recommender System Documentation App**  
Created with Streamlit | [GitHub Repository](https://github.com/username/MLOps_book_recommender_system)
""")