"""
Book Recommender System ML Pipeline DAG

This DAG orchestrates the full ML pipeline for the book recommender system, including:
- Data retrieval
- Data processing
- Feature building
- Model training
- Model evaluation
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import requests
import os
import json
import pandas as pd

# Define default arguments for the DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'book_recommender_pipeline',
    default_args=default_args,
    description='Book recommender system ML pipeline',
    schedule_interval='@weekly',  # Weekly execution
    start_date=datetime(2025, 4, 16),
    catchup=False,
    tags=['mlops', 'recommender', 'books'],
) as dag:

    # Task 1: Data Retrieval
    retrieve_data = BashOperator(
        task_id='retrieve_data',
        bash_command='cd /opt/airflow && python -m src.data.retrieve_raw_data',
    )

    # Task 2: Data Processing
    process_data = BashOperator(
        task_id='process_data',
        bash_command='cd /opt/airflow && python -m src.data.process_data',
    )

    # Task 3: Feature Building
    build_features = BashOperator(
        task_id='build_features',
        bash_command='cd /opt/airflow && python -m src.features.build_features',
    )

    # Task 4: Model Training
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && python -m src.models.train_model --config config/model_params.yaml',
    )

    # ----- API Test Tasks (run in parallel after training) -----
    start_api = BashOperator(
        task_id='start_api',
        bash_command="""
        cd /opt/airflow
        # Launch Uvicorn in background
        nohup python -m uvicorn src.fastAPI.api:app --host 0.0.0.0 --port 7860 \
          > /tmp/uvicorn.log 2>&1 &
        sleep 3  # allow server to initialize
        echo "Uvicorn logs:\n" && head -n 20 /tmp/uvicorn.log
        """,
    )
    wait_api = BashOperator(
        task_id='wait_for_api',
        bash_command="""
        until curl -sSf http://127.0.0.1:7860/health; do sleep 2; done
        """,
    )
    run_api_tests = BashOperator(
        task_id='run_api_tests',
        bash_command="""
        export API_URL=http://127.0.0.1:7860
        export USE_API_PREFIX=true
        echo "Running API pytest suite against $API_URL (using prefix)"
        python -m pytest /opt/airflow/src/fastAPI/test_api_pytest.py -vv --maxfail=1
        """,
    )
    stop_api = BashOperator(
        task_id='stop_api',
        bash_command="pkill -f 'uvicorn src.fastAPI.api' || true",
    )
    # -----------------------------------------------------------

    # Task 5: Model Evaluation
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='cd /opt/airflow && python -m src.models.evaluate_model --model-path models/collaborative.pkl',
    )   

    
    # Define task dependencies
    retrieve_data >> process_data >> build_features >> train_model
    # after training, kick off evaluation and API tests in parallel
    train_model >> evaluate_model
    # API testing flow with metrics pushed BEFORE stopping the API
    train_model >> start_api >> wait_api >> run_api_tests >> stop_api