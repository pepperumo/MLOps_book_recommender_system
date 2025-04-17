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

    # Task 5: Model Evaluation
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='cd /opt/airflow && python -m src.models.evaluate_model --model-path models/collaborative.pkl',
    )

    # Task 6: Push metrics to Prometheus
    def push_metrics_to_prometheus():
        """Push model evaluation metrics to Prometheus Pushgateway"""
        try:
            # Read evaluation results
            results_path = '/opt/airflow/data/results/evaluation_results.csv'
            if not os.path.exists(results_path):
                print(f"Error: Results file {results_path} not found")
                return

            # Parse the CSV to get actual metrics
            results_df = pd.read_csv(results_path, index_col=0)
            
            # Get the metrics for the collaborative model
            if 'collaborative' in results_df.index:
                model_metrics = results_df.loc['collaborative'].to_dict()
            else:
                model_metrics = results_df.iloc[0].to_dict()
            
            # Push metrics to Prometheus Pushgateway
            pushgateway_url = 'http://pushgateway:9091'
            job_name = 'book_recommender_model'
            
            # Format metrics for Prometheus
            metrics_data = ""
            for metric_name, metric_value in model_metrics.items():
                # Convert metrics like 'precision@10' to 'book_recommender_precision_at_10'
                formatted_name = metric_name.replace('@', '_at_')
                metrics_data += f"book_recommender_{formatted_name} {metric_value}\n"
            
            # Push to Pushgateway
            response = requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data=metrics_data
            )
            
            print(f"Pushed metrics to Prometheus Pushgateway: {response.status_code}")
            print(f"Metrics sent: {model_metrics}")
        except Exception as e:
            print(f"Error pushing metrics to Prometheus: {str(e)}")
            import traceback
            print(traceback.format_exc())

    push_metrics = PythonOperator(
        task_id='push_metrics',
        python_callable=push_metrics_to_prometheus,
    )

    # Define task dependencies
    retrieve_data >> process_data >> build_features >> train_model >> evaluate_model >> push_metrics