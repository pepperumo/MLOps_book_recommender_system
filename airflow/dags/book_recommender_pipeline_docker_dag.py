"""
Book Recommender System ML Pipeline DAG (Docker Version)

This DAG orchestrates the full ML pipeline for the book recommender system using Docker containers.
- Data retrieval
- Data processing
- Feature building
- Model training
- Model evaluation

This version uses DockerOperator to run tasks in separate containers.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
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
    'book_recommender_pipeline_docker',
    default_args=default_args,
    description='Book recommender system ML pipeline using Docker containers',
    schedule_interval='@weekly',  # Weekly execution
    start_date=datetime(2025, 4, 16),
    catchup=False,
    tags=['mlops', 'recommender', 'books', 'docker'],
) as dag:

    # Get absolute paths for volume mounts
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    config_dir = os.path.join(project_dir, 'config')

    # Task 1: Data Retrieval
    retrieve_data = DockerOperator(
        task_id='retrieve_data',
        image='book-recommender/data-retrieval:latest',
        container_name='airflow_data_retrieval',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops_book_recommender_system_app_network',
        mount_tmp_dir=False,
        mounts=[
            {
                'source': data_dir,
                'target': '/app/data',
                'type': 'bind'
            }
        ],
        command='python -m src.data.retrieve_raw_data',
    )

    # Task 2: Data Processing
    process_data = DockerOperator(
        task_id='process_data',
        image='book-recommender/data-ingestion:latest',
        container_name='airflow_data_processing',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops_book_recommender_system_app_network',
        mount_tmp_dir=False,
        mounts=[
            {
                'source': data_dir,
                'target': '/app/data',
                'type': 'bind'
            }
        ],
        command='python -m src.data.process_data',
    )

    # Task 3: Feature Building
    build_features = DockerOperator(
        task_id='build_features',
        image='book-recommender/data-ingestion:latest',  # Reuse the data-ingestion image
        container_name='airflow_feature_building',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops_book_recommender_system_app_network',
        mount_tmp_dir=False,
        mounts=[
            {
                'source': data_dir,
                'target': '/app/data',
                'type': 'bind'
            }
        ],
        command='python -m src.features.build_features',
    )

    # Task 4: Model Training
    train_model = DockerOperator(
        task_id='train_model',
        image='book-recommender/model-training:latest',
        container_name='airflow_model_training',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops_book_recommender_system_app_network',
        mount_tmp_dir=False,
        mounts=[
            {
                'source': data_dir,
                'target': '/app/data',
                'type': 'bind'
            },
            {
                'source': models_dir,
                'target': '/app/models',
                'type': 'bind'
            },
            {
                'source': config_dir,
                'target': '/app/config',
                'type': 'bind'
            }
        ],
        command='python -m src.models.train_model --config config/model_params.yaml',
    )

    # Task 5: Model Evaluation
    evaluate_model = DockerOperator(
        task_id='evaluate_model',
        image='book-recommender/model-training:latest',  # Reuse the model-training image
        container_name='airflow_model_evaluation',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops_book_recommender_system_app_network',
        mount_tmp_dir=False,
        mounts=[
            {
                'source': data_dir,
                'target': '/app/data',
                'type': 'bind'
            },
            {
                'source': models_dir,
                'target': '/app/models',
                'type': 'bind'
            }
        ],
        environment={
            'PROMETHEUS_PUSHGATEWAY': 'pushgateway:9091'
        },
        command='python -m src.models.evaluate_model --model-path models/collaborative.pkl',
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