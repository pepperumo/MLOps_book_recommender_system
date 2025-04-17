from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'book_recommender_api_tests',
    default_args=default_args,
    description='Start API and run pytest to validate endpoints',
    schedule_interval=None,
    start_date=datetime(2025, 4, 17),
    catchup=False,
) as dag:

    # Wait for pipeline DAG's feature build to complete
    wait_for_pipeline = ExternalTaskSensor(
        task_id='wait_for_pipeline_build',
        external_dag_id='book_recommender_pipeline',
        external_task_id='build_features',
        allowed_states=['success'],
        failed_states=['failed'],
        mode='reschedule',
        poke_interval=30,
        timeout=600
    )

    # Wait for pipeline to produce processed data
    wait_for_data = FileSensor(
        task_id='wait_for_data',
        fs_conn_id='fs_default',
        filepath='/opt/airflow/data/processed/merged_train.csv',
        poke_interval=10,
        timeout=300
    )

    # Wait for book_id_mapping file to be available
    wait_for_mapping = FileSensor(
        task_id='wait_for_mapping',
        fs_conn_id='fs_default',
        filepath='/opt/airflow/data/processed/book_id_mapping.csv',
        poke_interval=10,
        timeout=300
    )

    start_api = BashOperator(
        task_id='start_api',
        bash_command=(
            "cd /opt/airflow && \
             nohup python -m uvicorn src.fastAPI.api:app --host 0.0.0.0 --port 7860 \
             > /tmp/uvicorn.log 2>&1 &"
        ),
    )

    wait_api = BashOperator(
        task_id='wait_for_api',
        bash_command="""
        until curl -sSf http://127.0.0.1:7860/health; do
          sleep 2
        done
        """,
    )

    run_tests = BashOperator(
        task_id='run_api_tests',
        bash_command='pytest /opt/airflow/src/fastAPI/test_api_pytest.py --disable-warnings -q',
    )

    stop_api = BashOperator(
        task_id='stop_api',
        bash_command=(
            "pkill -f 'uvicorn src.fastAPI.api' || true"
        ),
    )

    # Ensure data available, then API startup, health check, tests, then shutdown
    wait_for_pipeline >> wait_for_data >> wait_for_mapping >> start_api >> wait_api >> run_tests >> stop_api