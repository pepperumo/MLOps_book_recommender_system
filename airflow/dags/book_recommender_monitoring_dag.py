"""
Book Recommender System Monitoring DAG

This DAG handles monitoring tasks for the book recommender system:
1. API health checks
2. System resource monitoring
3. Model performance tracking
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
import pandas as pd
import psutil
import os
import json
import sys

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
    'book_recommender_monitoring',
    default_args=default_args,
    description='Monitoring for book recommender system',
    schedule_interval='*/30 * * * *',  # Run every 30 minutes
    start_date=datetime(2025, 4, 16),
    catchup=False,
    tags=['mlops', 'monitoring', 'recommender'],
) as dag:

    # Task 1: Check API Health
    def check_api_health():
        """Check if the API is healthy and responding"""
        try:
            # API endpoint URL - adjust based on your actual service name
            api_url = "http://api-frontend-local:7860/health"
            
            # Make the request
            response = requests.get(api_url, timeout=5)
            status_code = response.status_code
            
            # Push metrics to Prometheus
            pushgateway_url = 'http://pushgateway:9091'
            job_name = 'api_health_check'
            
            # 1 for healthy, 0 for unhealthy
            health_status = 1 if status_code == 200 else 0
            
            metrics_data = f"""
            book_recommender_api_health {health_status}
            book_recommender_api_response_code {status_code}
            """
            
            requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data=metrics_data
            )
            
            print(f"API Health Check: Status Code {status_code}")
            return health_status == 1
        except Exception as e:
            print(f"API Health Check Error: {str(e)}")
            
            # Push error metric
            try:
                pushgateway_url = 'http://pushgateway:9091'
                job_name = 'api_health_check'
                requests.post(
                    f"{pushgateway_url}/metrics/job/{job_name}",
                    data="book_recommender_api_health 0\n"
                )
            except:
                pass
            
            return False
    
    api_health_check = PythonOperator(
        task_id='api_health_check',
        python_callable=check_api_health,
    )
    
    # Task 2: Monitor System Resources
    def monitor_system_resources():
        """Monitor system resources and push metrics to Prometheus"""
        try:
            # Get CPU, memory, and disk usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Push metrics to Prometheus
            pushgateway_url = 'http://pushgateway:9091'
            job_name = 'system_resources'
            
            metrics_data = f"""
            book_recommender_cpu_percent {cpu_percent}
            book_recommender_memory_percent {memory.percent}
            book_recommender_disk_percent {disk.percent}
            """
            
            response = requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data=metrics_data
            )
            
            print(f"System Resources: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%")
        except Exception as e:
            print(f"System Resources Error: {str(e)}")
    
    system_resources = PythonOperator(
        task_id='monitor_system_resources',
        python_callable=monitor_system_resources,
    )
    
    # Task 3: Track Model Performance
    def track_model_performance():
        """Track model performance metrics over time"""
        try:
            # Since we no longer save evaluation results, we'll skip this step
            print("Skipping model performance tracking as we no longer save evaluation results")
            return
            
            # Old code for reference:
            # results_path = '/opt/airflow/data/results/evaluation_results.csv'
            # 
            # # If results file doesn't exist, exit
            # if not os.path.exists(results_path):
            #     print(f"Error: Results file {results_path} not found")
            #     return
                
            # Read the model evaluation metrics
            results_df = pd.read_csv(results_path, index_col=0)
            
            # Get the metrics for the collaborative model
            if 'collaborative' in results_df.index:
                model_metrics = results_df.loc['collaborative'].to_dict()
            else:
                model_metrics = results_df.iloc[0].to_dict()
            
            # Push to Prometheus with a monitoring-specific job name
            pushgateway_url = 'http://pushgateway:9091'
            job_name = 'model_performance_monitoring'
            
            # Create Prometheus-friendly metrics
            metrics_data = ""
            
            # Extract precision and recall metrics at different k values
            for k in [5, 10, 20]:
                precision_key = f"precision@{k}"
                recall_key = f"recall@{k}"
                
                if precision_key in model_metrics:
                    precision_value = model_metrics[precision_key]
                    metrics_data += f"book_recommender_precision_at_{k} {precision_value}\n"
                
                if recall_key in model_metrics:
                    recall_value = model_metrics[recall_key]
                    metrics_data += f"book_recommender_recall_at_{k} {recall_value}\n"
            
            # Add additional derived metrics
            for k in [5, 10, 20]:
                precision_key = f"precision@{k}"
                recall_key = f"recall@{k}"
                
                if precision_key in model_metrics and recall_key in model_metrics:
                    precision_value = model_metrics[precision_key]
                    recall_value = model_metrics[recall_key]
                    
                    # Calculate F1 score
                    if precision_value + recall_value > 0:
                        f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
                        metrics_data += f"book_recommender_f1_at_{k} {f1_score}\n"
            
            # Push to Prometheus
            response = requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data=metrics_data
            )
            
            print(f"Model Performance Tracking: Pushed metrics to Prometheus: {response.status_code}")
            print(f"Metrics: {metrics_data}")
            
        except Exception as e:
            print(f"Model Performance Tracking Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    model_performance = PythonOperator(
        task_id='track_model_performance',
        python_callable=track_model_performance,
    )
    
    # Task 4: Analyze API Logs
    def analyze_api_logs():
        """Analyze API logs to extract error rates and performance metrics"""
        try:
            # Path to logs directory
            logs_dir = '/opt/airflow/logs'
            
            # Get today's date for log filtering
            today = datetime.now().strftime("%Y%m%d")
            
            # Look for API logs from today
            api_logs = [f for f in os.listdir(logs_dir) if f.startswith(f"api_{today}")]
            
            total_requests = 0
            error_requests = 0
            response_times = []
            
            # Process log files
            for log_file in api_logs:
                with open(os.path.join(logs_dir, log_file), 'r') as f:
                    for line in f:
                        # Count requests (adjust these patterns based on your actual log format)
                        if "request received" in line.lower():
                            total_requests += 1
                            
                        # Count errors
                        if "error" in line.lower() or "exception" in line.lower():
                            error_requests += 1
                        
                        # Extract response times if available in logs
                        if "response time:" in line.lower():
                            try:
                                response_time = float(line.split("response time:")[1].split("ms")[0].strip())
                                response_times.append(response_time)
                            except:
                                pass
            
            # Calculate metrics
            error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Push metrics to Prometheus
            pushgateway_url = 'http://pushgateway:9091'
            job_name = 'api_log_analysis'
            
            metrics_data = f"""
            book_recommender_total_requests {total_requests}
            book_recommender_error_requests {error_requests}
            book_recommender_error_rate {error_rate}
            book_recommender_avg_response_time {avg_response_time}
            """
            
            requests.post(
                f"{pushgateway_url}/metrics/job/{job_name}",
                data=metrics_data
            )
            
            print(f"Log Analysis: {total_requests} requests, {error_rate:.2f}% error rate")
        except Exception as e:
            print(f"Log Analysis Error: {str(e)}")
    
    log_analysis = PythonOperator(
        task_id='analyze_api_logs',
        python_callable=analyze_api_logs,
    )
    
    # Define task dependencies - these can run in parallel
    [api_health_check, system_resources, model_performance, log_analysis]