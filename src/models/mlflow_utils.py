"""
MLflow utility functions for experiment tracking.

This module provides helper functions for MLflow tracking and integration with DAGsHub.
"""
import os
import logging
import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

def setup_mlflow(repo_owner='pepperumo', repo_name='MLOps_book_recommender_system'):
    """
    Set up MLflow tracking with DAGsHub.
    
    Parameters
    ----------
    repo_owner : str
        DAGsHub repository owner
    repo_name : str
        DAGsHub repository name
        
    Returns
    -------
    bool
        True if setup was successful, False otherwise
    """
    try:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logger.info(f"MLflow tracking configured with DAGsHub for {repo_owner}/{repo_name}")
        return True
    except Exception as e:
        logger.warning(f"Could not set up MLflow tracking: {e}")
        return False

def log_params_from_model(model, prefix: str = ''):
    """
    Log parameters from a model object to MLflow.
    
    Parameters
    ----------
    model : object
        Model object with a params attribute
    prefix : str, optional
        Prefix to add to parameter names
    """
    if not hasattr(model, 'params'):
        logger.warning("Model doesn't have params attribute, skipping parameter logging")
        return
    
    for param_name, param_value in model.params.items():
        try:
            # Add prefix if provided
            full_param_name = f"{prefix}_{param_name}" if prefix else param_name
            
            # Handle different parameter types
            if isinstance(param_value, (str, int, float, bool)):
                mlflow.log_param(full_param_name, param_value)
            else:
                # For complex types, convert to string
                mlflow.log_param(full_param_name, str(param_value))
                
            logger.debug(f"Logged parameter {full_param_name}: {param_value}")
        except Exception as e:
            logger.warning(f"Could not log parameter {param_name}: {e}")

def log_metrics_safely(metrics: Dict[str, float], prefix: str = ''):
    """
    Safely log metrics to MLflow, handling special characters and non-numeric values.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics to log
    prefix : str, optional
        Prefix to add to metric names
    """
    for metric_name, metric_value in metrics.items():
        try:
            # Ensure metric is a numeric value
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                # Replace problematic characters
                clean_metric_name = metric_name.replace('@', '_').replace(' ', '_')
                
                # Add prefix if provided
                if prefix:
                    clean_metric_name = f"{prefix}_{clean_metric_name}"
                    
                mlflow.log_metric(clean_metric_name, float(metric_value))
                logger.info(f"Logged metric {clean_metric_name}: {metric_value}")
            else:
                logger.warning(f"Skipping non-numeric metric: {metric_name}={metric_value}")
        except Exception as e:
            logger.warning(f"Could not log metric {metric_name}: {e}")

def log_model_version_as_tag(model_version: str, config_path: Optional[str] = None):
    """
    Log model version information as tags in MLflow.
    
    Parameters
    ----------
    model_version : str
        Model version or configuration name
    config_path : str, optional
        Path to configuration file
    """
    try:
        mlflow.set_tag("model_version", model_version)
        
        # If config path is provided, try to extract config section
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if model_version in config:
                        # Create a compact string representation of the config
                        config_str = str(config[model_version])
                        mlflow.set_tag("config_summary", config_str[:250] + "..." if len(config_str) > 250 else config_str)
            except Exception as e:
                logger.warning(f"Could not extract config for tagging: {e}")
                
    except Exception as e:
        logger.warning(f"Could not set model version tags: {e}")

def log_precision_recall_curve(precision_dict: Dict[str, float], recall_dict: Dict[str, float], k_values: List[int]):
    """
    Create and log a precision-recall curve as an artifact in MLflow.
    
    Parameters
    ----------
    precision_dict : Dict[str, float]
        Dictionary of precision values at different k
    recall_dict : Dict[str, float]
        Dictionary of recall values at different k
    k_values : List[int]
        List of k values
    """
    try:
        # Extract precision and recall values
        precision_values = [precision_dict.get(f"precision@{k}", 0) for k in k_values]
        recall_values = [recall_dict.get(f"recall@{k}", 0) for k in k_values]
        
        # Create data for logging
        k_metrics = {
            f"k_values": k_values,
            f"precision_values": precision_values,
            f"recall_values": recall_values
        }
        
        # Save as CSV for visualization
        import pandas as pd
        df = pd.DataFrame({
            "k": k_values,
            "precision": precision_values,
            "recall": recall_values
        })
        
        # Save to temp file and log as artifact
        temp_file = "precision_recall_data.csv"
        df.to_csv(temp_file, index=False)
        mlflow.log_artifact(temp_file)
        
        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass
            
        logger.info("Successfully logged precision-recall data")
    except Exception as e:
        logger.warning(f"Could not create precision-recall data: {e}")

def get_dagshub_url(run_id: Optional[str] = None):
    """
    Get the DAGsHub URL for the current or specified MLflow run.
    
    Parameters
    ----------
    run_id : str, optional
        MLflow run ID, if None, uses the active run
        
    Returns
    -------
    str
        URL to the MLflow run in DAGsHub
    """
    try:
        if run_id is None and mlflow.active_run() is not None:
            run_id = mlflow.active_run().info.run_id
        
        if run_id:
            return f"https://dagshub.com/pepperumo/MLOps_book_recommender_system.mlflow/#/experiments/0/runs/{run_id}"
        else:
            logger.warning("No active run or run_id provided")
            return "https://dagshub.com/pepperumo/MLOps_book_recommender_system.mlflow"
    except Exception as e:
        logger.warning(f"Could not generate DAGsHub URL: {e}")
        return "https://dagshub.com/pepperumo/MLOps_book_recommender_system.mlflow"
