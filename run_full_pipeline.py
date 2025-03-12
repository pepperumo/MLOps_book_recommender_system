#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the complete book recommender system pipeline in sequence.
This script executes all components one by one using Python's subprocess module:
1. retrieve_raw_data
2. process_data
3. build_features
4. train_model
5. evaluate_model
6. test_model
7. predict_model
8. test_api
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'pipeline_run_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pipeline_runner')

def run_command(cmd, cwd=None):
    """Run a shell command and log its output."""
    logger.info(f"Running command: {cmd}")
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
        )
        
        # Log the output
        duration = time.time() - start_time
        if result.returncode == 0:
            logger.info(f"Command completed successfully in {duration:.2f} seconds")
            if result.stdout:
                logger.info(f"Output: {result.stdout[:500]}...")
            return True
        else:
            logger.error(f"Command failed with exit code {result.returncode} after {duration:.2f} seconds")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Exception while running command: {e} after {duration:.2f} seconds")
        return False

def run_pipeline():
    """Run all pipeline steps in sequence."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    activate_cmd = os.path.join(project_dir, "venv", "Scripts", "activate")
    
    # Define the sequence of commands to run
    commands = [
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.data.retrieve_raw_data\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.data.process_data\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.features.build_features\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.models.train_model\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.models.evaluate_model\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.models.test_model\"",
        f"powershell -Command \"& '{activate_cmd}' ; python -m src.models.predict_model user 125 --num 5\"",
        f"powershell -Command \"& '{activate_cmd}' ; pytest -xvs src/api/test_api.py\""
    ]
    
    # Store the names of each step for better logging
    step_names = [
        "Retrieve Raw Data",
        "Process Data",
        "Build Features",
        "Train Model",
        "Evaluate Model",
        "Test Model",
        "Predict Model",
        "Test API"
    ]
    
    # Run each command in sequence
    all_succeeded = True
    for i, (step_name, cmd) in enumerate(zip(step_names, commands), 1):
        logger.info(f"Step {i}/{len(commands)}: {step_name}")
        success = run_command(cmd, project_dir)
        
        if not success:
            logger.error(f"Pipeline failed at step {i}: {step_name}")
            all_succeeded = False
            break
        
        logger.info(f"Step {i} ({step_name}) completed successfully")
    
    if all_succeeded:
        logger.info("Pipeline completed successfully!")
        return 0
    else:
        logger.error("Pipeline failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_pipeline())
