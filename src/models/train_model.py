"""
Main interface for training book recommender models.

This module provides a command-line interface for training and evaluating
different types of book recommender models: collaborative filtering,
content-based filtering, and hybrid approaches.
"""
import pandas as pd
import numpy as np
import logging
import os
import sys
import traceback
from datetime import datetime
import argparse

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'train_model_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('train_model')

# Import the recommender implementations
from .train_model_base import load_data, evaluate_model_with_test_data
from .train_model_collaborative import CollaborativeRecommender, train_model as train_collaborative
from .train_model_content import ContentBasedRecommender, train_model as train_content
from .train_model_hybrid import HybridRecommender, train_model as train_hybrid


def train_selected_model(model_type='hybrid', collaborative_weight=0.7, eval_model=True):
    """
    Train a selected model type.
    
    Parameters
    ----------
    model_type : str
        Type of recommender to train: 'collaborative', 'content', or 'hybrid'
    collaborative_weight : float
        Weight to give collaborative filtering in hybrid model (0-1)
    eval_model : bool
        Whether to evaluate the model after training
        
    Returns
    -------
    BaseRecommender
        Trained recommender model
    """
    try:
        logger.info(f"Training {model_type} recommender model")
        
        # Train the selected model type
        if model_type == 'collaborative':
            recommender = train_collaborative()
        elif model_type == 'content':
            recommender = train_content()
        elif model_type == 'hybrid':
            recommender = train_hybrid(collaborative_weight)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
        if recommender is None:
            logger.error(f"Failed to train {model_type} recommender model")
            return None
            
        if eval_model:
            # Evaluate the model
            logger.info(f"Evaluating {model_type} recommender model")
            results = evaluate_model_with_test_data(recommender)
            logger.info(f"Evaluation results: {results}")
            
        return recommender
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.debug(traceback.format_exc())
        return None


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Train and evaluate book recommender models'
        )
        parser.add_argument(
            '--model-type', 
            type=str, 
            default='hybrid',
            choices=['collaborative', 'content', 'hybrid'],
            help='Type of recommender model to train'
        )
        parser.add_argument(
            '--collaborative-weight', 
            type=float, 
            default=0.7,
            help='Weight to give collaborative filtering in hybrid model (0-1)'
        )
        parser.add_argument(
            '--eval', 
            action='store_true',
            help='Evaluate the model after training'
        )
        parser.add_argument(
            '--model-dir', 
            type=str, 
            default='models',
            help='Directory to save the model'
        )
        args = parser.parse_args()
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Train the selected model
        recommender = train_selected_model(
            model_type=args.model_type,
            collaborative_weight=args.collaborative_weight,
            eval_model=args.eval
        )
        
        if recommender is None:
            logger.error("Model training failed")
            sys.exit(1)
            
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
