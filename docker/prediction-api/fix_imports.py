#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for model loading issues in the prediction API service.

This module ensures that the required model classes are properly defined
in the appropriate namespaces before attempting to load pickled models.
"""

import os
import sys
import pickle
import importlib
import io

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

try:
    # Import the necessary classes to make them available for unpickling
    # First, try to import from the src package
    from src.models.model_utils import BaseRecommender
    from src.models.train_model import CollaborativeRecommender
    print("Successfully imported CollaborativeRecommender from src.models.train_model")
except ImportError:
    try:
        # If that fails, try importing directly
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.model_utils import BaseRecommender
        from models.train_model import CollaborativeRecommender
        print("Successfully imported CollaborativeRecommender from models.train_model")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

# Create a custom Unpickler class to handle model loading
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Special case for our model classes
        if name == 'CollaborativeRecommender':
            return CollaborativeRecommender
        elif name == 'BaseRecommender':
            return BaseRecommender
        # For everything else, use the standard behavior
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            # If standard import fails, try to import the module directly
            try:
                module_obj = importlib.import_module(module)
                return getattr(module_obj, name)
            except (ImportError, AttributeError):
                print(f"Warning: Could not import {name} from {module}, returning None")
                return None

# Monkey patch the pickle.load functions to use our custom unpickler
original_load = pickle.load
original_loads = pickle.loads

def custom_load(file, **kwargs):
    return CustomUnpickler(file, **kwargs).load()

def custom_loads(data, **kwargs):
    return CustomUnpickler(io.BytesIO(data), **kwargs).load()

# Replace the standard pickle load functions with our custom versions
pickle.load = custom_load
pickle.loads = custom_loads

print("Import fix applied successfully. CollaborativeRecommender class is now available for unpickling.")
