#data_loader.py is a script that loads the data from the csv files and returns the data in the form of a pandas dataframe.
import os
import sys
import pandas as pd

def load_medicare_data():
    # Load the data from the csv files
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the data file
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'insurance', 'medicare.csv')
    
    # Read the dataset
    try:
        sampled_medicare_data = pd.read_csv(data_path, dtype='str')
        sampled_medicare_data = sampled_medicare_data.fillna('')  # Fill NaN values with empty strings
    except FileNotFoundError:
        raise FileNotFoundError(f"Medicare dataset not found at {data_path}. Please ensure the data file exists.")
    except Exception as e:
        raise Exception(f"Error loading Medicare dataset: {str(e)}")

    return sampled_medicare_data

def load_cuad_data():
    """Load and process the CUAD dataset."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'legal', 'CUAD_v1', 'master_clauses.csv')
    
    try:
        cuad_data = pd.read_csv(data_path, dtype='str')
        # Fill NaN values with empty strings
        cuad_data = cuad_data.fillna('')
        return cuad_data
    except FileNotFoundError:
        raise FileNotFoundError(f"CUAD dataset not found at {data_path}. Please ensure the data file exists.")
    except Exception as e:
        raise Exception(f"Error loading CUAD dataset: {str(e)}")