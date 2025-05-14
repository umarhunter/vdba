#data_loader.py is a script that loads the data from the csv files and returns the data in the form of a pandas dataframe.
import os
import sys
import pandas as pd

def load_medicare_data():
    # Load the data from the csv files
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the data file
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'insurance', 'medicare.csv')
    
    # Read the full dataset
    sampled_medicare_data = pd.read_csv(data_path, dtype='str')
    
    # # Sample 50,000 rows randomly
    # sampled_data = medicare_data.sample(n=50000, random_state=42)
    
    # # Save the sampled dataset back to the same file
    # sampled_data.to_csv(data_path, index=False)
    
    return sampled_medicare_data

def load_cuad_data():
    # Load the data from the csv files
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the data file
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'legal', 'CUAD_v1', 'master_clauses.csv')
    
    # Read the full dataset
    cuad_data = pd.read_csv(data_path, dtype='str')
    
    return cuad_data