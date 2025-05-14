#data_loader.py is a script that loads the data from the csv files and returns the data in the form of a pandas dataframe.
import os
import sys
import pandas as pd



def load_medicare_data():
    # Load the data from the csv files
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the data file
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'insurance', 'medicare', '2022', 'medicare.csv')

    medicare_data = pd.read_csv(data_path, dtype='str')

    return medicare_data