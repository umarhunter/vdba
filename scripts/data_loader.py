#data_loader.py is a script that loads the data from the csv files and returns the data in the form of a pandas dataframe.
import os
import sys
import pandas as pd



def load_medicare_data():
    # Load the data from the csv files
    # Get the project root directory (2 levels up from notebook)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Construct the path to the data file
    # data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'insurance', 'medicare', '2022', 'medicare.csv')
    data_path = os.path.join(PROJECT_ROOT, 'vdba', 'data', 'processed', 'sample_ny_data.csv')
    medicare_data = pd.read_csv(data_path, dtype='str').head(100)
    # ny_data = medicare_data[medicare_data['Rndrng_Prvdr_State_Abrvtn'] == 'NY']
    # ny_data_sample = ny_data.sample(n=50000, random_state=42)
    return medicare_data