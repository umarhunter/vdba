import pandas as pd
# data_loader.py is a script that loads the data from the csv files and returns the data in the form of a pandas dataframe.

def load_medicare_data():
    # Load the data from the csv files
    medicare_data = pd.read_csv('data/insurance/medicare/2022/medicare.csv', dtype='str')
    return medicare_data