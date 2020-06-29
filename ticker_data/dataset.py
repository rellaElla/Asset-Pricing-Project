import pandas as pd
from pathlib import Path
from preprocess import preprocess
from time import time

class Dataset(object):
    """
    Attributes:
    filename: str
    label_col: str
    text_col: str
    dataframe: pd.DataFrame
    Properties:
    data: returns raw data
    cleaned_data: returns cleaned data
    Methods:
    load: fetches csv file using filename
    preprocess_texts: calls preprocess on self.dataframe
    """
    def __init__(self, filename, label_col='label', text_col='text'):
        self.filename: str = filename
        self.label_col: str = label_col
        self.text_col: str = text_col

    @property
    def data(self):
        data: pd.DataFrame = self.dataframe[[self.label_col, self.text_col]].copy()
        data.columns = ['label', 'text']
        return data

    @property
    def cleaned_data(self):
        data: pd.DataFrame =  self.dataframe[[self.label_col, 'cleaned']]
        data.columns = ['label', 'text']
        return data

    def load(self):
        df: pd.DataFrame = pd.read_csv(Path(self.filename).resolve(), encoding='latin-1')
        self.dataframe = df

    def preprocess_texts(self, quiet=False):
        self.dataframe['cleaned'] = preprocess(self.dataframe[self.text_col], quiet)