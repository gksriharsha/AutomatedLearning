import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import pprint
import os
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler

class Dataset(DataFrame):
    possible_label_headers = ['classes', 'class',
                              'labels', 'label', 'output', 'problems']

    _scaled = False
    def __init__(self, task='Supervised_Classification', data=None, labels=None, *args, **kwargs):
        if(task == 'Supervised_Classification'):
            self._supervised_classification_init(data, labels, *args, **kwargs)
            self.task = task
        else:
            print('Task not found')

    @property
    def _constructor(self):
        return Dataset

    def __call__(self):
        about_dict = {}
        about_dict['Using Dask'] = 'Yes' if self.use_dask else 'No'
        if(self.use_dask):
            try:
                shape = self.data.compute().shape
                about_dict['rows'] = shape[0]
                about_dict['columns'] = shape[1]
                about_dict['Unique classes'] = len(self.labels.compute().unique())
            except AttributeError:
                shape1 = self.X_train.compute().shape
                shape2 = self.X_test.compute().shape
                about_dict['rows'] = shape1[0]+shape2[0]
                about_dict['columns'] = shape1[1]+shape2[1]
                about_dict['Unique classes'] = len(self.y_train.compute().unique())
        else:
            try:
                about_dict['rows'] = self.data.shape[0]
                about_dict['columns'] = self.data.shape[1]
                about_dict['Unique classes'] = len(self.labels.unique())
                about_dict['elements'] = self.data.shape[0]*self.data.shape[1]
            except AttributeError:
                shape1 = self.X_train.shape
                shape2 = self.X_test.shape
                about_dict['rows'] = shape1[0]+shape2[0]
                about_dict['columns'] = shape1[1]+shape2[1]
                about_dict['Unique classes'] = len(self.y_train.unique())
        return about_dict

    def _check_for_classification(self):
        self.preprocesses = []
        if(self.use_dask):
            try:
                if(Dataset.no_of_nans(self.data.compute()) != 0):   
                    self.preprocesses.append('NANs')                
            except AttributeError:
                if(Dataset.no_of_nans(self.X_train.compute() != 0)):
                    self.preprocesses.append('NANs')


        else:
            try:
                if(Dataset.no_of_nans(self.data) != 0):   
                    self.preprocesses.append('NANs')                
            except AttributeError:
                if(Dataset.no_of_nans(self.X_train != 0)):
                    self.preprocesses.append('NANs')
            try:
                try:
                    int(self.labels.iloc[0])
                except AttributeError:
                    int(self.y_train.iloc[0])
            except:
                self.preprocesses.append('Non-numeric labels')

            if(self.preprocesses == []):
                self.preprocesses.append('All clear')
        return self.preprocesses

    def _supervised_dask_init(self):
        dataset = dd.read_csv(self.path)
        names = [i.lower() for i in dataset.columns]
        #global possible_label_headers
        if(any(i in names for i in self.possible_label_headers)):
            self.data = dataset.iloc[:, :-1]
            self.labels = dataset.iloc[:, -1]

    def _supervised_classification_init(self, data, labels, *args, **kwargs):
        path = kwargs.pop('path', None)
        data_path = kwargs.pop('data_path', None)
        labels_path = kwargs.pop('labels_path', None)

        super(Dataset, self).__init__(*args, **kwargs)

        self.path = path
        self.data_path = data_path
        self.labels_path = labels_path
        self.use_dask = False

        #global possible_label_headers

        if (not (path == None)):
            if ('.csv' not in path):
                print('Only CSV files are supported')
            else:
                file_size = os.stat(path).st_size/(1024*1024)
                if(file_size < 3):
                    dataset = pd.read_csv(path)
                    names = [i.lower() for i in dataset.columns]
                    if(any(i in names for i in self.possible_label_headers)):
                        self.data = dataset.iloc[:, :-1]
                        self.labels = dataset.iloc[:, -1]
                    else:
                        print('Labels are not found in the path')
                else:
                    self.use_dask = True
                    self._supervised_dask_init()

        elif(not(data_path == None)):
            try:
                if('.csv' not in kwargs['data_path']):
                    print('Only CSV files are supported')
                else:
                    try:
                        self.data = pd.read_csv(kwargs['data_path'])
                        self.labels = pd.read_csv(kwargs['labels_path'])
                    except:
                        print('Labels path not found')

            except Exception as e:
                print(e)

        else:
            if (isinstance(data, DataFrame)):
                self.data = data
            else:
                self.data = DataFrame(data)
            if (isinstance(labels, DataFrame)):
                self.labels = labels
            else:
                self.labels = DataFrame(labels)

        self._check_for_classification()
        self.train_test_split()

    @staticmethod
    def no_of_nans(set_to_be_checked):
        return set_to_be_checked.isnull().sum().sum()

    def train_test_split(self):
        if(self.use_dask):
            from dask_ml.model_selection import train_test_split
        else:
            from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, random_state=42)
        del self.data
        del self.labels
    
"""     def _scale_data(self):
        if(not self._scaled):
            self._scale = True """


if (__name__ == '__main__'):
    #dataset = Dataset(path=r'C:\Users\806707\Downloads\kc2 (1).csv')
    dataset = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print(dataset())
    print(dataset._check_for_classification())
    """ (a,b,c,d) = dataset.train_test_split()
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape) """
