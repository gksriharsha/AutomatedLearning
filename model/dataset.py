import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import pprint


class Dataset(DataFrame):
    def __init__(self, task='Supervised_Classification', data=None, labels=None, *args, **kwargs):
        if(task == 'Supervised_Classification'):
            self._supervised_classification_init(data, labels,*args, **kwargs)
            self.task = task
        else:
            print('Task out found')

    @property
    def _constructor(self):
        return Dataset

    def __call__(self):
        about_dict = {}
        about_dict['rows'] = self.data.shape[0]
        about_dict['columns'] = self.data.shape[1]
        about_dict['Unique classes'] = len(self.labels.unique())
        about_dict['elements'] = self.data.shape[0]*self.data.shape[1]
        return about_dict

    def _check_for_classification(self):
        self.preprocesses = []
        if(Dataset.no_of_nans(self.data) != 0):
            self.preprocesses.append('NANs')
        try:
            int(self.labels.iloc[0])
        except:
            self.preprocesses.append('Non-numeric labels')
        
        return self.preprocesses
    
    def _supervised_classification_init(self, data, labels,*args,**kwargs):
        path = kwargs.pop('path', None)
        data_path = kwargs.pop('data_path', None)
        labels_path = kwargs.pop('labels_path', None)

        super(Dataset, self).__init__(*args, **kwargs)

        self.path = path
        self.data_path = data_path
        self.labels_path = labels_path

        label_names = ['classes', 'class',
                       'labels', 'label', 'output', 'problems']

        if (not (path == None)):
            if ('.csv' not in path):
                print('Only CSV files are supported')
            else:
                dataset = pd.read_csv(path)
                names = dataset.columns
                if(any(i in names for i in label_names)):
                    self.data = dataset.iloc[:, :-1]
                    self.labels = dataset.iloc[:, -1]
                else:
                    print('Labels are not found in the path')

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

    @staticmethod
    def no_of_nans(set_to_be_checked):
        return set_to_be_checked.isnull().sum().sum()


if (__name__ == '__main__'):
    dataset = Dataset(path=r'C:\Users\806707\Downloads\kc2 (1).csv')    
    print(dataset._check_for_classification())
