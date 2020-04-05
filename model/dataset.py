import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import pprint
import os
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
from random import randint
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.impute import KNNImputer
from csv import DictReader
from ttictoc import TicToc
from sklearn import preprocessing
class Dataset(DataFrame):

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
        if(not(self.meta_id == None)):
            MetaData_line = {}
            with open('dataset_fetch/meta/MetaData.csv','r') as read_obj:
                csv_dict_reader = DictReader(read_obj)
                for row in csv_dict_reader:
                    if(row['fID'] == self.meta_id):
                        MetaData_line = row
                        break
                about_dict['rows'] = int(MetaData_line['NumberOfInstances'])
                about_dict['columns'] = int(MetaData_line['NumberOfFeatures'])
                about_dict['Unique classes'] = int(MetaData_line['NumberOfClasses'])
                about_dict['Contains NANs'] = int(MetaData_line['NumberOfMissingValues'])
                about_dict['elements'] = int(MetaData_line['NumberOfInstances'])*int(MetaData_line['NumberOfFeatures'])

        else:
            if(self.use_dask):
                try:
                    shape = self.data.compute().shape
                    about_dict['rows'] = shape[0]
                    about_dict['columns'] = shape[1]
                    about_dict['Unique classes'] = len(np.unique(self.labels.compute()))
                    about_dict['elements'] = shape[0]*shape[1]
                    about_dict['Contains NANs'] = 'Yes' if self.data.compute().isnull().sum().sum() != 0 else 'No'
                except AttributeError:
                    shape1 = self.X_train.compute().shape
                    shape2 = self.X_test.compute().shape
                    about_dict['rows'] = shape1[0]+shape2[0]
                    about_dict['columns'] = shape1[1]+shape2[1]
                    about_dict['Unique classes'] = len(self.y_train.compute().unique())
                    about_dict['elements'] = shape1[0]*shape1[1] + shape2[0]*shape2[1]
                    about_dict['Contains NANs'] = 'Yes' if self.data.isnull().sum().sum() != 0 else 'No'
            else:
                try:
                    about_dict['rows'] = self.data.shape[0]
                    about_dict['columns'] = self.data.shape[1]
                    about_dict['Unique classes'] = len(np.unique(self.labels))
                    about_dict['elements'] = self.data.shape[0]*self.data.shape[1]
                except AttributeError:
                    shape1 = self.X_train.shape
                    shape2 = self.X_test.shape
                    about_dict['rows'] = shape1[0]+shape2[0]
                    about_dict['columns'] = shape1[1]+shape2[1]
                    about_dict['elements'] = self.data.shape[0]*self.data.shape[1]
                    about_dict['Unique classes'] = len(self.y_train.unique())
        return about_dict

    def _check_for_classification(self):
        self.preprocesses = []
        def check_nans(self):
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
        def check_numeric_labels(self):
            try:
                try:
                    int(self.labels.iloc[0])
                except AttributeError:
                    int(self.y_train.iloc[0])
            except:
                self.preprocesses.append('Non-numeric labels')

        check_nans(self)
        check_numeric_labels(self)
        if(self.preprocesses == []):
            self.preprocesses.append('All clear')
        return self.preprocesses

    def _supervised_dask_init(self):
        if(self.path):
            dataset = dd.read_csv(self.path,na_values=['?',"?",'-',"-",'_',"_",'\'?\'','\"?\"'])
        if(self.url_path):
            dataset = dd.read_csv(self.url_path,na_values=['?',"?",'-',"-",'_',"_",'\'?\'','\"?\"'])
        if(not(self.meta_id == None)):
            MetaData_line = {}
            with open('dataset_fetch/meta/Data.csv','r') as read_obj:
                csv_dict_reader = DictReader(read_obj)
                for row in csv_dict_reader:
                    if(row['Metadata'] == self.meta_id):
                        MetaData_line = row
                        break
            Labels_name = MetaData_line['Label_header'].strip()
            self.labels = dataset.loc[:,Labels_name]
            dataset = dataset.drop(columns = Labels_name,axis=1)
            self.data = dataset
        else:
            names = [i.lower() for i in dataset.columns]
            if(any(i in names for i in self.possible_label_headers)):
                label_header = list(set(names).intersection(set(self.possible_label_headers)))
                if(len(label_header) == 1):                 
                    self.labels = dataset.loc[:,list(map(lambda x:True if x.lower() == label_header[0] else False,dataset.columns))]
                    dataset.drop(label_header[0],axis=1)
                    self.data = dataset
                    self.data = self.data.apply(pd.to_numeric,axis=1, args=('coerce',),meta = self.data.dtypes)

    def _supervised_classification_init(self, data, labels, *args, **kwargs):
        path = kwargs.pop('path', None)
        url_path = kwargs.pop('url_path',None)
        data_path = kwargs.pop('data_path', None)
        labels_path = kwargs.pop('labels_path', None)
        meta_id = kwargs.pop('meta_id',None)

        super(Dataset, self).__init__(*args, **kwargs)

        self.path = path
        self.url_path = url_path
        self.data_path = data_path
        self.labels_path = labels_path
        self.use_dask = False
        self.imputed_columns = []
        self._scaled = False
        self._split = False
        self.meta_id = meta_id
        self.possible_label_headers = ['classes', 'class',
                              'labels', 'label', 'output', 'problems']

        if (not (path == None)):
            if ('.csv' not in path):
                print('Only CSV files are supported')
            else:
                file_size = os.stat(path).st_size/(1024*1024)
                if(file_size < 3):
                    dataset = pd.read_csv(path,na_values=['?',"?",'-',"-",'_',"_",'\'?\'','\"?\"'])
                    if(not(self.meta_id == None)):
                        MetaData_line = {}
                        with open('dataset_fetch/meta/Data.csv','r') as read_obj:
                            csv_dict_reader = DictReader(read_obj)
                            for row in csv_dict_reader:
                                if(row['Metadata'] == self.meta_id):
                                    MetaData_line = row
                                    break
                        Labels_name = MetaData_line['Label_header'].strip()
                        self.labels = dataset.loc[:,Labels_name]
                        dataset = dataset.drop(columns = Labels_name,axis=1)
                        self.data = dataset
                        self.meta_id = meta_id
                    else:
                        names = [i.lower() for i in dataset.columns]
                        if(any(i in names for i in self.possible_label_headers)):
                            label_header = list(set(names).intersection(set(self.possible_label_headers)))
                            if(len(label_header) == 1):                 
                                self.labels = dataset.loc[:,list(map(lambda x:True if x.lower() == label_header[0] else False,dataset.columns))]
                                dataset.drop(label_header[0],axis=1)
                                self.data = dataset
                                self.data = self.data.apply(pd.to_numeric,axis=1, args=('coerce',))
                        else:
                            print('Labels are not found in the path')
                else:
                    self.use_dask = True
                    self._supervised_dask_init()

        elif(not(data_path == None)):
            try:
                if('.csv' not in data_path):
                    print('Only CSV files are supported')
                else:
                    try:
                        self.data = pd.read_csv(kwargs['data_path'])
                        self.labels = pd.read_csv(kwargs['labels_path'])
                    except:
                        print('Labels path not found')

            except Exception as e:
                print(e)

        elif(not(url_path == None)):
            if ('http' not in url_path):
                print('Only URL files are supported')
            else:
                downloaded_csv = dd.read_csv(url_path,na_values=['?',"?",'-',"-",'_',"_",'\'?\'','\"?\"'])
                file_size = downloaded_csv.memory_usage().sum().compute()/(1024*1024)
                if(file_size < 3):
                    dataset = downloaded_csv.compute()
                    if(not(self.meta_id == None)):
                        MetaData_line = {}
                        with open('dataset_fetch/meta/Data.csv','r') as read_obj:
                            csv_dict_reader = DictReader(read_obj)
                            for row in csv_dict_reader:
                                if(row['Metadata'] == self.meta_id):
                                    MetaData_line = row
                                    break
                        Labels_name = MetaData_line['Label_header'].strip()
                        self.labels = dataset.loc[:,Labels_name]
                        dataset = dataset.drop(columns = Labels_name,axis=1)
                        self.data = dataset
                        self.meta_id = meta_id
                    else:                 
                        names = [i.lower() for i in dataset.columns]
                        if(any(i in names for i in self.possible_label_headers)):
                            label_header = list(set(names).intersection(set(self.possible_label_headers)))
                            if(len(label_header) == 1):                 
                                self.labels = dataset.loc[:,list(map(lambda x:True if x.lower() == label_header[0] else False,dataset.columns))]
                                dataset.drop(label_header[0],axis=1)
                                self.data = dataset
                                #self.data = self.data.apply(pd.to_numeric,axis=1, args=('coerce',),meta = self.data.dtypes)
                        else:
                            print('Labels are not found in the path')
                else:
                    self.use_dask = True
                    self._supervised_dask_init()
        else:
            if (isinstance(data, DataFrame)):
                self.data = data
            else:
                self.data = DataFrame(data)
            if (isinstance(labels, DataFrame)):
                self.labels = labels
            else:
                self.labels = DataFrame(labels)

        #self.data.replace(to_replace=r"'[?!@#$%&]'|[?!@#$%&]",value=np.nan,regex=True)

    def split(self):
        if(self.use_dask):
            from dask_ml.model_selection import train_test_split
        else:
            from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, random_state=42)
        del self.data
        del self.labels
        self._split = True
    
    def contains_text_check(self):
        dataset = self.data
        textcolumns = dataset.applymap(np.isreal)
        if(textcolumns.sum().sum() > 0 ):
            return True
        else:
            return False

    def _scale_data(self):
        if(not self._scaled):
            self._scale = True
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)

    def _impute_data(self,**params):
        self.imputation_method = params.pop('imputation_method')
        full_dataset = pd.concat([self.data,self.labels],axis=1)
        def remove_nan_columns(self):
            #nan_indices = np.argwhere(full_dataset.isnull().values>0)
            nan_values = np.array(full_dataset.isnull().sum(axis=0))
            column_numbers = list(np.where(nan_values>0)[0])
            for i in column_numbers:
                if(nan_values[i]*100/full_dataset.shape[0] >= 50):
                    full_dataset.drop(full_dataset.columns[[i]],axis=1,inplace=True)
            self.labels = full_dataset.iloc[:,-1]
            self.data = full_dataset.iloc[:,:-1]
        remove_nan_columns(self)
        self.imputed_columns = self.data.columns[self.data.isnull().any()].tolist()
        self.nan_indices = np.argwhere(self.data.isnull().values>0)
        if(self.imputation_method == 'mean'):            
            self.data =  self.data.fillna( self.data.mean())
        elif(self.imputation_method == 'median'):
            self.data =  self.data.fillna( self.data.median())
        elif(self.imputation_method == 'KNN'):
            imputer = KNNImputer(n_neighbors=2, weights="uniform")
            self.data = imputer.fit_transform(self.data)

    def encode(self):
        if(not(self.meta_id == None)):
            #MetaData_line = {}
            with open('dataset_fetch/meta/Data.csv','r') as read_obj:
                csv_dict_reader = DictReader(read_obj)
                for row in csv_dict_reader:
                    if(row['Metadata'] == self.meta_id):
                        MetaData_line = row
                        break
                #Column_info = MetaData_line['FeatureType'].split(',')
                self.data = pd.get_dummies(self.data)
                le = preprocessing.LabelEncoder()
                self.labels = le.fit_transform(self.labels)
                #print(len(self.labels))
if(__name__ == '__main__'):
    #dataset = Dataset(path=r'C:\Users\806707\Downloads\kc2 (1).csv')
    #dataset = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    t = TicToc()
    t.tic()
    dataset = Dataset(url_path=r'https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff',meta_id ='c3d0a210-6f55-4b89-8a38-6e3404653c78')
    t.toc()
    print(t.elapsed)
    #dataset = Dataset(url_path=r'https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff')
    #dataset = Dataset(path=r'C:\Users\806707\Downloads\arrhythmia.csv')
    #print(dataset())
    #print(dataset._impute_data(imputation_method='KNN'))
    #print(dataset())
    print(dataset.encode())
    print(dataset.data)
    print(dataset.labels)
    #print(dataset._check_for_classification())
    """ (a,b,c,d) = dataset.train_test_split()
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape) """
