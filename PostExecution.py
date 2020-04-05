import pandas

KNN = pandas.read_csv('results/KNN.csv')

results = pandas.read_csv('results/results.csv')

Metadata = pandas.read_csv('dataset_fetch/meta/MetaData.csv')

KNN_results = pandas.merge(KNN,results)

KNN_results = pandas.merge(KNN_results,Metadata,how='left')

KNN_results.to_csv('results/KNN_results.csv',index=False)