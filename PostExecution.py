import pandas

KNN = pandas.read_csv('CloudResults/results/KNN.csv')

results = pandas.read_csv('CloudResults/results/results.csv')

Metadata = pandas.read_csv('CloudResults/MetaData.csv')

KNN_results = pandas.merge(KNN,results,on='eID',how='left')

KNN_results = pandas.merge(KNN_results,Metadata,how='left',left_on='fID_x',right_on='fID')

KNN_results.to_csv('CloudResults/results/KNN_results.csv',index=False)