from sklearn.decomposition import PCA
import sys

sys.path.append(r'V:\Python\AutomatedLearning')

from model.dataset import Dataset

def perform_pca(dataset):
    pca = PCA(n_components=3)
    if(not(dataset._split)):
        dataset.split()
    pca.fit(dataset.X_train)
    dataset.X_train = pca.transform(dataset.X_train)
    dataset.X_test = pca.transform(dataset.X_test)
    print(pca.explained_variance_ratio_)


if(__name__ == '__main__'):
    d = Dataset(url_path=r'https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff',meta_id ='c3d0a210-6f55-4b89-8a38-6e3404653c78')
    d.encode()
    print(d.data)
    perform_pca(d)
    print(d.X_train)
    print(d.X_test)