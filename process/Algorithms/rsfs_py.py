import numpy
import numpy.matlib
import time
import math
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 


def RSFS(dataset, Parameters):
    Feature_train, Feature_test, label_train, label_test = dataset.X_train,dataset.X_test,dataset.y_train,dataset.y_test
    max_iters = 3000000
    n_dummyfeats = Parameters['Dummy feats']
    max_delta = 0.05
    k_neighbors = 3
    #label_test = label_test.astype('int')
    #label_train = label_train.astype('int')
    verbose = 1
    N_classes = len(numpy.unique(label_train))
    number_of_features = numpy.size(Feature_train, axis=1)
    relevance = numpy.zeros((number_of_features,))
    dummy_relevance = numpy.zeros((n_dummyfeats,))
    #stored =[]

    if (Parameters['fn'] == 'sqrt'):
        feats_to_take = round(math.sqrt(number_of_features))
        #feats_to_take = feats_to_take.astype('int')
        dummy_feats_to_take = round(math.sqrt(n_dummyfeats))
        #dummy_feats_to_take = dummy_feats_to_take.astype('int')
    if (Parameters['fn'] == '10log'):
        feats_to_take = round(10 * math.log10(number_of_features))
        #feats_to_take = feats_to_take.astype('int')
        dummy_feats_to_take = round(10 * math.log10(n_dummyfeats))
        #dummy_feats_to_take = dummy_feats_to_take.astype('int')

    feat_N = numpy.zeros(max_iters)

    totcorrect = numpy.zeros(N_classes)
    totwrong = numpy.zeros(N_classes)

    iteration = 1
    deltaval = math.inf
    cutoff = Parameters['cutoff']    
    Threshold = Parameters['Threshold']
    probs = numpy.zeros(numpy.shape(relevance))
    #if(Parameters['Classifier'] == 'KNN'):
    clf = KNeighborsClassifier(n_neighbors=k_neighbors)
    while (iteration <= max_iters and deltaval > max_delta):
        feature_indices =  numpy.floor(number_of_features * numpy.random.rand(1, feats_to_take))
        feature_indices = feature_indices.astype('int')
        # if ('stored' in locals()):
        #     for i in list(range(0, len(stored))):
        #         feature_indices = feature_indices(feature_indices != stored(i))

        
        
        class_hypos = clf.fit(Feature_train[:, numpy.resize(feature_indices,(numpy.size(feature_indices),))], label_train).predict(Feature_test[:,numpy.resize(feature_indices,(numpy.size(feature_indices),))])
        
        correct = numpy.zeros(N_classes)
        wrong = numpy.zeros(N_classes)

        for j in list(numpy.arange(0, numpy.size(label_test))):
            if (label_test[j] == class_hypos[j]):
                correct[label_test[j] - 1] = correct[label_test[j] - 1] + 1
            else:
                wrong[label_test[j] - 1] = wrong[label_test[j] - 1] + 1

        totcorrect = totcorrect + correct
        totwrong = totwrong + wrong

        performance_criterion = numpy.mean(numpy.array(correct) * 100 / (numpy.array(correct) + numpy.array(wrong)))
        expected_criterion_value = numpy.mean(numpy.array(totcorrect) * 100 / (numpy.array(totcorrect) + numpy.array(totwrong)))

        target = performance_criterion - expected_criterion_value
        pos = feature_indices
        relevance[pos] += target

        dummy_indices = numpy.floor(n_dummyfeats * numpy.random.rand(1,dummy_feats_to_take))
        dummy_indices = dummy_indices.astype('int')
        target = dummy_relevance[dummy_indices] + performance_criterion - expected_criterion_value
        pos = dummy_indices
        for x, y in zip(pos, target):
            dummy_relevance[x] = y
        if(iteration>5):
            probs = norm.cdf(relevance, loc=numpy.mean(dummy_relevance), scale=numpy.std(dummy_relevance))


        feat_N[iteration] = numpy.size(numpy.where(probs > cutoff))

        if (iteration % Threshold == 0):
            if (verbose == 1):
                deltaval = numpy.std(feat_N[iteration - (Threshold-1):iteration]) / numpy.mean(feat_N[iteration - (Threshold-1):iteration])
                print('RSFS: ', feat_N[iteration], 'features chosen so far (iteration: ', iteration, '/', max_iters,'). Delta: ', deltaval)

        iteration = iteration + 1

        # if (Parameters['RSFS']['stored'] == 1):
        #     top = Parameters['RSFS']['top']
        #     Threshold = Parameters['RSFS']['Threshold']
        #     if (iteration > Threshold):
        #         S = numpy.where(probs > cutoff)
        #     W = relevance[S]
        #     comm = [S, W]
        #     comm = comm[comm[:, 1].argsort(),]
        #     if (len(S) >= top):
        #         stored.extend(comm[0:top-1, 1])
        #     else:
        #         stored.extend(comm[0:len(S)-1, 1])
        #     stored = list(numpy.unique(stored))

    S = numpy.where(probs>cutoff)
    W = relevance[S]
    dataset.X_train = Feature_train[:,list(S)[0]]
    dataset.X_test = Feature_test[:,list(S)[0]]
    

