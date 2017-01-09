import random, numpy as np

import sys
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import os,re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import metrics
from sklearn.pipeline import Pipeline
from nltk.stem import *
from nltk.stem.porter import *
from sklearn.svm import LinearSVC
import nltk.stem
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("Illegal use of Arguments: Best_configuration.py <Training_samples_location> <Testing_Samples_Location>")
        exit(1)
    train = sys.argv[1]
    test =  sys.argv[2]
    ''' Extracting the training samples '''
    header_list = []
    labels = []
    i=0
    for root, dirs, files in os.walk('C:/Users/sthatipally/Downloads/Training'):
        for name in files:
            fo = open(root +"/"+name, "r")
            content = fo.read().replace('\n', ' ')
            body = re.sub(r'^(.*) Lines: (\d)+ ', "", content)
            header_list.append(unicode(body,errors='ignore'))
            labels.append(i)
        i=i+1

    ''' Extracting the testing samples '''
    header_test = []
    test_labels = []
    i = 0
    for root, dirs, files in os.walk('C:/Users/sthatipally/Downloads/Test'):
        for name in files:
            fo = open(root +"/"+name, "r")
            content = fo.read().replace('\n', ' ')
            body = re.sub(r'^(.*) Lines: (\d)+ ', "", content)
            header_test.append(unicode(body,errors='ignore'))
            test_labels.append(i)
        i=i+1

    def shuffle(train, test, size):
        shuffled_train = []
        shuffled_train_labels = []
        index =[]
        for i in range(len(labels)):
            index.append(i)
        random.shuffle(index)
        for i in index:
            shuffled_train.append(header_list[i])
            shuffled_train_labels.append((labels[i]))
        return shuffled_train[:size],shuffled_train_labels[:size]
    subsets = [(i+1)*100 for i in range(20)]

    from sklearn.naive_bayes import MultinomialNB
    def find_scores(estimator):
        sizes = []
        scores = []
        for i in subsets:
            text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', estimator),])
            train, labels_sub = shuffle(header_list, labels, i)
            text_clf = text_clf.fit(train, labels_sub)
            predicted = text_clf.predict(header_test)
            sizes.append(i)
            scores.append(metrics.f1_score(test_labels, predicted, average='macro'))
        return (sizes,scores)

    sizes_NB,scores_NB = find_scores(MultinomialNB())
    print(sizes_NB,scores_NB)
    sizes_svm,scores_svm = find_scores(SGDClassifier(loss='hinge', penalty='l2',
    ))
    print(sizes_svm,scores_svm)
    sizes_log,scores_log = find_scores(linear_model.LogisticRegression())
    print(sizes_log,scores_log)
    sizes_RF,scores_RF = find_scores(RandomForestClassifier(n_estimators=100))
    print(sizes_RF,scores_RF)
    import numpy as np


    plt.figure()
    plt.title("learning curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(sizes_NB, scores_NB, 'o-', color="r",
             label="Bayes score")
    plt.plot(sizes_log, scores_log, 'o-', color="g",
             label="Logistic Score")
    plt.plot(sizes_svm, scores_svm, 'o-', color="y",
             label="SVM score")
    plt.plot(sizes_RF, scores_RF, 'o-', color="b",
             label="Random Forest score")
    plt.legend(loc="best")
    plt.show()