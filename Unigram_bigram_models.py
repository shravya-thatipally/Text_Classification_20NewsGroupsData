import os,re,time

import sys
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
from sklearn.naive_bayes import MultinomialNB

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

    print ("UNIGRAM BASELINE")
    #### Naive bayes using pipeline #####
    from sklearn.pipeline import Pipeline
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Naive bayes")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))
    #SVM###
    from sklearn.linear_model import SGDClassifier
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',
    )),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("SVM")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))
    ## logistic regression ##
    from sklearn import linear_model
    start_time = time.time()
    logistic = linear_model.LogisticRegression()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),('tfidf', TfidfTransformer()),('logistic', logistic)])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print ("Logistic")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))

    from sklearn.ensemble import RandomForestClassifier
    start_time = time.time()
    Randomforest = RandomForestClassifier(n_estimators=100)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('Randomforest', Randomforest)])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print ("Random Forest")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))

    print("####### BIGRAM BASELINE #######")
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2),
             token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
    text_clf = text_clf.fit(header_list, labels)
    print ("Bigram Model-- Naive Bayes")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))

    from sklearn.linear_model import SGDClassifier
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',
    )),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print ("SVM")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))

    from sklearn import linear_model
    start_time = time.time()
    logistic = linear_model.LogisticRegression()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2),
             token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('logistic', logistic)])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print ("logistic")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))

    from sklearn.ensemble import RandomForestClassifier
    start_time = time.time()
    Randomforest = RandomForestClassifier(n_estimators=100)
    text_clf = Pipeline([('vect',CountVectorizer(ngram_range=(2, 2),
             token_pattern=r'\b\w+\b', min_df=1) ),('tfidf', TfidfTransformer()),('Randomforest', Randomforest)])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print ("Random Forest")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))