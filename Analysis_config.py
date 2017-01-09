import os,re

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
import time

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
    for root, dirs, files in os.walk(train):
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
    for root, dirs, files in os.walk(test):
        for name in files:
            fo = open(root +"/"+name, "r")
            content = fo.read().replace('\n', ' ')
            body = re.sub(r'^(.*) Lines: (\d)+ ', "", content)
            header_test.append(unicode(body,errors='ignore'))
            test_labels.append(i)
        i=i+1

    ''' applying stemmer snowball and below are the references'''
    # http://cswww.essex.ac.uk/poesio/Teach/807/Labs/Lab1/CE807%20Lab%201.pdf
    # http://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn

    #SVM baseline###
    start_time = time.time()
    text_clf0 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
             token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',
    )),])
    text_clf0 = text_clf0.fit(header_list, labels)
    predicted0 = text_clf0.predict(header_test)
    print("SVM Baseline")
    print ("F1:",metrics.f1_score(test_labels, predicted0, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted0))
    print ("precision:",metrics.precision_score(test_labels, predicted0, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted0, average='macro'))
    print("Tine in seconds %s" %(time.time()-start_time))
    text_clf01 = Pipeline([('vect', CountVectorizer(stop_words = 'english',ngram_range=(1, 1),
             token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',
    )),])
    text_clf01 = text_clf01.fit(header_list, labels)
    predicted01 = text_clf01.predict(header_test)
    print("Model 1")
    print("Removed Stop Words")
    print ("F1:",metrics.f1_score(test_labels, predicted01, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted01))
    print ("precision:",metrics.precision_score(test_labels, predicted01, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted01, average='macro'))

    english_stemmer = nltk.stem.SnowballStemmer('english')
    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer,self).build_analyzer()
            return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
    stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
    stemvectorizer = StemmedCountVectorizer(min_df=1)

    text_clf2 = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',)),])
    text_clf2 = text_clf2.fit(header_list, labels)
    predicted2 = text_clf2.predict(header_test)
    print("Model 2")
    print("porter stemmer + stop words + l2 penality + hinge loss + tfid transformer")
    print ("F1:",metrics.f1_score(test_labels, predicted2, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted2))
    print ("precision:",metrics.precision_score(test_labels, predicted2, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted2, average='macro'))


    # model 3
    text_clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l1',
    )),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 3")
    print("Porter Stemmer + Stop Words + L1 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))


    # model 4
    # porter stemmer only
    text_clf = Pipeline([('vect', stemvectorizer),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge')),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 4")
    print("Porter Stemmer  ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))

    # model 5
    # porter stemmer + stop words
    text_clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge')),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 5")
    print("Porter Stemmer + stop words ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))




    text_clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', SGDClassifier(loss='hinge', penalty='l1'))
    ])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 6")
    print ("Porter Stemmer + Stop Words + L1 based Feature Selection + L1 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))


    # With L1 Linearsvc + L2 penality + stop words + porter stemmer
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    text_clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', SGDClassifier(loss='hinge', penalty='l2'))
    ])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 7")
    print ("Porter Stemmer + Stop Words + L1 based Feature Selection + L2 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))



    #Porter Stemmer + Stop Words + Univariate Feature Selection + L2 Penalization
    # Univariate feature selection
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),
      ('feature_selection', SelectKBest(chi2, k=2)),
      ('classification', SGDClassifier(loss='hinge', penalty='l2'))
    ])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 8")
    print ("Porter Stemmer + Stop Words + Univariate Feature Selection + L2 Penalization")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))

    # Tree Based feature selection
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),
      ('feature_selection', SelectFromModel(ExtraTreesClassifier(), prefit=True)),
      ('classification', SGDClassifier(loss='hinge', penalty='l2'))
    ])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 9")
    print ("Porter Stemmer + Stop Words +Tree Based  Feature Selection + L2 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))


    text_clf = Pipeline([('vect', stemvectorizer),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l1',
    )),])
    text_clf = text_clf.fit(header_list, labels)
    predicted = text_clf.predict(header_test)
    print("Model 10")
    print("Porter Stemmer +  L1 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted))
    print ("precision:",metrics.precision_score(test_labels, predicted, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted, average='macro'))


    # With L2 Linearsvc + L2 penality + stop words + porter stemmer
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    text_clf11 = Pipeline([('vect', stem_vectorizer),('tfidf', TfidfTransformer()),
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', SGDClassifier(loss='hinge', penalty='l2'))
    ])
    text_clf11 = text_clf11.fit(header_list, labels)
    predicted11 = text_clf11.predict(header_test)
    print("Model 11")
    print ("Porter Stemmer + Stop Words + L2 based Feature Selection + L2 Penalization ")
    print ("F1:",metrics.f1_score(test_labels, predicted11, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted11))
    print ("precision:",metrics.precision_score(test_labels, predicted11, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted11, average='macro'))

