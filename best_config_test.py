import os,re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import metrics
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Illegal use of Arguments: Best_configuration.py <Training_samples_location> <Testing_Samples_Location>")
        exit(1)

    test =  sys.argv[1]
    header_list = []
    labels = []
    i=0

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

    text_clf01 = joblib.load('Training_model.pkl')
    predicted01 = text_clf01.predict(header_test)
    print("Removed Stop Words + L2 penalization")
    print ("F1:",metrics.f1_score(test_labels, predicted01, average='macro'))
    print ("accuracy:", metrics.accuracy_score(test_labels, predicted01))
    print ("precision:",metrics.precision_score(test_labels, predicted01, average='macro'))
    print ("recall:",metrics.recall_score(test_labels, predicted01, average='macro'))