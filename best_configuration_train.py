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
    train = sys.argv[1]
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

    text_clf01 = Pipeline([('vect', CountVectorizer(stop_words = 'english',ngram_range=(1, 1),
             token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf',
    SGDClassifier(loss='hinge', penalty='l2',)),])
    text_clf01 = text_clf01.fit(header_list, labels)
    joblib.dump(text_clf01, 'Training_model.pkl')
    print("Removed Stop Words + L2 penalization")
    print ("Tranining model is saved")

