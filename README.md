# Text_Classification_20NewsGroupsData

The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
Download link: http://qwone.com/~jason/20Newsgroups/

Selected 4 classes for this project.
1. rec.sport.hockey
2. sci.med
3. soc.religion.christian
4. talk.religion.misc

In each class, there two sets of documents,one for training and one for test. The format of each document is as follows:

1) Header - Consists of fields such as <From>, <Subject>, <Organization> and <Lines> fields. 
2) Body - The main body of the document.

Classifiers used - Naive Bayes, Logistic Regression, Support Vector Machines, and Random Forests.
 
Configurations used:
1. Unigram Baseline (UB) -- Basic sentence segmentation and tokenization. Use all words.
2. Bigram Baseline (BB) -- Use all bigrams. (e.g. I ran a race => {I ran, ran a, a race}. )
Applied all the classifiers for these configurations and selected best model from that and applied some more techniques namely:
1) Feature representations
2) Feature selection
3) Hyperparameters

My Best configuration is obtained by removing stop words with L2 penalization on SVM. I have used SGDclassifier with hinge loss ( a linear SVM, as the number of features is more than 10000 most likely points are linearly separable ) as the classifier with above mentioned.
I have created two python files one to build a model and another to test the model:

Execute the following command to get a learned model from a training dataset:

- python best_config_train.py Testing_Samples_Location.

Execute the following command to pre-learned model run on a test dataset:

python best_config_test.py Testing_Samples_Location

Exploration for best params in Unigram Configuration: 

python Analysis_config.py Training_samples_location Testing_Samples_Location

Code for Learning curves plot:
python Learning_curves.py

Baseline models:
python Unigram_bigram_models.py Training_samples_location Testing_Samples_Location


