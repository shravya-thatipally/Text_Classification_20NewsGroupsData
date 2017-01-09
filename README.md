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
Applied all the classifiers for these configurations and selected best configuration from that and applied some
