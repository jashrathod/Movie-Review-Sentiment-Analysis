import os
import pickle
import random
import warnings
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC

warnings.filterwarnings("ignore")


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("short_reviews/positive.txt", "r", encoding="ISO-8859-1").read()
short_neg = open("short_reviews/negative.txt", "r", encoding="ISO-8859-1").read()

all_words = []
documents = []
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)


####### TRAINING AND TESTING #######


# original_classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Original Naive Bayes Accuracy:", (nltk.classify.accuracy(original_classifier, testing_set)) * 100)
#
# mnb = SklearnClassifier(MultinomialNB())
# mnb.train(training_set)
# print("MNB Accuracy: ", (nltk.classify.accuracy(mnb, testing_set)) * 100)
#
# bnb = SklearnClassifier(BernoulliNB())
# bnb.train(training_set)
# print("BNB Accuracy: ", (nltk.classify.accuracy(bnb, testing_set)) * 100)
#
# log = SklearnClassifier(LogisticRegression())
# log.train(training_set)
# print("Logistic Regression Accuracy: ", (nltk.classify.accuracy(log, testing_set)) * 100)
#
# sgd = SklearnClassifier(SGDClassifier())
# sgd.train(training_set)
# print("SGD Accuracy: ", (nltk.classify.accuracy(sgd, testing_set)) * 100)
#
# svc = SklearnClassifier(SVC())
# svc.train(training_set)
# print("SVC Accuracy: ", (nltk.classify.accuracy(svc, testing_set)) * 100)
#
# lsvc = SklearnClassifier(LinearSVC())
# lsvc.train(training_set)
# print("LinearSVC Accuracy: ", (nltk.classify.accuracy(lsvc, testing_set)) * 100)
#
# nsvc = SklearnClassifier(NuSVC())
# nsvc.train(training_set)
# print("NuSVC Accuracy: ", (nltk.classify.accuracy(nsvc, testing_set)) * 100)


####################################


####### PICKLING THE MODELS #######


# filename = 'ONB_Classifier.pickle'
# pickle.dump(original_classifier, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'MNB_Classifier.pickle'
# pickle.dump(mnb, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'BNB_Classifier.pickle'
# pickle.dump(bnb, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'Logistic_Regression.pickle'
# pickle.dump(log, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'SGD_Classifier.pickle'
# pickle.dump(sgd, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'SVC.pickle'
# pickle.dump(svc, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'Linear_SVC.pickle'
# pickle.dump(lsvc, open(os.path.join('pickled_models/', filename), 'wb'))
#
# filename = 'NuSVC.pickle'
# pickle.dump(nsvc, open(os.path.join('pickled_models/', filename), 'wb'))


###################################


####### LOADING THE PICKLED MODELS #######


filename = open('pickled_models//ONB_Classifier.pickle', 'rb')
original_classifier = pickle.load(filename)

filename = open('pickled_models//MNB_Classifier.pickle', 'rb')
mnb = pickle.load(filename)

filename = open('pickled_models//BNB_Classifier.pickle', 'rb')
bnb = pickle.load(filename)

filename = open('pickled_models//Logistic_Regression.pickle', 'rb')
log = pickle.load(filename)

filename = open('pickled_models//SGD_Classifier.pickle', 'rb')
sgd = pickle.load(filename)

filename = open('pickled_models//SVC.pickle', 'rb')
svc = pickle.load(filename)

filename = open('pickled_models//Linear_SVC.pickle', 'rb')
lsvc = pickle.load(filename)

filename = open('pickled_models//NuSVC.pickle', 'rb')
nsvc = pickle.load(filename)


##########################################


voted_classifier = VoteClassifier(original_classifier, mnb, bnb, log, sgd, lsvc, nsvc)


def sentiment(text):
    feats = find_features(text)
    pred_sentiment = voted_classifier.classify(feats)
    pred_confidence = voted_classifier.confidence(feats)
    if pred_sentiment == 'pos':
        pred_sentiment = 'Positive'
    else:
        pred_sentiment = 'Negative'
    print("\n", text, "\nSENTIMENT:", pred_sentiment, "\tCONFIDENCE:", pred_confidence)