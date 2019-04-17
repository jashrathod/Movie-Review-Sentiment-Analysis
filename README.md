# Movie Review Sentiment Analysis

This project aims at providing the User with the ‘Sentiment’ of a review of a movie. In simple words, it will be predicted whether the input text is a positive or negative comment. It uses Natural Language Processing (NLP) techniques to provide the User with results as accurate as possible.

## Data Preparation

Various data preparation tasks involve:

1. Tokenization
2. Parts of Speech (POS) Tagging
3. Removing StopWords
4. Stemming 
5. Lemmatization

Out of these, Tokenization and POS Tagging were used so as to obtain better results.

## Building the Models

In this project, a set of eight models were trained. The training dataset was in the form of .txt files, one for each, positive and negative reviews. The accuracy of each model was about 70%. To obtain better results, a voting system was established which is used to consider outputs of multiple models as input and output the mode of these models as the final prediction. To avoid training of models each time, these models were pickled and loaded as and when needed.

## Models

1. Original Naive Bayes Classifier
2. Multinomial Naive Bayes (Sklearn Classifier)
3. Binomial Naive Bayes (Sklearn Classifier)
4. Logistic Regression 
5. SGD Classifier
6. Support Vector Classifier (SVC)  (Very poor performance hence rejected)
7. Linear Support Vector Classifier
8. NuSVC
