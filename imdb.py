import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC

path_train_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/pos'
path_train_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/neg'
path_test_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/pos'
path_test_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/neg'


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def read_data(path):
    files = os.listdir(path)
    data = []
    files.sort()
    i = 0
    for file in files:
        if i > 100:
            break
        i = i + 1
        if not os.path.isdir(file):
            with open(path + '/' + file, 'r') as f:
                data.append(f.read())
    return data


test_pos_data = read_data(path_test_pos)
test_neg_data = read_data(path_test_neg)
train_data = []
train_label = []
test_label = []
train_pos_data = read_data(path_train_pos)
train_neg_data = read_data(path_train_neg)
for i in range(len(train_pos_data)):
    train_data.append(train_pos_data[i])
    train_label.append(1)
    train_data.append(train_neg_data[i])
    train_label.append(-1)
test_data = test_pos_data + test_neg_data
for i in range(len(test_pos_data)):
    test_label.append(1)
for i in range(len(test_neg_data)):
    test_label.append(-1)

stemmer = PorterStemmer()
stop_words = list(stopwords.words('english'))
for i in range(len(train_data)):
    train_data[i] = re.sub(r'[^A-z ]', '', train_data[i])
    word_list = word_tokenize(train_data[i])
    for j in range(len(word_list)):
        word_list[j] = (stemmer.stem(word_list[j]))
    train_data[i] = ''
    for j in range(len(word_list)):
        if word_list[j] not in stop_words:
            train_data[i] = train_data[i] + word_list[j] + ' '

for i in range(len(test_data)):
    test_data[i] = re.sub(r'[^A-z ]', '', test_data[i])
    word_list = word_tokenize(test_data[i])
    for j in range(len(word_list)):
        word_list[j] = (stemmer.stem(word_list[j]))
    test_data[i] = ''
    for j in range(len(word_list)):
        if word_list[j] not in stop_words:
            test_data[i] = test_data[i] + word_list[j] + ' '

vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LinearSVC(random_state=0, tol=1e-5))])

param_dist = {'clf__dual': [True, False],
              'clf__loss': ['hinge', 'squared_hinge']}

n_iter_search = 100
random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5)
start = time()
random_search.fit(train_data, train_label)
print(random_search.cv_results_)
print(random_search.best_params_)
print(random_search.best_estimator_['clf'].get_params())
report(random_search.cv_results_)
