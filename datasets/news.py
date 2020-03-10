from sklearn.datasets import fetch_20newsgroups
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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform


def get_dataset():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers', 'quotes'])
    X = []
    y = []
    stemmer = PorterStemmer()
    stop_words = list(stopwords.words('english'))
    i = 0
    for i in range(len(newsgroups_train.data)):
        #if i>100:
          #  break
        i = i+1
        #newsgroups_train.data[i] = re.sub(r'[^A-z ]', '', newsgroups_train.data[i])
        word_list = word_tokenize(newsgroups_train.data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        x = ''
        for j in range(len(word_list)):
            if word_list[j] not in stop_words:
                if len(word_list[j])<20:
                    x = x + word_list[j] + ' '
        X.append(x)
        y.append(newsgroups_train.target[i])

    newsgroups_test = fetch_20newsgroups(subset='test', remove=['headers', 'footers', 'quotes'])
    X_test = []
    y_test = []
    for i in range(len(newsgroups_test.data)):
        #newsgroups_test.data[i] = re.sub(r'[^A-z ]', '', newsgroups_test.data[i])
        word_list_test = word_tokenize(newsgroups_test.data[i])
        for j in range(len(word_list_test)):
            word_list_test[j] = (stemmer.stem(word_list_test[j]))
        x = ''
        for j in range(len(word_list_test)):
            if word_list_test[j] not in stop_words:
                if len(word_list_test[j])<20:
                    x = x + word_list_test[j] + ' '
        X_test.append(x)
        y_test.append(newsgroups_test.target[i])

    return X, X_test, y, y_test


def svm(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df = 5)), ('clf', LinearSVC(random_state=0))])

    param_dist = {'clf__dual': [True, False],
                  'clf__loss': ['hinge', 'squared_hinge'],
                  'clf__C': np.power(10, np.arange(-2, 2, dtype=float)),
                  'clf__tol': np.power(10, np.arange(-10, -3, dtype=float)),
                  'clf__fit_intercept': [True, False]
                  }

    n_iter_search = 20
    random_search = GridSearchCV(vect_and_clf, param_grid=param_dist, cv=5, n_jobs = 4)

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    for candidate in candidates:
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
        print ("Parameters: {0}".format(results['params'][candidate]))
    test_label_predict = random_search.best_estimator_.predict(test_data)
    accruracy = accuracy_score(test_label, test_label_predict)
    print(results)
    print('svm_accruracy = ',accruracy)

    return


    # # use a full grid over all parameters
    # param_grid = {'average': [True, False],
    #               'l1_ratio': np.linspace(0, 1, num=10),
    #               'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
    #
    # # run grid search
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # start = time()
    # grid_search.fit(X, y)
    #
    # print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #       % (time() - start, len(grid_search.cv_results_['params'])))
    # report(grid_search.cv_results_)


if __name__ == '__main__':
    dataset = get_dataset()
    svm(dataset)
