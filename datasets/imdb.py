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

path_train_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/pos'
path_train_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/neg'
path_test_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/pos'
path_test_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/neg'

# path_train_pos = '/Users/tianchima/Desktop/aclImdb/train/pos' 
# path_train_neg = '/Users/tianchima/Desktop/aclImdb/train/neg' 
# path_test_pos = '/Users/tianchima/Desktop/aclImdb/test/pos' 
# path_test_neg = '/Users/tianchima/Desktop/aclImdb/test/neg' 


def read_data(path):
    files = os.listdir(path)
    data = []
    files.sort()
    i = 0
    for file in files:
        #if i > 100:
         #   break
        i = i+1
        if not os.path.isdir(file):
            with open(path + '/' + file, 'r', encoding='UTF-8') as f:
                data.append(f.read())
    return data


def get_dataset():
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

    return train_data, test_data, train_label, test_label


def decision_tree(dataset):
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df = 5)), ('clf', DecisionTreeClassifier(random_state=0))])

    param_dist = {'clf__criterion': ['gini', 'entropy'],
                  'clf__splitter': ['best', 'random'],
                  'clf__max_features': ['auto', 'sqrt', 'log2']}

    n_iter_search = 100
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
    print('decision_tree_accruracy = ',accruracy)

    return


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


if __name__ == '__main__':

    dataset = get_dataset()

    svm(dataset)
    #decision_tree(dataset)