from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import numpy as np

from time import time


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


def get_dataset():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers', 'quotes'])
    X = []
    y = []
    stemmer = PorterStemmer()
    stop_words = list(stopwords.words('english'))
    for i in range(len(newsgroups_train.data)):
        word_list = word_tokenize(newsgroups_train.data[i])
        stop_words = list(stopwords.words('english'))
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        x = ''
        for j in range(len(word_list)):
            if word_list[j] not in stop_words:
                x = x + word_list[j] + ' '
        X.append(x)
        y.append(newsgroups_train.target[i])

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(min_df=5)
    vectors_train = vectorizer.fit_transform(X_train)
    vectors_validation = vectorizer.transform(X_validation)

    return vectors_train, vectors_validation, y_train, y_validation


def svm(dataset):
    clf = LinearSVC(random_state=0, tol=1e-5)

    # specify parameters and distributions to sample from
    param_dist = {'dual': [True, False],
                  'penalty': ['l1', 'l2'],
                  'loss': ['hinge', 'squared_hinge']}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(dataset[0], dataset[2])
    print(random_search.score(dataset[1], dataset[3]))
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


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
