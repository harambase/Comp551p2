import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from time import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import imdb
import newsgroup


def report(results, n_top=3, file=None):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            if file != None:
                file.write("Model with rank: {0}".format(i))
                file.write("Mean validation score: {0:.3f} (std: {1:.3f})"
                           .format(results['mean_test_score'][candidate],
                                   results['std_test_score'][candidate]))
                file.write("Parameters: {0}".format(results['params'][candidate]))
                file.write("")


def svm():
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df=5)), ('clf', LinearSVC(random_state=0, verbose=10))])

    param_dist = {'clf__dual': [True, False],
                  'clf__loss': ['hinge', 'squared_hinge'],
                  'clf__C': loguniform(1e-3, 1e3),
                  'clf__tol': loguniform(1e-11, 1e-4),
                  'clf__fit_intercept': [True, False]}

    return vect_and_clf, param_dist


def lr():
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df=5)), ('clf', LogisticRegression(random_state=0))])

    param_dist = {'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
                  'clf__dual': [True, False],
                  'clf__C': loguniform(1e-3, 1e3),
                  'clf__tol': loguniform(1e-11, 1e-4),
                  'clf__fit_intercept': [True, False],
                  'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'clf__max_iter': stats.randint(50, 200),
                  'clf__warm_start': [True, False],
                  'clf__multi_class': ['auto', 'ovr', 'multinomial'],
                  'clf__l1_ratio': stats.uniform(0, 1)}

    return vect_and_clf, param_dist


def dt():
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df=5)), ('clf', DecisionTreeClassifier(random_state=0))])

    param_dist = {'clf__criterion': ['gini', 'entropy'],
                  'clf__splitter': ['best', 'random'],
                  'clf__max_depth': np.arange(2, 50, dtype=int).tolist() + [None],
                  'clf__min_samples_split': loguniform(1e-4, 1e-1),
                  'clf__min_samples_leaf': stats.randint(1, 200),
                  'clf__max_features': np.linspace(0.01, 1, num=10, dtype=float).tolist() + ['auto', 'sqrt', 'log2', None],
                  'clf__max_leaf_nodes': [None] + np.array(np.power(10, np.arange(1, 4, step=0.5)), dtype=int).tolist(),
                  'clf__min_impurity_decrease': [0.0] + np.array(np.power(10, np.arange(-10, -4, step=0.5)), dtype=float).tolist()}

    return vect_and_clf, param_dist


def ada():
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df=5)), ('clf', AdaBoostClassifier(random_state=0))])

    param_dist = {'clf__learning_rate': stats.uniform(0, 1),
                  'clf__n_estimators': stats.randint(10, 400),
                  'clf__algorithm': ['SAMME', 'SAMME.R'],
                  'clf__base_estimator': [None, LinearSVC(), LogisticRegression(), MultinomialNB()]}

    return vect_and_clf, param_dist


def forest():
    vect_and_clf = Pipeline([('vect', TfidfVectorizer(min_df=5)), ('clf', RandomForestClassifier(random_state=0))])

    param_dist = {'clf__n_estimators': np.array(np.power(10, np.arange(1, 3, step=0.1)), dtype=int),
                  'clf__criterion': ['gini', 'entropy'],
                  'clf__max_depth': np.arange(2, 50, dtype=int).tolist() + [None],
                  'clf__min_samples_split': loguniform(1e-4, 1e-1),
                  'clf__min_samples_leaf': stats.randint(1, 200),
                  'clf__max_features': np.linspace(0.01, 1, num=10, dtype=float).tolist() + ['auto', 'sqrt', 'log2', None],
                  'clf__max_leaf_nodes': [None] + np.array(np.power(10, np.arange(1, 4, step=0.5)), dtype=int).tolist(),
                  'clf__min_impurity_decrease': [0.0] + np.array(np.power(10, np.arange(-10, -4, step=0.5)), dtype=float).tolist(),
                  'clf__bootstrap': [True, False],
                  'clf__oob_score': [True, False],
                  'clf__warm_start': [True, False],
                  'clf__max_samples': stats.uniform(0, 1),
                  'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
                  'clf__verbose': [True, False],
                  'clf__min_weight_fraction_leaf': loguniform(1e-4, 1e-1)}

    return vect_and_clf, param_dist


def run(model, vect_and_clf, param_dist, n_iter_search, dataset):
    f = open(model + str(time()) + ".txt", "w+")

    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, verbose=10)

    train_data, test_data, train_label, test_label = dataset
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    f.write("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
    test_label_predict = random_search.best_estimator_.predict(test_data)
    print('accruracy = ', accuracy_score(test_label, test_label_predict))
    f.write('accruracy = %f' % accuracy_score(test_label, test_label_predict))
    f.close()

if __name__ == '__main__':
    # print("read IMDB")
    # imdb_data = imdb.get_dataset()
    print("read NewsGroup")
    news_data = newsgroup.get_dataset()
    n_iter_search = 20

    # print("init SVM")
    # vect_and_clf, param_dist = svm()
    # print("run SVM on news")
    # run('SVM', vect_and_clf, param_dist, n_iter_search, news_data)
    # print("run SVM on imdb")
    # run('SVM', vect_and_clf, param_dist, n_iter_search, imdb_data)

    print("init LR")
    vect_and_clf, param_dist = lr()
    print("run lr on news")
    run('lr', vect_and_clf, param_dist, n_iter_search, news_data)
    print("run SVM on imdb")
    # run('lr', vect_and_clf, param_dist, n_iter_search, imdb_data)

    print("init dt")
    vect_and_clf, param_dist = dt()
    print("run dt on news")
    run('dt', vect_and_clf, param_dist, n_iter_search, news_data)
    print("run dt on imdb")
    # run('dt', vect_and_clf, param_dist, n_iter_search, imdb_data)

    print("init ada")
    vect_and_clf, param_dist = ada()
    print("run ada on news")
    run('ada', vect_and_clf, param_dist, n_iter_search, news_data)
    print("run ada on imdb")
    # run('ada', vect_and_clf, param_dist, n_iter_search, imdb_data)
    #
    print("init forest")
    vect_and_clf, param_dist = forest()
    print("run forest on news")
    run('forest', vect_and_clf, param_dist, n_iter_search, news_data)
    print("run forest on imdb")
    # run('forest', vect_and_clf, param_dist, n_iter_search, imdb_data)







