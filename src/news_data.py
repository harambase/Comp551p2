from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups


def get_dataset():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    X = y = X_test = y_test = []
    stemmer = PorterStemmer()
    stop_words = list(stopwords.words('english'))

    for i in range(len(newsgroups_train.data)):
        # newsgroups_train.data[i] = re.sub(r'[^A-z ]', '', newsgroups_train.data[i])
        word_list = word_tokenize(newsgroups_train.data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        x = ''
        for j in range(len(word_list)):
            if word_list[j] not in stop_words:
                if len(word_list[j]) < 20:
                    x = x + word_list[j] + ' '
        X.append(x)
        y.append(newsgroups_train.target[i])

    for i in range(len(newsgroups_test.data)):
        # newsgroups_test.data[i] = re.sub(r'[^A-z ]', '', newsgroups_test.data[i])
        word_list_test = word_tokenize(newsgroups_test.data[i])
        for j in range(len(word_list_test)):
            word_list_test[j] = (stemmer.stem(word_list_test[j]))
        x = ''
        for j in range(len(word_list_test)):
            if word_list_test[j] not in stop_words:
                if len(word_list_test[j]) < 20:
                    x = x + word_list_test[j] + ' '
        X_test.append(x)
        y_test.append(newsgroups_test.target[i])

    return X, X_test, y, y_test
