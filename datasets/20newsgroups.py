from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

newsgroups_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers', 'quotes'])


def data():
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

