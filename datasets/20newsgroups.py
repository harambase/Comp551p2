from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.ensemble import BaggingClassifier


newsgroups_train = fetch_20newsgroups(subset='train',remove=['headers', 'footers', 'quotes'])

X=[]
y=[]
stemmer = PorterStemmer()
for i in range(len(newsgroups_train.data)):
	word_list = word_tokenize(newsgroups_train.data[i])
	for j in range(len(word_list)):
		word_list[j] = (stemmer.stem(word_list[j]))
	x=''
	for j in range(len(word_list)):
		if len(word_list[j])<15:
			x=x+word_list[j]+' '
	X.append(x)
	y.append(newsgroups_train.target[i])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words = list(stopwords.words('english')), min_df = 5)
vectors_train = vectorizer.fit_transform(X_train)
vectors_validation=vectorizer.transform(X_validation)
