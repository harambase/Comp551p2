import nltk
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re
import os
from sklearn.pipeline import Pipeline


path_train_pos = '/Users/tianchima/Desktop/aclImdb/train/pos' 
path_train_neg = '/Users/tianchima/Desktop/aclImdb/train/neg' 
path_test_pos = '/Users/tianchima/Desktop/aclImdb/test/pos' 
path_test_neg = '/Users/tianchima/Desktop/aclImdb/test/neg' 
def read_data(path):
	files= os.listdir(path)
	data = []
	files.sort()
	for file in files: 
		if not os.path.isdir(file): 
			with open(path+'/'+file, 'r') as f:
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
 	train_data[i] = re.sub(r'[^A-z ]','',train_data[i])
 	word_list = word_tokenize(train_data[i])
 	for j in range(len(word_list)):
 		word_list[j] = (stemmer.stem(word_list[j]))
 	train_data[i] = ''
 	for j in range(len(word_list)):
 		if word_list[j] not in stop_words:
 			train_data[i] = train_data[i] + word_list[j]+' '

for i in range(len(test_data)):
 	test_data[i] = re.sub(r'[^A-z ]','',test_data[i])
 	word_list = word_tokenize(test_data[i])
 	for j in range(len(word_list)):
 		word_list[j] = (stemmer.stem(word_list[j]))
 	test_data[i] = ''
 	for j in range(len(word_list)):
 		if word_list[j] not in stop_words:
 			test_data[i] = test_data[i] + word_list[j]+' '
