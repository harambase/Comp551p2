from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import os


def read_data(path):
    files = os.listdir(path)
    data = []
    files.sort()
    i = 0
    for file in files:
        i = i + 1
        if not os.path.isdir(file):
            with open(path + '/' + file, 'r', encoding='UTF-8') as f:
                data.append(f.read())
    return data


def get_dataset():
    path_train_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/pos'
    path_train_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/neg'
    path_test_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/pos'
    path_test_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/neg'

    test_pos_data = read_data(path_test_pos)
    test_neg_data = read_data(path_test_neg)

    train_data = train_label = test_label = []

    train_pos_data = read_data(path_train_pos)
    train_neg_data = read_data(path_train_neg)

    stemmer = PorterStemmer()

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

    stop_words = list(stopwords.words('english'))

    for i in range(len(train_data)):
        train_data[i] = re.sub(r'[^A-z ]', '', train_data[i])
        word_list = word_tokenize(train_data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        train_data[i] = ''
        for j in range(len(word_list)):
            if word_list[j] not in stop_words:
                if len(word_list[j]) < 20:
                    train_data[i] = train_data[i] + word_list[j] + ' '

    for i in range(len(test_data)):
        test_data[i] = re.sub(r'[^A-z ]', '', test_data[i])
        word_list = word_tokenize(test_data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        test_data[i] = ''
        for j in range(len(word_list)):
            if word_list[j] not in stop_words:
                if len(word_list[j]) < 20:
                    test_data[i] = test_data[i] + word_list[j] + ' '

    return train_data, test_data, train_label, test_label

