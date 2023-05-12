# coding:utf-8
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import joblib
from sklearn import svm
from utils import read_jsonlist, get_stopwords, remove_stop_words
from ltp import LTP


def dataloader(file_path):
    stop_words = get_stopwords('./stopwords.txt')
    tmp_texts = []
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.strip().split('\t')
            tmp_texts.append(line_split[1])
            labels.append(line_split[0])
    model = LTP()
    # TfidfVectorizer 所需要的形式为在同一个字符串中，分词空格隔开
    tmp_texts = remove_stop_words(model.pipeline(tmp_texts, tasks=["cws"], return_dict=False), stop_words)[0]
    for text in tmp_texts:
        text = " ".join(text)
        texts.append(text)
    return texts, labels


class QuestionClassifier:
    def __init__(self, svm_model=None, tf_idf_model=None):
        # print(self.test_texts)
        # print(self.test_labels)
        if svm_model is not None and os.path.exists(svm_model):
            self.svm_model = joblib.load(svm_model)
        else:
            self.svm_model = svm.SVC(C=100.0, gamma=0.05)
        if tf_idf_model is not None and os.path.exists(tf_idf_model):
            self.tf_idf_model = joblib.load(tf_idf_model)
        else:
            self.tf_idf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

    def train(self, train_file):
        train_texts, train_labels = dataloader(train_file)
        # 调库实现求tf-idf矩阵,
        # token_pattern=r"(?u)\b\w+\b"
        print(train_texts)
        train_data = self.tf_idf_model.fit_transform(train_texts)
        print(train_data)
        self.svm_model.fit(train_data, np.asarray(train_labels))

        # save model
        joblib.dump(self.svm_model, 'saved_model/svm.pkl')
        joblib.dump(self.tf_idf_model, 'saved_model/tf_idf.pkl')

    def predict(self, test_file):
        test_texts, test_labels = dataloader(test_file)
        test_data = self.tf_idf_model.transform(test_texts)
        result = self.svm_model.predict(test_data)
        score = self.svm_model.score(test_data, test_labels)
        print(score)


def main():
    test_file = 'data/test_questions.txt'
    train_file = 'data/train_questions.txt'
    svm_model = 'saved_model/svm.pkl'
    tf_idf_model = 'saved_model/tf_idf.pkl'
    qc = QuestionClassifier(svm_model, tf_idf_model)
    # qc.train(train_file)
    # 0.7809885931558935
    qc.predict(test_file)


if __name__ == '__main__':
    main()
