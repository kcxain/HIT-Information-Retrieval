# coding: utf-8

import math

import numpy as np

from utils import tokenize


class BM25:
    def __init__(self, idx, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.index = idx
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(self.corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        nd = self._initialize()
        self._calc_idf(nd)

    def _initialize(self):
        nd = {}
        num_doc = 0
        for document in self.corpus:
            document = ''.join(document)
            document = tokenize(document, remove_stop=False)  # 分词后的列表
            #             document = document.split(' ')
            self.doc_len.append(len(document))  # 每篇doc的词的数量
            num_doc += len(document)  # 全部词的总数

            freq = {}
            for word in document:
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1
            self.doc_freqs.append(freq)

            for word, f in freq.items():
                nd[word] = nd.get(word, 0) + 1  # 拉普拉斯平滑

        self.avgdl = num_doc / self.corpus_size  # 文档平均长度
        return nd

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def cal_scores(self, query):  # 利用全0向量直接计算全部索引文档
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query.split(' '):  # 分词处理好的query
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return scores

    def get_topk(self, query, k=3):
        scores = self.cal_scores(query)
        topk = np.argsort(scores)[::-1][:k]  # 返回对应的索引
        return [self.index[i] for i in topk]
