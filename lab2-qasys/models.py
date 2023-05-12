import math
import numpy as np


def cal_doc_tf(doc):
    """
    计算单个文档每个词的词频
    :param doc: 字符串列表
    :return: 字典，key为字符串，value为int
    """
    tf = {}
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    return tf


class BM25:

    def __init__(self, docs):
        # 文档总数
        self.N = len(docs)
        # 文档平均长度
        self.avg_dl = sum([len(doc) + 0.0 for doc in docs]) / self.N
        self.docs = docs

        # self.doc2id = {}
        # # 构造文档到索引的映射
        # for i, doc in enumerate(docs):
        #     self.doc2id[doc] = i

        # 每个词在每个文档中出现频数
        self.f = []
        # 词在文档中出现的文档数目
        self.df = {}
        # 倒置文档频率
        self.idf = {}
        self.k1 = 1.5
        self.k3 = 1.2
        self.b = 0.75
        self.cal_idf()

    def cal_idf(self):
        for doc in self.docs:
            doc_tf = cal_doc_tf(doc)
            self.f.append(doc_tf)
            for k in doc_tf.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.N + 1) - math.log(v + 1)

    def get_rsv(self, query, index):
        # query的tf
        query_tf = cal_doc_tf(query)
        rsv = 0
        for word in query:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            tf_td = self.f[index][word]
            tf_tq = query_tf[word]
            rsv += (self.idf[word] * tf_td * (self.k1 + 1) * tf_tq * (self.k3 + 1)
                    / (tf_td + self.k1 * (1 - self.b + self.b * d / self.avg_dl)) * (self.k3 + tf_tq))
        return rsv

    def doc_sort(self, query):
        # 返回降序排列的文档列表
        scores = []
        for i in range(len(self.docs)):
            rsv = self.get_rsv(query, i)
            scores.append(rsv)
        scores = np.array(scores)
        # 返回降序排列的文档索引
        top_index = np.argsort(-scores)
        return [self.docs[i] for i in top_index]


if __name__ == '__main__':
    docs = [['今天', '我', '去了', '北京大学', '。'],
            ['三体', '是', '一本', '小说'],
            ['我', '在', '北京大学', '上', '三体', '课程'],
            ['自然语言', '计算机科学', '领域', '人工智能', '领域', '中', '一个', '方向'],
            ['研究', '人', '计算机', '之间', '自然语言', '通信', '理论', '方法'],
            ['自然语言', '一门', '融', '语言学', '计算机科学', '数学', '一体', '科学'],
            ['这一', '领域', '研究', '涉及', '自然语言'],
            ['日常', '语言'],
            ['语言学', '研究'],
            ['区别'],
            ['自然语言', '研究', '自然语言'],
            ['在于', '研制', '自然语言', '通信', '计算机系统'],
            ['特别', '软件系统'],
            ['计算机科学', '一部分']]
    query = ['我', '计算机科学', '北京大学', '三体', '自然语言']
    model = BM25(docs)
    print(model.doc_sort(query))

