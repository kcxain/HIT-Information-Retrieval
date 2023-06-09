# coding:utf-8
import json
import os.path

from ltp import LTP
from tqdm import tqdm
from tqdm.contrib import tzip
from models import BM25
from utils import read_jsonlist, get_stopwords, write_json
from functools import reduce
import pickle


def get_docs():
    doc_path = './data/passages_multi_sentences.json'
    doc_saved_path = './data/preprocessed_sentences.json'
    stop_words_path = './stopwords.txt'
    dev_file = './data/train.json'
    ltp = LTP()
    s = DocSearch(doc_path, doc_saved_path, stop_words_path, ltp)
    return s.docs


class DocSearch:
    def __init__(self, doc_path, doc_saved_path, stop_words_path, seg_model):
        self.docs = read_jsonlist(doc_path)
        self.stop_words = get_stopwords(stop_words_path)
        self.doc_saved_path = doc_saved_path
        self.seg_model = seg_model.to("cuda:0")
        self.seg_docs()
        self.docs_list = []
        for doc in tqdm(self.docs):
            doc_array = [word for sentence in doc['seg_doc'] for word in sentence]
            self.docs_list.append(doc_array)
        self.search_model = BM25(self.docs_list)

        flag = True
        if os.path.exists('index/word2id.pickle'):
            with open('index/word2id.pickle', 'rb') as f:
                self.word2id = pickle.load(f)
        else:
            flag = False
        if os.path.exists('index/word2doc.pickle'):
            with open('index/word2doc.pickle', 'rb') as f:
                self.word2doc = pickle.load(f)
        else:
            flag = False
        if os.path.exists('index/doc_pos.pickle'):
            with open('index/doc_pos.pickle', 'rb') as f:
                self.doc_pos = pickle.load(f)
        else:
            flag = False

        if not flag:
            self.word2id = {}
            self.word2doc = []
            self.doc_pos = []
            self.index()

    def remove_stop_words(self, words_list):
        ret = []
        for text_word in words_list:
            if text_word not in self.stop_words:
                ret.append(text_word)
        return ret

    def save_docs(self):
        with open(self.doc_saved_path, 'w', encoding="utf8") as f:
            for line in self.docs:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    def read_docs(self):
        """
        从分好词的文档中读入
        :return: 是否读入成功
        """
        try:
            docs = read_jsonlist(self.doc_saved_path)
            if self.docs is not None and len(docs) == len(self.docs):
                self.docs = docs
                print(len(self.docs))
                return True
        except Exception:
            return False
        return False

    def seg_docs(self):
        if os.path.exists(self.doc_saved_path):
            if self.read_docs():
                return
        model = self.seg_model
        for pdoc in tqdm(self.docs):
            doc = pdoc["document"]
            seg_doc = self.remove_stop_words(model.pipeline(doc, tasks=["cws"], return_dict=False))
            pdoc["seg_doc"] = seg_doc[0]
        self.save_docs()

    def search_pid(self, query):
        if isinstance(query, str):
            model = self.seg_model
            query = self.remove_stop_words(model.pipeline(query, tasks=["cws"], return_dict=False))[0]
        pid_index, _ = self.search_model.doc_sort(query)
        return pid_index

    def evaluate(self, dev_file):
        dev_list = read_jsonlist(dev_file)
        num = len(dev_list)
        match = 0
        query_list = []
        label_list = []
        for dev in tqdm(dev_list):
            query_list.append(dev['question'])
            label_list.append(dev['pid'])
        seg_query_list = self.remove_stop_words(self.seg_model.pipeline(query_list, tasks=["cws"], return_dict=False))[
            0]
        for seg_query, label in tzip(seg_query_list, label_list):
            pid = self.search_pid(seg_query)[0]
            if pid == label:
                match += 1
            print(match)
        # acc: 0.8069880418535127
        print(f'acc: {match / num}')

    def predict(self, test_file):
        test_list = read_jsonlist(test_file)
        query_list = []
        for test in tqdm(test_list):
            query_list.append(test['question'])
        seg_query_list = self.remove_stop_words(self.seg_model.pipeline(query_list, tasks=["cws"], return_dict=False))[
            0]
        for seg_query, test in tzip(seg_query_list, test_list):
            pid = self.search_pid(seg_query)[0]
            test['pid'] = int(pid)
            test['seg_q'] = seg_query
        # print(test_list[0])
        write_json('./data/test_ans.json', test_list)

    def save_index(self):
        with open('index/word2id.pickle', 'wb') as f:
            pickle.dump(self.word2id, f)
        with open('index/word2doc.pickle', 'wb') as f:
            pickle.dump(self.word2doc, f)
        with open('index/doc_pos.pickle', 'wb') as f:
            pickle.dump(self.doc_pos, f)

    def index(self):
        words = set()
        for doc in self.docs_list:
            for word in doc:
                words.add(word)
        words = list(words)
        for i, word in enumerate(words):
            self.word2id[word] = i
            self.word2doc.append([])
            self.doc_pos.append([])

        for i, doc in enumerate(tqdm(self.docs_list)):
            for j, word in enumerate(doc):
                if i not in self.word2doc[self.word2id[word]]:
                    self.word2doc[self.word2id[word]].append(i)
                self.doc_pos[self.word2id[word]].append((i, j))
        self.save_index()

    def search(self, query):
        # print(self.word2doc)

        def intersect(x, y):
            return x | y

        def union(x, y):
            return x & y

        if '&&' in query:
            word_list = query.split('&&')
        elif '||' in query:
            word_list = query.split('||')
        else:
            word_list = [query]
        docs_set = []
        for word in word_list:
            doc_set = set(self.word2doc[self.word2id[word]])
            docs_set.append(doc_set)
        res = set()
        if '&&' in query:
            res = reduce(union, docs_set)
        elif '||' in query:
            res = reduce(intersect, docs_set)
        else:
            res = docs_set[0]
        return res


def main():
    doc_path = './data/passages_multi_sentences.json'
    doc_saved_path = './data/preprocessed_sentences.json'
    stop_words_path = './stopwords.txt'
    dev_file = './data/train.json'
    test_file = './data/test.json'
    ltp = LTP()
    s = DocSearch(doc_path, doc_saved_path, stop_words_path, ltp)
    # print(s.docs[0])
    # s.search_pid("腾讯")
    # s.predict(test_file)
    while 1:
        query = input("Query > ")
        if query == 'q':
            break
        print(s.search(query))


if __name__ == '__main__':
    main()
