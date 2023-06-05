# coding: utf-8

import json

import jieba
from ltp import LTP


class Trie:
    def __init__(self):
        self.root = {}  # 用字典存储
        self.end_of_word = '#'  # 用#标志一个单词的结束

    def insert(self, word: str):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word

    # 查找一个单词是否完整的存在于字典树里，判断node是否等于#
    def search(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node


def get_stop_dic(file_path):
    stop_dic = Trie()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_dic.insert(line.strip())
    return stop_dic


stop_dic = get_stop_dic('../lab1-webcraw/stopwords.txt')


def save_json(pages, save_path):
    with open(save_path, 'w', encoding='utf-8-sig') as f:
        for data in pages:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')


def read_json(file_path):
    pages_dic = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            p = json.loads(line)
            pages_dic.append(p)
    return pages_dic


ltp = LTP()


def tokenize_ltp(content: str, remove_stop=True):
    seg, _ = ltp.seg([content])
    words = []
    for w in seg[0]:
        if remove_stop and stop_dic.search(w):
            continue
        words.append(w)
    return words  # 返回分词结果列表


def tokenize(content: str, remove_stop=True):
    words = []
    for w in jieba.cut(content):
        if remove_stop and stop_dic.search(w):
            continue
        words.append(w)
    return words  # 返回分词结果列表


def get_pos(sent):
    seg, hidden = ltp.seg([sent])
    pos = ltp.pos(hidden)[0]
    return seg[0], pos


def get_ner(sent):
    seg, hidden = ltp.seg([sent])
    ner = ltp.ner(hidden)
    entis = {'Nh': [], 'Ni': [], 'Ns': []}
    for nn in ner[0]:
        tag, start, end = nn
        enti = ''.join(seg[0][start:end + 1])
        entis[tag].append(enti)
    return entis
