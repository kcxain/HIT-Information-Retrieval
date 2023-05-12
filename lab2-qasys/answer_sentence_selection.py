import json
import os

import Levenshtein
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from preprocessed import read_jsonlist
from tqdm import tqdm
from scipy.linalg import norm
from numpy import dot
from utils import seg_sentence, pos_tag, write_json, ner


def dataloader(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        train_questions = []
        train_answers = []
        lines = f.readlines()
        train_tmps = []
        for line in lines:
            train_tmps.append(json.loads(line))
        train_tmps = sorted(train_tmps, key=lambda x: x["qid"])
        for train_tmp in train_tmps:
            train_questions.append(
                {
                    "qid": train_tmp.pop("qid"),
                    "question": train_tmp.pop("question"),
                }
            )
            train_answers.append(train_tmp)
    return train_questions, train_answers


def cal_lcs_len(str_a, str_b):
    lena = len(str_a)
    lenb = len(str_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if str_a[i] == str_b[j]:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    return c[lena][lenb]


def build_features(q_words, ans_words, tf_idf_vec):
    feature_list = []
    tags = {'a', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v'}
    query = ''.join(q_words)
    answer = ''.join(ans_words)
    vectors = tf_idf_vec.transform([' '.join(q_words), ' '.join(ans_words)]).toarray()
    norm_val = (norm(vectors[0]) * norm(vectors[1]))
    # answer句词数
    feature_list.append('1:%d' % len(ans_words))
    # len(q_words) - len(ans_words)
    feature_list.append('2:%d' % abs(len(q_words) - len(ans_words)))
    # co unigram
    feature_list.append('3:%f' % len(set(q_words).intersection(set(ans_words))))
    # LCS between query and answer
    feature_list.append('4:%d' % cal_lcs_len(query, ans_words))
    # 编辑距离
    feature_list.append('5:%d' % Levenshtein.distance(query, answer))
    # 相同命名实体的数量
    # feature_list.append('6:%d' % len(set(ner(q_words)) & set(ner(ans_words))))
    # tf-idf相似度
    feature_list.append('6:%f' % ((dot(vectors[0], vectors[1]) / norm_val) if norm_val else 0))
    # 句子中是否包含中文或英文冒号
    feature_list.append('7:%d' % (1 if ':' in ans_words or '：' in ans_words else 0))
    return feature_list


class AnswerRank:
    def __init__(self, train_data_path='./data/train.json',
                 train_feature_path='./data/train_feature.json',
                 dev_feature_path='./data/dev_feature.json',
                 test_data_path='./data/test_ans_cls.json',
                 test_feature_path='./data/test_feature.json',
                 model_path='./data/svm_rank.pkl',
                 dev_predict_path='./data/dev_predict.json',
                 test_predict_path='./data/test_predict.json',
                 test_ans_path='./data/test_ans.json'):
        self.train_data_path = train_data_path
        self.train_feature_path = train_feature_path
        self.dev_feature_path = dev_feature_path
        self.test_feature_path = test_feature_path
        self.model_path = model_path
        self.dev_predict_path = dev_predict_path
        self.test_predict_path = test_predict_path
        self.test_data_path = test_data_path
        self.test_ans_path = test_ans_path

    def load_train_data(self):
        if os.path.exists(self.train_feature_path) and os.path.exists(self.dev_feature_path):
            return
        seg_passages = read_jsonlist('./data/preprocessed_sentences.json')
        train_list = read_jsonlist(self.train_data_path)
        feature_list = []
        for item in tqdm(train_list):  # 遍历train.json文件中的每一行query信息
            qid, pid, q_words, ans_words_lst, features = item['qid'], item['pid'], seg_sentence(item['question']), \
                [seg_sentence(line) for line in item['answer_sentence']], []

            tf_idf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tf_idf_vec.fit_transform(' '.join(word_lst) for word_lst in seg_passages[pid]['seg_doc'])

            for word_lst in seg_passages[pid]['seg_doc']:
                value = 3 if word_lst in ans_words_lst else 0
                feature = ' '.join(build_features(q_words, word_lst, tf_idf_vec))
                features.append('%d qid:%d %s' % (value, qid, feature))
            feature_list.append(features)
        feature_list.sort(key=lambda lst: int(lst[0].split()[1].split(':')[1]))

        train_features, test_features = train_test_split(
            feature_list, test_size=0.10, shuffle=False, random_state=0
        )

        with open(self.train_feature_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([feature for feature_lst in train_features for feature in feature_lst]))
        with open(self.dev_feature_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([feature for feature_lst in test_features for feature in feature_lst]))
        return train_features, test_features

    def rank_svm(self):
        train_cmd = '.\svm_rank_learn.exe -c 10 %s %s' % (self.train_feature_path, self.model_path)
        predict_cmd = '.\svm_rank_classify.exe %s %s %s' % (
            self.dev_feature_path, self.model_path, self.dev_predict_path)
        os.system('%s && %s' % (train_cmd, predict_cmd))

    def evaluate(self):
        with open(self.dev_feature_path, 'r', encoding='utf-8') as f1, open(self.dev_predict_path, 'r',
                                                                            encoding='utf-8') as f2:
            y_true, y_predict, right = {}, {}, 0
            for line1, line2 in zip(f1, f2):
                if len(line1) == 1:
                    break
                qid = int(line1.split()[1].split(':')[1])
                lst1, lst2 = y_true.get(qid, []), y_predict.get(qid, [])
                lst1.append((int(line1[0]), len(lst1)))
                lst2.append((float(line2.strip()), len(lst2)))
                y_true[qid], y_predict[qid] = lst1, lst2

            for qid in y_true:
                lst1 = sorted(y_true[qid], key=lambda item: item[0], reverse=True)
                lst2 = sorted(y_predict[qid], key=lambda item: item[0], reverse=True)
                # print(lst1)
                # input(">>")
                if lst1[0][1] == lst2[0][1]:
                    right += 1
            return right, len(y_true)

    def load_test_data(self):
        if os.path.exists(self.test_feature_path):
            return
        seg_passages = read_jsonlist('./data/preprocessed_sentences.json')
        test_list = read_jsonlist(self.test_data_path)
        feature_list = []
        for item in tqdm(test_list):
            # print(item)
            # input('>>')
            qid, pid, q_words, features = item['qid'], item['pid'], item['question'], []
            tf_idf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tf_idf_vec.fit_transform(' '.join(word_lst) for word_lst in seg_passages[pid]['seg_doc'])

            for word_lst in seg_passages[pid]['seg_doc']:
                feature = ' '.join(build_features(q_words, word_lst, tf_idf_vec))
                features.append('0 qid:%d %s' % (qid, feature))
            feature_list.append(features)
        feature_list.sort(key=lambda lst: int(lst[0].split()[1].split(':')[1]))
        with open(self.test_feature_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([feature for features in feature_list for feature in features]))

    def train(self):
        if os.path.exists(self.model_path):
            return
        else:
            self.load_train_data()
            print("start train")
            train_cmd = '.\svm_rank_learn.exe -c 10 %s %s' % (self.train_feature_path, self.model_path)
            print("start predict")
            predict_cmd = '.\svm_rank_classify.exe %s %s %s' % (
                self.dev_feature_path, self.model_path, self.dev_predict_path)
            os.system('%s && %s' % (train_cmd, predict_cmd))
            right_predict, num = self.evaluate()
            print('exact match：{}；total num：{}；acc：{}%'.format(right_predict, num, (right_predict / num) * 100))

    def predict(self):
        self.load_test_data()
        os.system(
            '.\svm_rank_classify.exe %s %s %s' % (
                self.test_feature_path, self.model_path, self.test_predict_path))
        with open(self.test_feature_path, 'r', encoding='utf-8') as f1, open(self.test_predict_path, 'r',
                                                                             encoding='utf-8') as f2:
            labels = {}
            for line1, line2 in zip(f1, f2):
                if len(line1) == 1:
                    break
                qid = int(line1.split()[1].split(':')[1])
                if qid not in labels:
                    labels[qid] = []
                labels[qid].append((float(line2.strip()), len(labels[qid])))
            seg_passages, res_lst = read_jsonlist('data/preprocessed_sentences.json'), read_jsonlist(self.test_data_path)
            for item in res_lst:
                qid, pid, q_words = item['qid'], item['pid'], item['question']
                rank_lst, seg_passage = sorted(labels[qid], key=lambda val: val[0], reverse=True), seg_passages[pid]['seg_doc']
                print(rank_lst[0])
                item['answer_sentence'] = seg_passage[rank_lst[0][1]]
            write_json(self.test_ans_path, res_lst)


def main():
    ranker = AnswerRank()
    # ranker.train()
    ranker.predict()


if __name__ == '__main__':
    main()
