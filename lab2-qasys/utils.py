import json
from ltp import LTP

model = LTP().to("cuda:0")


def pos_tag(words):
    return model.pipeline(words, tasks=["pos"]).pos


def seg_sentence(sentence):
    stopwords = get_stopwords()
    return remove_stop_words(model.pipeline(sentence, tasks=["cws"], return_dict=False), stopwords)[0]


def get_stopwords(stop_words_file='./stopwords.txt'):
    stop_words = []
    for line in open(stop_words_file, "r", encoding="utf-8"):
        stop_words.append(line.strip())
    return stop_words


def remove_stop_words(words_list, stop_words):
    ret = []
    for text_word in words_list:
        if text_word not in stop_words:
            ret.append(text_word)
    return ret


def read_jsonlist(file_path):
    l = []
    with open(file_path, 'r', encoding="utf8") as f:
        jsonlines = f.readlines()
    for line in jsonlines:
        json_line = json.loads(line)
        l.append(json_line)
    return l


def write_json(json_path, obj):
    with open(json_path, 'w', encoding="utf8") as f:
        for line in obj:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
