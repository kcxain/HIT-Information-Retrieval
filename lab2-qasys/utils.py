import json


def get_stopwords(stop_words_file):
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
