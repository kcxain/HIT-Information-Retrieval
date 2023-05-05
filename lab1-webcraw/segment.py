import os
import json
import re
from ltp import LTP
from tqdm import tqdm


def get_stopwords(stop_words_file):
    stop_words = []
    for line in open(stop_words_file, "r", encoding="utf-8"):
        stop_words.append(line.strip())
    return stop_words


class Segment:
    def __init__(self, data_file, target_file, stop_words_file, model):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.target_file = target_file
        self.stop_words = get_stopwords(stop_words_file)
        self.model = model

    def seg_page(self, page):
        model = self.model
        title = page['title']
        contents = page['paragraphs']
        segmented_title = self.remove_stop_words(model.pipeline(title, tasks=["cws"], return_dict=False))
        segmented_parapraghs = self.remove_stop_words(model.pipeline(contents, tasks=["cws"], return_dict=False))
        new_page = {
            "url": page["url"],
            "segmented_title": segmented_title[0],
            "segmented_parapraghs": segmented_parapraghs[0],
            "file_name": page["file_name"]
        }
        # print(new_page)
        return new_page

    def seg_all(self):
        pages = []
        for page in tqdm(self.data):
            pages.append(self.seg_page(page))

        with open(self.target_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False)

    def remove_stop_words(self, words_list):
        ret = []
        for text_word in words_list:
            if text_word not in self.stop_words:
                ret.append(text_word)
        return ret


def main():
    ltp = LTP()
    Seg = Segment(data_file='./data.json', target_file='./data_seg.json', stop_words_file='./stopwords.txt', model=ltp)
    Seg.seg_all()


if __name__ == "__main__":
    main()
