# coding: utf-8
import json
import os

import joblib
import pandas as pd
from harvesttext import HarvestText
from tqdm import trange
from win32com.client import Dispatch

from RetrieverModel import Retriever
from utils import read_json, save_json

word = Dispatch('Word.Application')  # 全局word应用程序
word.Visible = 0  # 后台运行,不显示
word.DisplayAlerts = 0  # 不警告

web_model_path = './model/web_model.pkl'
file_model_path = './model/file_model.pkl'
file_dt_path = './data/files_content.json'
web_data_path = '../lab1-webcraw/preprocessed.json'


def load_web_dt():
    with open(web_data_path, 'r', encoding='utf-8') as f:
        data_web = json.load(f)
    web_dt = pd.DataFrame(data_web[:1000], columns=['url', 'segmented_title', 'segmented_parapraghs', 'file_name'])
    web_dt['idx'] = [i for i in range(len(web_dt))]
    return data_web, web_dt


def process_doc(file_name):
    """
    只处理包含doc/docx的附件
    :param file_name: 文件
    :return:
    """
    # TODO 此处需替换为附件文件夹的全局路径
    attachments_dir = '*\\lab1-webcraw\\attachments\\'
    ht = HarvestText()
    if not (file_name.endswith('doc') or file_name.endswith('docx')):
        return '无doc/docx文档'
    pps = []
    print(f'{attachments_dir}{file_name}')
    try:
        doc = word.Documents.Open(FileName=f'{attachments_dir}{file_name}', Encoding='utf-8-sig')
    except:
        print(file_name)
        return
    try:
        for para in doc.paragraphs:
            pp = ht.clean_text(para.Range.Text)
            pp = repr(pp).replace('\\x07', '')[1:-1].replace('\\r', '').replace(' ', '')
            if pp:
                pps.append(pp)
    except:
        print(file_name)
    doc.Close()
    content = ''.join(pps)
    return content


def gen_file_index(data_web):
    file_dt = []
    for i in trange(len(data_web)):
        title = ''.join(data_web[i]['segmented_title'])
        url = data_web[i]['url'][-6:]
        files = data_web[i]['file_name']
        if not files:
            continue
        for fn in files:
            file_name = f'{title}{url}\\{fn}'
            content = process_doc(file_name)
            file_dt.append({'file_name': fn, 'content': content})
    word.Quit()
    save_json(file_dt, file_dt_path)
    return file_dt


def main():
    data_web, web_dt = load_web_dt()

    # 是否存在附件内容索引
    if os.path.exists(file_dt_path):
        file_dt = read_json(file_dt_path)
    else:
        file_dt = gen_file_index(data_web)

    retriever = Retriever(web_dt, file_dt)
    retriever.build_model(web_model_path, file_model_path)

    # 保存检索模型
    joblib.dump(retriever, './model/retriever_model.pkl')


if __name__ == '__main__':
    main()
