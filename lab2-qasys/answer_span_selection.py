import re
from utils import pos_tag, read_jsonlist, write_json
from tqdm import tqdm


def predict(test_path):
    test_data = read_jsonlist(test_path)
    res = []
    for data in tqdm(test_data):
        ans_words = data['answer_sentence']
        cls = data['cls']
        ans = "".join(ans_words)
        ans_pos = pos_tag(ans_words)
        data['answer'] = ''
        if '：' in ans or ':' in ans:
            data['answer'] = ans.split('：')[1] if '：' in ans else ans.split(':')[1]
        elif cls.startswith('DES'):
            data['answer'] = pos_answer(ans_words, ans_pos, [])
        elif cls.startswith('HUM'):
            data['answer'] = pos_answer(ans_words, ans_pos, ['ni', 'nh', 'nt'])
        elif cls.startswith('LOC'):
            data['answer'] = pos_answer(ans_words, ans_pos, ['ns', 'nl'])
        elif cls.startswith('NUM'):
            for idx, tag in enumerate(ans_pos):
                if tag == 'm' and idx < len(ans_pos) - 1 and ans_pos[idx + 1] == 'q':
                    data['answer'] = ans_words[idx] + ans_words[idx + 1]
                    break
        elif cls.startswith('TIME'):
            if cls == 'TIME_YEAR':
                res_lst = re.findall(r'\d{2,4}年', ans)
            elif cls == 'TIME_MONTH':
                res_lst = re.findall(r'\d{1,2}月', ans)
            elif cls == 'TIME_DAY':
                res_lst = re.findall(r'\d{1,2}日}', ans)
            elif cls == 'TIME_WEEK':
                res_lst = re.findall(r'((周|星期|礼拜)[1-7一二三四五六日])', ans)
                res_lst = [item[0] for item in res_lst]
            elif cls == 'TIME_RANGE':
                res_lst = re.findall(r'\d{2,4}[年]?[-到至]\d{2,4}[年]?', ans)
            else:
                res_lst = re.findall(r'\d{1,4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?', ans)
            if not res_lst:
                res_lst = re.findall(r'\d{1,4}[年/-]\d{1,2}月?', ans)
            if not res_lst:
                res_lst = re.findall(r'\d{1,2}[月/-]\d{1,2}[日号]?', ans)
            if not res_lst:
                res_lst = re.findall(r'\d{2,4}年', ans)
            if not res_lst:
                res_lst = re.findall(r'\d{1,2}月', ans)
            data['answer'] = res_lst[0] if res_lst is not None and len(res_lst) > 0 else ans
        if data['answer'] == '':
            data['answer'] = ans
        res.append({
            'qid': data['qid'],
            'question': data['question'],
            'answer_pid': [data['pid']],
            'answer': data['answer']
        })
    write_json('./test_answer.json', res)


def pos_answer(words, words_pos, pos):
    res = []
    for i in range(len(words_pos)):
        if words_pos[i] in pos:
            res.append(words[i])
    if len(res):
        return ''.join(res)
    else:
        return ''.join(words)


if __name__ == '__main__':
    predict('./data/test_ans.json')
