from urllib import request, parse
import requests
from bs4 import BeautifulSoup
from queue import Queue
import os
import json
from threading import Thread


def list2queue(list):
    q = Queue()
    for item in list:
        q.put(item)
    return q


class Spider:
    def __init__(self, base_url, start_urls, k, thread_num=16):
        self.urls_queue = Queue()
        self.base_url = base_url
        self.start_urls = start_urls
        self.thread_num = 10
        self.k = k
        self.results = []

    def run(self):
        self.get_urls()

        ths = []
        for _ in range(self.thread_num):
            th = Thread(target=self.craw_url)
            th.start()
            ths.append(th)
        for th in ths:
            th.join()

        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False)

    def craw_url(self):
        while not self.urls_queue.empty():
            print(f'nums: {self.urls_queue.qsize()}')
            # url = "http://today.hit.edu.cn/article/2023/03/21/101753"
            # url = "http://today.hit.edu.cn/article/2023/03/18/101656"
            # url = "http://today.hit.edu.cn/article/2023/04/19/102791"
            url = self.urls_queue.get()
            try:
                # 不设置 timeout, 容易到登陆界面
                soup = BeautifulSoup(request.urlopen(url, timeout=100), 'html.parser')
            except Exception:
                continue

            title = soup.title.string.split('|')[0].strip()
            # print(title)
            # 所有 p 标签
            raw_contents = soup.find_all('p')
            if raw_contents is None:
                continue
            contents = ""
            for raw in raw_contents:
                # 带有 font 的:
                # <p style="text-align:justify"><font face="仿宋"> 为</font><font face="仿宋">
                font_span = raw.find_all('font')
                if font_span is not None:
                    fonts = ""
                    for raw_font in font_span:
                        fc = raw_font.string
                        if fc is None:
                            continue
                        fonts += fc.strip()
                    contents += fonts

                # TODO &nbsp 在开头时，.string 会出错
                rc = raw.string
                if rc is None:
                    continue
                contents += rc.strip()
            if contents == "":
                continue
            page = {"url": url, "title": title, "paragraphs": contents, "file_name": []}
            print(url)

            file_names = []
            file_links = []
            file_spans = soup.find_all('span', class_='file')
            if file_spans is None:
                continue
            for file_span in file_spans:
                # print(file_span)
                file_link = file_span.find('a').get('href')
                file_name = file_span.find('a').string
                if "http" not in file_link:
                    file_link = parse.urljoin(self.base_url, file_link)
                if not os.path.exists(f'./attachments/{title}{url[-6:]}'):
                    os.mkdir(f'./attachments/{title}{url[-6:]}')
                path = os.path.join(f'attachments/{title}{url[-6:]}', file_name)
                # 之前没下载过才下载，方便debug
                if not os.path.exists(path):
                    response = requests.get(file_link, stream=True)
                    with open(path, "wb") as f:
                        f.write(response.content)
                file_names.append(file_name)
                file_links.append(file_link)
            page["file_name"] = file_names
            print(page)
            self.results.append(page)

    def get_urls(self):
        if os.path.exists(r'urls.json'):
            print('Load urls from Cache')
            with open('./urls.json') as f:
                urls = json.load(f)
            self.urls_queue = list2queue(urls)
            print(f'nums: {self.urls_queue.qsize()}')
            return

        urls = set()
        for start_url in self.start_urls:
            # for index in tqdm(range(0, 50), postfix="Page"):
            for index in range(0, 50):
                print(f'Page: {index}')
                # 目录页
                page_url = start_url + '?page=' + str(index)
                try:
                    soup = BeautifulSoup(request.urlopen(page_url), 'html.parser')
                except Exception:
                    continue
                for link in soup.find_all('a'):
                    url = link.get('href')
                    if url is not None and 'article/' in url and 'http' not in url:
                        urls.add(parse.urljoin(self.base_url, url))
                        print(parse.urljoin(self.base_url, url))
        urls = list(urls)
        with open('./urls.json', 'w') as f:
            json.dump(urls, f)
        self.urls_queue = list2queue(urls)
        print(f'nums: {self.urls_queue.qsize()}')


def main():
    start_urls = ["http://today.hit.edu.cn/category/10", "http://today.hit.edu.cn/category/11"]
    spider = Spider("http://today.hit.edu.cn", start_urls, 1000)
    spider.run()


if __name__ == '__main__':
    main()
