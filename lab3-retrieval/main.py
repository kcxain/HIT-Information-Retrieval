# coding: utf-8
import sys

import joblib
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QAbstractItemView

from gui.login import Ui_Form as login_Form
from gui.search import Ui_Form as search_Form
from preprocess import load_web_dt

data_web, web_dt = load_web_dt()

web_header = ['网址', '文章标题', '相关文档列表']
file_header = ['相关文档']


class MainWindow(QMainWindow, login_Form):
    switch_window = pyqtSignal()  # 界面跳转信号
    level_info = pyqtSignal(int)  # 页面间传递参数

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # 添加登录按钮信号和槽, 调用相应函数注意display函数不加小括号()
        self.login_button.clicked.connect(self.login)

    def login(self):
        user = self.user_comboBox.currentText()  # 当前角色
        if user == '老师':
            level = 1
        elif user == '学生':
            level = 2
        elif user == '家长':
            level = 3
        else:  # 游客无需密码访问
            level = 4

        pwd = self.password_input.text()  # 利用line Edit控件对象text()函数获取界面输入

        QMessageBox.information(self, '成功', '登录成功！')
        self.password_input.clear()
        self.level_info.emit(level)  # 传递level信息
        self.switch_window.emit()  # 页面跳转


# 老师：level=1; 可获取相关性前5的数据, url:可访问，title:可访问, 文档:可访问, paragraph:可访问
# 学生：level=2; 可获取相关性前5的数据, url:可访问，title:可访问, 文档:禁止访问, paragraph:可访问
# 家长：level=3; 可获取相关性前3的数据, url:禁止访问，title:可访问, 文档:禁止访问, paragraph:可访问
# 游客：level=4; 可获取相关性前1的数据, url:禁止访问，title:可访问, 文档:禁止访问, paragraph:可访问


class SearchWindow(QMainWindow, search_Form):
    switch_window = pyqtSignal()  # 界面跳转信号

    def __init__(self):
        super(SearchWindow, self).__init__()
        self.setupUi(self)
        self.retriever = joblib.load('./model/retriever_model.pkl')
        # 按钮触发绑定
        self.web_button.clicked.connect(self.web_search)
        self.file_button.clicked.connect(self.file_search)
        self.return_button.clicked.connect(self.go_back)
        self.table_show.doubleClicked.connect(self.show_para)

    def show_para(self, index):
        num = len(self.paras)
        row = index.row()
        if num == 0:
            QMessageBox.information(self, '错误', '数据为空！')
        else:
            #             if row <= num:
            QMessageBox.information(self, '内容', self.paras[row])

    def go_back(self):
        self.model.clear()
        self.switch_window.emit()  # 页面跳转

    def get_data(self, level_info):
        self.level = level_info  # 接受 login 页面传过来的level_info，并保存

    def web_search(self):
        query = self.query_input.text()
        self.query_input.clear()
        self.model = QStandardItemModel(5, 3)  # 5行3列
        self.model.setHorizontalHeaderLabels(web_header)
        self.paras = []
        if query == '':
            QMessageBox.information(self, '错误', '检索内容为空！')
        # '网址','文章标题','相关文档列表'
        else:
            top5 = self.retriever.search_web(query)
            print(top5)
            if self.level == 1:  # 老师
                for i in range(5):
                    data = self.retriever.web_dt.iloc[top5[i]]
                    print(data)
                    self.paras.append(''.join(data['segmented_parapraghs']))
                    self.model.setItem(i, 0, QStandardItem(data['url']))
                    self.model.setItem(i, 1, QStandardItem(''.join(data['segmented_title'])))
                    self.model.setItem(i, 2, QStandardItem(str(data['file_name'])))
            elif self.level == 2:  # 学生
                for i in range(5):
                    data = self.retriever.web_dt.iloc[top5[i]]
                    self.paras.append(''.join(data['segmented_parapraghs']))
                    self.model.setItem(i, 0, QStandardItem(data['url']))
                    self.model.setItem(i, 1, QStandardItem(''.join(data['segmented_title'])))
                    self.model.setItem(i, 2, QStandardItem('当前用户暂无访问权限'))
            elif self.level == 3:  # 家长
                for i in range(3):
                    data = web_dt.iloc[top5[i]]
                    self.paras.append(''.join(data['segmented_parapraghs']))
                    self.model.setItem(i, 0, QStandardItem('当前用户暂无访问权限'))
                    self.model.setItem(i, 1, QStandardItem(''.join(data['segmented_title'])))
                    self.model.setItem(i, 2, QStandardItem('当前用户暂无访问权限'))

            else:  # 游客
                i = 0
                data = self.retriever.web_dt.iloc[top5[i]]
                self.paras.append(''.join(data['segmented_parapraghs']))
                self.model.setItem(i, 0, QStandardItem('当前用户暂无访问权限'))
                self.model.setItem(i, 1, QStandardItem(''.join(data['segmented_title'])))
                self.model.setItem(i, 2, QStandardItem('当前用户暂无访问权限'))
            QMessageBox.information(self, '成功', '检索数据成功！')

        self.table_show.setModel(self.model)
        self.table_show.setColumnWidth(0, 300)
        self.table_show.setColumnWidth(1, 370)
        self.table_show.setColumnWidth(2, 370)
        self.table_show.setRowHeight(0, 60)
        self.table_show.setRowHeight(1, 60)
        self.table_show.setRowHeight(2, 60)
        self.table_show.setRowHeight(3, 60)
        self.table_show.setRowHeight(4, 60)
        #         self.table_show.horizontalHeader().setStretchLastSection(True)
        self.table_show.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.query_input.clear()

    def file_search(self):
        query = self.query_input.text()
        self.model = QStandardItemModel(5, 1)  # 5行1列
        self.model.setHorizontalHeaderLabels(file_header)
        self.paras = []
        if self.level != 1:
            QMessageBox.information(self, '警告', '您当前暂无访问权限！')
        else:
            # '文档名称'
            if query == '':
                QMessageBox.information(self, '错误', '检索内容为空！')
            else:  # 老师
                top5 = self.retriever.search_file(query)
                for i in range(5):
                    data = self.retriever.file_dt.iloc[top5[i]]
                    self.paras.append(data['content'])
                    self.model.setItem(i, 0, QStandardItem(data['file_name']))
                QMessageBox.information(self, '成功', '检索文档成功！')

        self.table_show.setModel(self.model)
        self.table_show.setColumnWidth(0, 1300)
        self.table_show.setRowHeight(0, 60)
        self.table_show.setRowHeight(1, 60)
        self.table_show.setRowHeight(2, 60)
        self.table_show.setRowHeight(3, 60)
        self.table_show.setRowHeight(4, 60)
        self.table_show.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.query_input.clear()


# 利用控制器来控制页面的跳转
class Controller:
    def __init__(self):
        self.login = MainWindow()
        self.search = SearchWindow()
        pass

    # login 窗口
    def show_login(self):
        self.search.close()
        self.login.switch_window.connect(self.show_search)  # 跳转信号绑定
        self.login.level_info.connect(self.search.get_data)  # 参数传递信号绑定
        self.login.show()

    # 跳转到 search 窗口, 并关闭原页面
    def show_search(self):
        self.login.close()
        self.search.switch_window.connect(self.show_login)  # 跳转信号绑定
        self.search.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    controller = Controller()  # 控制器实例
    controller.show_login()  # 默认展示的是 login 页面

    sys.exit(app.exec_())
