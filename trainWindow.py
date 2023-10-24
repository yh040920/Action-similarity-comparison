import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QFileDialog, \
    QDialog

from topdown_demo_with_mmdet import main


class WorkThread(QThread):
    # 定义一个信号
    finished = pyqtSignal()  # 定义一个信号

    def __init__(self, x, y):
        # 初始化函数
        super(WorkThread, self).__init__()
        self.parser = x
        self.fileName = y
        self._isStopped = False

    def run(self):
        main(self.parser, input_dir=self.fileName, self=self)
        self.finished.emit()

    def stop(self):
        self._isStopped = True
        self.exit()


class trainWindow(QDialog):
    def __init__(self, parser):
        super().__init__()
        self.trainTread = None
        self.fileName = None
        self.parser = parser
        self.setWindowTitle('预处理视频')
        self.resize(500, 300)

        # 创建标签，文本展示框和文件选择按钮
        self.label = QLabel("训处理视频路径")
        self.lineEdit = QLineEdit()
        self.button = QPushButton("选择文件")
        self.button.clicked.connect(self.select_file)

        # 使用QHBoxLayout并排放置这三个组件
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label)
        h_layout.addWidget(self.lineEdit)
        h_layout.addWidget(self.button)
        self.button1 = QPushButton("开始处理")
        self.button1.clicked.connect(self.deal_with)
        # 创建文本输出框
        self.textEdit = QTextEdit()

        # 使用QVBoxLayout垂直放置这两个布局
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.button1)
        v_layout.addWidget(self.textEdit)

        self.setLayout(v_layout)

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '打开文件', './result', 'MP4 files (*.mp4)')
        if file_name:
            self.lineEdit.setText(file_name)
            self.fileName = file_name

    def deal_with(self):
        if self.fileName:
            self.textEdit.append("——————开始处理——————")
            self.trainTread = WorkThread(self.parser, self.fileName)
            self.trainTread.finished.connect(self.on_finished)  # 连接信号到槽函数
            self.trainTread.start()
        else:
            self.textEdit.append("请选择处理文件！")

    def on_finished(self):
        # 这个槽函数会在TrainThread线程发出finished信号时被调用
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.textEdit.append("处理完成")
        self.textEdit.append(f"结果保存在：{script_directory}\\result文件夹下")
        self.trainTread.stop()

    def closeEvent(self, event):
        if self.trainTread:
            reply = QtWidgets.QMessageBox.question(self, '警告', '确认退出?', QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.trainTread.stop()
                event.accept()
            else:
                event.ignore()


