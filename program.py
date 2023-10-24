import os.path
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget, QMenuBar, QAction, QFileDialog, \
    QTextEdit, QPushButton, QHBoxLayout, QCheckBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QFont
import sys

from matplotlib import pyplot as plt

from trainWindow import trainWindow
from NewWindow import NewWindow
from Threads import TeacherThread, VideoThread


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.timer = None
        self.isOver = False
        self.train_window = None
        self.new_window = None
        self.parser = {'det_config': './mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
                       'det_checkpoint': './checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
                       'pose_config': './checkpoints/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py',
                       'pose_checkpoint': './checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192'
                                          '-63eb25f7_20230126.pth',
                       'device': 'cuda:0'}

        self.thread = None
        self.thread_teacher = None
        self.isConnected = None
        self.setWindowTitle('Yyh')  # 窗口标题为空
        self.setGeometry(300, 100, 1600, 1200)  # 设置窗口大小和位置

        # 创建菜单栏
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)
        file_menu = menubar.addMenu('配置')
        open_action = QAction('预测参数配置', self)
        file_menu.addAction(open_action)
        # 当点击open_action时，打开新的窗口
        open_action.triggered.connect(self.open_new_window)

        open_action1 = QAction('视频预处理', self)
        file_menu.addAction(open_action1)
        # 当点击open_action时，打开新的窗口
        open_action1.triggered.connect(self.open_train_window)

        # 创建主窗口部件和布局
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        # 创建两个标签用于显示图片，并设置相同的大小
        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        # self.label1.setFixedSize(360, 640)  # 设置标签的大小
        # 创建QFont对象
        font = QFont()
        font.setPointSize(20)  # 设置字体大小为10
        self.label = QLabel('', self)
        self.label.setFont(font)
        self.label2.setFixedSize(500, 500)  # 设置标签的大小
        self.label1.setFixedSize(800, 800)

        # 创建一个水平布局，并将两个标签添加到这个布局中
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(self.label1)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label)
        h_layout.addStretch(1)  # 添加一个可伸缩的空间
        h_layout.addWidget(self.label2)
        h_layout.addStretch(1)

        layout.addLayout(h_layout)  # 将水平布局添加到主布局中

        button_layout = QHBoxLayout()

        check_layout = QVBoxLayout()
        cb1 = QCheckBox('跟练', self)
        cb1.setChecked(True)
        cb2 = QCheckBox('检验', self)
        check_layout.addWidget(cb1)
        check_layout.addWidget(cb2)
        self.group = QButtonGroup(self)
        self.group.addButton(cb1, 1)  # 第二个参数是设置这个按钮的 id
        self.group.addButton(cb2, 2)
        self.group.setExclusive(True)
        self.group.buttonClicked.connect(self.on_button_clicked)

        button_layout.addLayout(check_layout)
        self.button1 = QPushButton('打开teacher视频', self)
        layout.addWidget(self.button1)
        self.button1.clicked.connect(self.open_file)

        self.button2 = QPushButton('开始对比', self)
        layout.addWidget(self.button2)
        self.button2.clicked.connect(self.start_contrast)

        self.button3 = QPushButton('停止对比', self)
        layout.addWidget(self.button3)
        self.button3.clicked.connect(self.stop_contrast)

        self.button5 = QPushButton('开启摄像头', self)
        layout.addWidget(self.button5)
        self.button5.clicked.connect(self.start_webcam)

        self.button4 = QPushButton('关闭摄像头', self)
        layout.addWidget(self.button4)
        self.button4.clicked.connect(self.renew_webcam)

        self.button6 = QPushButton('删除teacher视频', self)
        layout.addWidget(self.button6)
        self.button6.clicked.connect(self.renew_teacherVideo)

        self.button6.setFixedSize(100, 50)
        self.button1.setFixedSize(100, 50)
        self.button2.setFixedSize(100, 50)
        self.button3.setFixedSize(100, 50)
        self.button4.setFixedSize(100, 50)
        self.button5.setFixedSize(100, 50)
        button_layout.addWidget(self.button6)
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        button_layout.addWidget(self.button5)
        button_layout.addWidget(self.button4)
        layout.addLayout(button_layout)

        # 创建文本编辑框用于显示文本信息
        self.text_edit = QTextEdit(self)
        layout.addWidget(self.text_edit)

    def on_button_clicked(self):
        self.isOver = False
        checked_id = self.group.checkedId()
        print(f"Checked button id is: {checked_id}")
        self.renew_teacherVideo()

    def setImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        qtImage = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.label2.setPixmap(QPixmap.fromImage(qtImage))

    def setLable1Image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytesPerLine = ch * w
        qtImage = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.label1.setPixmap(QPixmap.fromImage(qtImage))

    def open_file(self):
        self.isOver = False
        file_name, _ = QFileDialog.getOpenFileName(self, '打开文件', './result', 'MP4 files (*.mp4)')
        if file_name:
            if os.path.exists(file_name.split('.')[0] + '.json'):
                if self.thread_teacher:
                    self.thread_teacher.exit()
                    self.thread_teacher = None
                self.thread_teacher = TeacherThread(file_name, self.text_edit, self.group)
                self.thread_teacher.update_label1.connect(self.setLable1Image)
                self.thread_teacher.stop_contrast.connect(self.stop_contrast)
                self.thread_teacher.emit_all_scores.connect(self.start_plt_show)

                self.thread_teacher.start()
                self.text_edit.append(f"打开文件:{file_name}")
            else:
                self.text_edit.append("视频不是teacher视频，请先预测！")

    def start_plt_show(self, scores_list):
        scores_list = scores_list * 100
        plt.plot(scores_list)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        mean_score = np.mean(scores_list)
        plt.title(f'你的得分是：{np.round(mean_score, 2)}!', fontsize=20, color='red',
                  bbox={'facecolor': 'grey', 'pad': 10})
        plt.ylim(0, 100)
        plt.xlabel('对比次数')
        plt.ylabel('分数')
        plt.show()

    def refresh(self):
        current_value = int(self.label.text())
        if current_value > 0:
            self.label.setText(str(current_value - 1))
        else:
            self.timer.stop()
            self.label.setText("开始")
            self.thread.emit_to_teacher.connect(self.thread_teacher.loop_list)
            self.isConnected = True
            self.label.setText("")

    def start_contrast(self):

        if self.thread_teacher and self.thread:
            if not self.isConnected and not self.isOver:
                self.label.setText('3')
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.refresh)
                self.timer.start(1000)

        elif self.thread is None:
            self.text_edit.append("摄像头未打开！")
        else:
            self.text_edit.append("teacher未打开！")

    def stop_contrast(self, *args):
        if len(args) == 1:
            self.isOver = args[0]
        try:
            if self.thread_teacher and self.thread_teacher.loop_list and self.thread:
                if self.group.checkedId() == 1:
                    self.thread.emit_to_teacher.disconnect(self.thread_teacher.loop_list)
                else:
                    self.thread.emit_to_teacher.disconnect(self.thread_teacher.loop_list)
                    self.renew_webcam()
                self.isConnected = False

        except TypeError:
            pass

    def handle_parameters(self, parameters):
        self.parser['det_config'] = parameters['det_config']
        self.parser['det_checkpoint'] = parameters['det_checkpoint']
        self.parser['pose_config'] = parameters['pose_config']
        self.parser['pose_checkpoint'] = parameters['pose_checkpoint']
        self.parser['device'] = parameters['device-thr']

    def open_new_window(self):
        self.new_window = NewWindow()
        self.new_window.show()

        # 连接new_window的signal到handle_parameters函数
        self.new_window.parameters_set.connect(self.handle_parameters)

    def start_webcam(self):
        if self.thread:
            pass
        else:
            self.thread = VideoThread(self.parser)
            self.thread.start()
            self.thread.changePixmap.connect(self.setImage)

    def renew_webcam(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.label2.setPixmap(QPixmap(""))

    def open_train_window(self):
        self.train_window = trainWindow(self.parser)
        self.train_window.show()

    def renew_teacherVideo(self):
        self.isOver = False
        self.isConnected = False
        if self.thread_teacher:
            self.thread_teacher.stop()
            self.thread_teacher = None
            self.label1.setPixmap(QPixmap(""))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
