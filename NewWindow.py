from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QLineEdit, QComboBox, QFileDialog


class NewWindow(QDialog):
    # 定义一个信号，用于在参数设置好后发射
    parameters_set = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setGeometry(300, 300, 800, 450)  # 设置窗口大小和位置
        self.lower_bound_input = None
        self.device_selection = None
        self.file_path4 = None
        self.file_path3 = None
        self.file_path2 = None
        self.file_path1 = None
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("配置")

        # 添加新的组件
        self.add_new_components(layout)

        # 在这里添加你想要的用户交互选项
        button = QPushButton("确定", self)
        layout.addWidget(button)

        # 当点击按钮时，发射parameters_set信号
        button.clicked.connect(self.update_parameters)

    def add_new_components(self, layout):
        # 标签：“预测脚本文件”、路径展示框 、文件选择按钮
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("检测脚本文件"))
        self.file_path1 = QLineEdit(self)
        hbox1.addWidget(self.file_path1)
        file_button1 = QPushButton("选择文件", self)
        file_button1.clicked.connect(lambda: self.open_file_dialog(self.file_path1))
        hbox1.addWidget(file_button1)

        layout.addLayout(hbox1)

        # 标签：“预测脚本权重”、路径展示框 、文件选择按钮
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("检测脚本权重"))
        self.file_path2 = QLineEdit(self)
        hbox2.addWidget(self.file_path2)
        file_button2 = QPushButton("选择文件", self)
        file_button2.clicked.connect(lambda: self.open_file_dialog(self.file_path2))
        hbox2.addWidget(file_button2)

        layout.addLayout(hbox2)

        hboxt = QHBoxLayout()
        hboxt.addWidget(QLabel("预测脚本文件"))
        self.file_path3 = QLineEdit(self)
        hboxt.addWidget(self.file_path3)
        file_button3 = QPushButton("选择文件", self)
        file_button3.clicked.connect(lambda: self.open_file_dialog(self.file_path3))
        hboxt.addWidget(file_button3)  # 将file_button3添加到hboxt

        layout.addLayout(hboxt)

        hboxt1 = QHBoxLayout()
        hboxt1.addWidget(QLabel("预测脚本权重"))
        self.file_path4 = QLineEdit(self)
        hboxt1.addWidget(self.file_path4)
        file_button4 = QPushButton("选择文件", self)
        file_button4.clicked.connect(lambda: self.open_file_dialog(self.file_path4))
        hboxt1.addWidget(file_button4)  # 将file_button4添加到hboxt1

        layout.addLayout(hboxt1)

        # 标签：“设备选择”、选项选择组件
        hbox3 = QHBoxLayout()
        hbox3.addWidget(QLabel("设备选择"))
        self.device_selection = QComboBox(self)
        self.device_selection.addItems(["cpu", "cuda:0"])  # 添加你的设备选项
        hbox3.addWidget(self.device_selection)

        layout.addLayout(hbox3)

        self.file_path1.setText("./mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
        self.file_path2.setText("./checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
        self.file_path3.setText("./checkpoints/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py")
        self.file_path4.setText("./checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192"
                                "-63eb25f7_20230126.pth")

    def update_parameters(self, param_value):
        # 假设这是你的参数
        parameters = {"det_config": self.file_path1.text(),
                      "det_checkpoint": self.file_path2.text(),
                      "pose_config": self.file_path3.text(),
                      "pose_checkpoint": self.file_path4.text(),
                      "device-thr": self.device_selection.currentText()}
        self.parameters_set.emit(parameters)
        self.close()

    def open_file_dialog(self, line_edit):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*)",
                                                   options=options)

        if file_name:
            line_edit.setText(file_name)
