import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, \
    QFileDialog, QListWidget, QListWidgetItem, QScrollArea, QMessageBox, QSizePolicy, QGridLayout, QSlider
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtWidgets import QLineEdit  # 添加到现有的imports中
import requests  # 用于API调用
import json  # 用于处理API响应
import os  # 用于文件操作


def init_face_analyzer(det_size=(640, 640)):
    # 初始化人脸检测器，使用 GPU
    analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    analyzer.prepare(ctx_id=0, det_size=det_size)
    return analyzer


def load_source_face(img_path, analyzer):
    # 加载源图像并提取人脸
    source_img = cv2.imread(img_path)
    if source_img is None:
        raise FileNotFoundError(f"无法加载源图像: {img_path}")
    faces = analyzer.get(source_img)
    if not faces:
        raise ValueError("未在源图像中检测到人脸")
    return faces[0]


def setup_camera(resolution=(320, 240), fps=30):
    # 初始化摄像头捕获，并设置分辨率和帧率
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise IOError("无法打开摄像头")
    return cap


def swap_faces_in_frame(frame, analyzer, swapper, source_face):
    # 在帧中进行人脸替换
    target_faces = analyzer.get(frame)
    if target_faces:
        target_face = target_faces[0]
        # 使用 GPU 进行人脸替换
        swapped_frame = swapper.get(frame, target_face, source_face, paste_back=True)
        return swapped_frame
    return frame


class FaceSwapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时换脸应用")
        self.setGeometry(100, 100, 1200, 800)  # 初始窗口大小

        # 初始化 InsightFace 组件
        self.face_analyzer = init_face_analyzer()
        self.face_swapper = get_model('inswapper_128.onnx', download=False,
                                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # 源图像路径和人脸
        self.source_face = None

        # 摄像头初始化
        self.cap = setup_camera(resolution=(640, 480), fps=30)

        # 是否开启换脸
        self.is_swapping = False

        # 录制视频相关变量
        self.recording = False
        self.out = None
        self.record_start_time = None

        # 预设的四张图片路径
        self.preset_images = [
            "pictures/img.png",
            "pictures/img_2.png",
            "pictures/img_3.png",
            "pictures/img_4.png"
        ]

        # 创建 GUI 元素
        self.init_ui()

        # 定时器更新视频流
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 每 33 毫秒更新一次

        # 添加预设图片到列表
        self.init_preset_images()

    def init_ui(self):
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1c1c1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #323236;
                border: none;
                border-radius: 10px;
                color: #ffffff;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3a3a3e;
            }
            QPushButton:pressed {
                background-color: #2c2c30;
            }
            QListWidget {
                background-color: #2c2c30;
                border: none;
                border-radius: 10px;
                color: #ffffff;
            }
        """)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 左侧：视频显示区域
        video_layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: #2c2c30; border-radius: 15px; padding: 10px;")
        self.video_label.setMinimumSize(640, 480)  # 设置最小尺寸
        self.video_label.setMaximumSize(960, 720)  # 设置最大尺寸
        video_layout.addWidget(self.video_label)

        # FPS 标签
        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.fps_label)

        # 录制时间标签
        self.record_time_label = QLabel("", self)
        self.record_time_label.setAlignment(Qt.AlignCenter)
        self.record_time_label.hide()
        video_layout.addWidget(self.record_time_label)

        # 底部按钮布局
        bottom_layout = QHBoxLayout()

        # 截图按钮
        self.screenshot_button = QPushButton("📸 截图", self)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        bottom_layout.addWidget(self.screenshot_button)

        # 录制按钮
        self.record_button = QPushButton("🔴 开始录制", self)
        self.record_button.clicked.connect(self.toggle_recording)
        bottom_layout.addWidget(self.record_button)

        video_layout.addLayout(bottom_layout)
        main_layout.addLayout(video_layout)

        # 中间：提示信息区域
        face_list_layout = QVBoxLayout()
        title_label = QLabel("Tips", self)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)  # 设置文字居中
        face_list_layout.addWidget(title_label)
        
        # 添加提示内容
        tips_label = QLabel("1.点击图像作为换脸目标，滑动换脸开关；", self)
        tips_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        tips_label.setWordWrap(True)  # 允许文本换行
        face_list_layout.addWidget(tips_label)
        
        # 添加第二条提示
        tips_label2 = QLabel("2.可点击图片下方本地文件夹中更换换脸目标；", self)
        tips_label2.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        tips_label2.setWordWrap(True)
        face_list_layout.addWidget(tips_label2)
        
        # 添加第三条提示
        tips_label3 = QLabel("3.点击视频下方按钮，进行截图和录屏；", self)
        tips_label3.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        tips_label3.setWordWrap(True)
        face_list_layout.addWidget(tips_label3)

        # 创建2x2网格布局
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background-color: #2c2c30; border-radius: 15px; padding: 20px;")
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(20)
        self.image_buttons = []
        self.replace_buttons = []

        for i in range(4):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setAlignment(Qt.AlignCenter)

            # 创建图片按钮
            img_button = QPushButton()
            img_button.setFixedSize(150, 150)
            img_button.setIconSize(QSize(140, 140))
            img_button.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3e;
                    border-radius: 15px;
                }
                QPushButton:hover {
                    background-color: #444448;
                }
            """)
            img_button.clicked.connect(lambda checked, index=i: self.select_preset_image(index))
            self.image_buttons.append(img_button)
            container_layout.addWidget(img_button)

            # 创建替换按钮
            replace_button = QPushButton("更换图片")
            replace_button.setStyleSheet("font-size: 12px; padding: 5px 10px;")
            replace_button.clicked.connect(lambda checked, index=i: self.replace_preset_image(index))
            self.replace_buttons.append(replace_button)
            container_layout.addWidget(replace_button)

            row = i // 2
            col = i % 2
            grid_layout.addWidget(container, row, col, Qt.AlignCenter)

        face_list_layout.addWidget(grid_widget)
        main_layout.addLayout(face_list_layout)

        # 右侧：控制面板
        control_layout = QVBoxLayout()
        control_widget = QWidget()
        control_widget.setStyleSheet("background-color: #2c2c30; border-radius: 15px; padding: 20px;")
        control_inner_layout = QVBoxLayout(control_widget)

        # 源图像选择按钮
        self.select_button = QPushButton("选择源图像", self)
        self.select_button.clicked.connect(self.select_source_image)
        control_inner_layout.addWidget(self.select_button)

        # 换脸开关
        switch_layout = QHBoxLayout()
        switch_label = QLabel("换脸开关", self)
        self.face_swap_switch = QSlider(Qt.Horizontal)
        self.face_swap_switch.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #323236;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0a84ff;
                border: 2px solid #0a84ff;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #40a9ff;
                border: 2px solid #40a9ff;
            }
            QSlider::sub-page:horizontal {
                background: #0a84ff;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #323236;
                border-radius: 4px;
            }
        """)
        self.face_swap_switch.setFixedWidth(80)  # 增加滑块宽度
        self.face_swap_switch.setMinimum(0)
        self.face_swap_switch.setMaximum(1)
        self.face_swap_switch.valueChanged.connect(self.toggle_face_swap)
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(self.face_swap_switch)
        control_inner_layout.addLayout(switch_layout)

        # 源图像显示区域
        self.image_label = QLabel("源图像", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #323236; border-radius: 10px; padding: 10px;")
        self.image_label.setMinimumSize(160, 120)
        control_inner_layout.addWidget(self.image_label)

        control_layout.addWidget(control_widget)
        main_layout.addLayout(control_layout)

        # 设置布局比例
        main_layout.setStretch(0, 2)  # 视频区域
        main_layout.setStretch(1, 1)  # 备选图像区域
        main_layout.setStretch(2, 1)  # 控制面板

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.record_button.setText("🔴 开始录制")
            self.record_time_label.hide()
            self.out.release()
            if hasattr(self, 'record_timer'):
                self.record_timer.stop()
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("录制完成")
            msg_box.setText(f"视频已保存为: {self.current_video_path}")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #323236;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0a84ff;
                    border: none;
                    border-radius: 5px;
                    color: #ffffff;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #40a9ff;
                }
            """)
            msg_box.exec_()
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = int(time.time())
            self.current_video_path = f"recording_{timestamp}.mp4"
            self.out = cv2.VideoWriter(self.current_video_path, fourcc, 30.0, (640, 480))
            self.recording = True
            self.record_button.setText("⏹️ 停止录制")
            self.record_time_label.show()
            self.record_start_time = time.time()
            self.record_timer = QTimer(self)
            self.record_timer.timeout.connect(self.update_record_time)
            self.record_timer.start(1000)  # 更新录制时间每秒一次

    def update_record_time(self):
        elapsed_time = int(time.time() - self.record_start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        self.record_time_label.setText(f"录制时间: {minutes:02}:{seconds:02}")

    def take_screenshot(self):
        # 获取当前视频帧
        ret, frame = self.cap.read()
        if ret:
            # 如果开启换脸且有源图像，则执行换脸
            if self.is_swapping and self.source_face is not None:
                frame = swap_faces_in_frame(frame, self.face_analyzer, self.face_swapper, self.source_face)

            timestamp = int(time.time())
            screenshot_path = f"screenshot_{timestamp}.png"
            cv2.imwrite(screenshot_path, frame)
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("截图成功")
            msg_box.setText(f"截图已保存为: {screenshot_path}")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #323236;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0a84ff;
                    border: none;
                    border-radius: 5px;
                    color: #ffffff;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #40a9ff;
                }
            """)
            msg_box.exec_()

    def select_source_image(self):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择源图像", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                # 加载源图像并提取人脸
                self.source_face = load_source_face(file_path, self.face_analyzer)

                # 显示源图像
                source_img = cv2.imread(file_path)
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                h, w, ch = source_img.shape
                bytes_per_line = ch * w
                q_img = QImage(source_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

                print(f"成功加载源图像: {file_path}")
            except Exception as e:
                print(f"加载源图像失败: {str(e)}")

    def toggle_face_swap(self, value):
        if value == 0:  # 滑块在左边
            self.is_swapping = False
            print("换脸状态: 关闭")
        else:  # 滑块在右边
            if self.source_face is None:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("提示")
                msg_box.setText("请选择换脸目标人物图像")
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #323236;
                    }
                    QMessageBox QLabel {
                        color: #ffffff;
                    }
                    QPushButton {
                        background-color: #0a84ff;
                        border: none;
                        border-radius: 5px;
                        color: #ffffff;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #40a9ff;
                    }
                """)
                msg_box.exec_()
                self.face_swap_switch.setValue(0)  # 将滑块重置到左边
                return
            self.is_swapping = True
            print("换脸状态: 开启")

    def select_preset_image(self, index):
        if index < len(self.preset_images):
            try:
                self.source_face = load_source_face(self.preset_images[index], self.face_analyzer)

                # 显示源图像
                source_img = cv2.imread(self.preset_images[index])
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                h, w, ch = source_img.shape
                bytes_per_line = ch * w
                q_img = QImage(source_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

                print(f"选择了预设图像: {self.preset_images[index]}")
            except Exception as e:
                print(f"选择预设图像失败: {str(e)}")

    def replace_preset_image(self, index):
        if index < len(self.preset_images):
            file_path, _ = QFileDialog.getOpenFileName(self, "选择新图像", "", "Image Files (*.png *.jpg *.jpeg)")
            if file_path:
                try:
                    # 加载新图像并提取人脸
                    new_face = load_source_face(file_path, self.face_analyzer)

                    # 更新预设图像路径
                    self.preset_images[index] = file_path

                    # 更新按钮图标
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    icon = QIcon(QPixmap.fromImage(q_img).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.image_buttons[index].setIcon(icon)

                    print(f"替换了预设图像: {file_path}")
                except Exception as e:
                    print(f"替换预设图像失败: {str(e)}")

    def init_preset_images(self):
        for i, img_path in enumerate(self.preset_images):
            if i >= 4:
                break
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                icon = QIcon(QPixmap.fromImage(q_img).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image_buttons[i].setIcon(icon)
            except Exception as e:
                print(f"加载预设图像失败: {str(e)}")

    def update_frame(self):
        # 读取摄像头帧
        ret, frame = self.cap.read()
        if ret:
            # 缩小图像尺寸以提高性能
            frame = cv2.resize(frame, (640, 480))

            # 如果开启换脸且有源图像，则执行换脸
            if self.is_swapping and self.source_face is not None:
                frame = swap_faces_in_frame(frame, self.face_analyzer, self.face_swapper, self.source_face)

            # 录制视频时使用BGR格式
            if self.recording:
                self.out.write(frame)  # 直接写入frame，因为原始frame就是BGR格式

            # 显示时转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

            # 显示 FPS
            current_time = time.time()
            if hasattr(self, 'last_time') and current_time - self.last_time > 1.0:
                self.fps = getattr(self, 'frame_count', 0)
                self.frame_count = 0
                self.last_time = current_time
            self.frame_count = getattr(self, 'frame_count', 0) + 1
            self.fps_label.setText(f"FPS: {getattr(self, 'fps', 0)}")

    def closeEvent(self, event):
        # 释放资源
        self.cap.release()
        if self.out is not None:
            self.out.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec_())



