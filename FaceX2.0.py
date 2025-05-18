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



# import cupy as cp


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
        self.setGeometry(100, 100, 1200, 800)

        # 初始化 InsightFace 组件
        self.face_analyzer = init_face_analyzer()
        self.face_swapper = get_model('inswapper_128.onnx', download=False,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # 源图像路径和人脸
        self.source_face = None

        # 摄像头初始化 - 降低分辨率以提高性能
        self.cap = setup_camera(resolution=(320, 240), fps=30)

        # 是否开启换脸
        self.is_swapping = False

        # 录制视频相关变量
        self.recording = False
        self.out = None
        self.record_start_time = None

        # 帧处理控制
        self.frame_count = 0
        self.process_every_n_frames = 5  # 增加处理间隔到5帧
        self.last_processed_frame = None
        self.last_landmarks = None
        self.last_face = None
        self.processing_enabled = True
        
        # 添加平滑处理
        self.face_history = []  # 存储最近的人脸检测结果
        self.max_history = 10   # 增加历史记录长度
        self.landmark_history = []  # 存储最近的特征点
        self.smooth_factor = 0.85   # 增加平滑因子
        self.face_detection_confidence = 0.5  # 人脸检测置信度阈值
        self.last_valid_face = None  # 存储最后一个有效的人脸检测结果
        self.face_detection_fail_count = 0  # 人脸检测失败计数
        self.max_fail_count = 10  # 最大失败次数

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
        tips_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        # tips_label.setAlignment(Qt.AlignCenter)  # 注释掉居中对齐
        tips_label.setWordWrap(True)  # 允许文本换行
        face_list_layout.addWidget(tips_label)
        
        # 添加第二条提示
        tips_label2 = QLabel("2.可点击图片下方本地文件夹中更换换脸目标；", self)
        tips_label2.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        # tips_label2.setAlignment(Qt.AlignCenter)  # 注释掉居中对齐
        tips_label2.setWordWrap(True)
        face_list_layout.addWidget(tips_label2)
        
        # 添加第三条提示
        tips_label3 = QLabel("3.点击视频下方按钮，进行截图和录屏；", self)
        tips_label3.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        # tips_label3.setAlignment(Qt.AlignCenter)  # 注释掉居中对齐
        tips_label3.setWordWrap(True)
        face_list_layout.addWidget(tips_label3)
        
        # 添加第四条提示
        tips_label4 = QLabel("4.可以通过调节滑动条对面部进行微调", self)
        tips_label4.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
        # tips_label4.setAlignment(Qt.AlignCenter)  # 注释掉居中对齐
        tips_label4.setWordWrap(True)
        face_list_layout.addWidget(tips_label4)

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
        self.face_swap_switch.setFixedWidth(80)
        self.face_swap_switch.setMinimum(0)
        self.face_swap_switch.setMaximum(1)
        self.face_swap_switch.valueChanged.connect(self.toggle_face_swap)
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(self.face_swap_switch)
        control_inner_layout.addLayout(switch_layout)

        # 添加三个新的滑动条
        # 滑动条1
        slider1_layout = QHBoxLayout()
        slider1_label = QLabel("脸宽", self)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setStyleSheet(self.face_swap_switch.styleSheet())  # 使用相同的样式
        self.slider1.setFixedWidth(200)  # 修改宽度从80到200
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)  # 设置默认值为50
        self.slider1.valueChanged.connect(self.update_parameters)  # 添加值变化的连接
        slider1_layout.addWidget(slider1_label)
        slider1_layout.addWidget(self.slider1)
        control_inner_layout.addLayout(slider1_layout)

        # 滑动条2
        slider2_layout = QHBoxLayout()
        slider2_label = QLabel("眼长", self)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setStyleSheet(self.face_swap_switch.styleSheet())
        self.slider2.setFixedWidth(200)
        self.slider2.setMinimum(30)
        self.slider2.setMaximum(70)
        self.slider2.setValue(50)
        self.slider2.valueChanged.connect(self.update_parameters)
        slider2_layout.addWidget(slider2_label)
        slider2_layout.addWidget(self.slider2)
        control_inner_layout.addLayout(slider2_layout)

        # 滑动条3
        slider3_layout = QHBoxLayout()
        slider3_label = QLabel("眼高", self)
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setStyleSheet(self.face_swap_switch.styleSheet())
        self.slider3.setFixedWidth(200)
        self.slider3.setMinimum(30)
        self.slider3.setMaximum(70)
        self.slider3.setValue(50)
        self.slider3.valueChanged.connect(self.update_parameters)
        slider3_layout.addWidget(slider3_label)
        slider3_layout.addWidget(self.slider3)
        control_inner_layout.addLayout(slider3_layout)

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
        if value == 0:
            self.is_swapping = False
            # 清除所有缓存
            self.last_processed_frame = None
            self.last_landmarks = None
            self.last_face = None
            self.landmark_history = []
            self.face_detection_fail_count = 0
            print("换脸状态: 关闭")
        else:
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
                self.face_swap_switch.setValue(0)
                return
            self.is_swapping = True
            # 清除缓存
            self.last_processed_frame = None
            self.last_landmarks = None
            self.last_face = None
            self.landmark_history = []  # 清除特征点历史
            self.face_detection_fail_count = 0  # 重置失败计数
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

    def smooth_landmarks(self, new_landmarks):
        """平滑特征点"""
        if not self.landmark_history:
            self.landmark_history = [new_landmarks] * self.max_history
            return new_landmarks
        
        # 更新历史记录
        self.landmark_history.pop(0)
        self.landmark_history.append(new_landmarks)
        
        # 计算平滑后的特征点
        smoothed = np.zeros_like(new_landmarks)
        for landmarks in self.landmark_history:
            smoothed += landmarks
        smoothed /= len(self.landmark_history)
        
        # 应用平滑因子
        result = self.smooth_factor * smoothed + (1 - self.smooth_factor) * new_landmarks
        return result

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            # 降低处理分辨率
            small_frame = cv2.resize(frame, (320, 240))
            
            # 帧计数
            self.frame_count += 1
            
            # 每N帧处理一次
            if self.frame_count % self.process_every_n_frames == 0 and self.processing_enabled:
                # 如果开启换脸且有源图像，则执行换脸
                if self.is_swapping and self.source_face is not None:
                    try:
                        # 人脸检测
                        faces = self.face_analyzer.get(small_frame)
                        
                        if len(faces) > 0 and faces[0].det_score > self.face_detection_confidence:
                            # 获取人脸特征点
                            face = faces[0]
                            
                            # 打印调试信息
                            print("检测到人脸，置信度:", face.det_score)
                            print("人脸框:", face.bbox)
                            
                            # 获取特征点
                            if hasattr(face, 'kps'):
                                landmarks = face.kps
                                print("使用kps特征点:", landmarks.shape)
                            elif hasattr(face, 'landmark'):
                                landmarks = face.landmark
                                print("使用landmark特征点:", landmarks.shape)
                            else:
                                print("未找到特征点属性")
                                landmarks = None
                            
                            if landmarks is not None and len(landmarks) > 0:
                                # 平滑特征点
                                smoothed_landmarks = self.smooth_landmarks(landmarks)
                                
                                # 更新有效人脸
                                self.last_valid_face = face
                                self.face_detection_fail_count = 0
                                
                                # 执行换脸
                                frame = swap_faces_in_frame(small_frame, self.face_analyzer, self.face_swapper, self.source_face)
                                
                                # 缓存结果
                                self.last_processed_frame = frame.copy()
                                self.last_landmarks = smoothed_landmarks
                                self.last_face = face
                            else:
                                print("未检测到有效的特征点")
                                self.face_detection_fail_count += 1
                        else:
                            # 人脸检测失败
                            self.face_detection_fail_count += 1
                            print("人脸检测失败或置信度不足")
                            
                        # 如果失败次数未超过阈值，使用上一帧的结果
                        if self.face_detection_fail_count < self.max_fail_count and self.last_processed_frame is not None:
                            frame = self.last_processed_frame
                        else:
                            frame = small_frame
                            self.last_processed_frame = None
                            self.last_landmarks = None
                            self.last_face = None
                    except Exception as e:
                        print(f"处理人脸时出错: {str(e)}")
                        frame = small_frame
                else:
                    frame = small_frame
            else:
                # 使用缓存的结果
                if self.last_processed_frame is not None:
                    frame = self.last_processed_frame
                else:
                    frame = small_frame

            # 如果检测到人脸，应用美颜效果
            if self.last_landmarks is not None:
                try:
                    # 获取滑动条的值
                    face_strength = (self.slider1.value() - 50) / 50.0
                    # 修改眼睛缩放系数的计算方式，增加变形范围
                    eye_width = 1.0 + (self.slider2.value() - 50) / 50.0 * 1.0  # 范围从0.0到2.0
                    eye_height = 1.0 + (self.slider3.value() - 50) / 50.0 * 1.0  # 范围从0.0到2.0

                    print(f"眼睛变形参数: width={eye_width}, height={eye_height}")
                    # 处理图像（包含美颜效果）
                    processed_frame = process_image(frame, self.last_landmarks, face_strength, eye_width, eye_height)
                except Exception as e:
                    print(f"应用美颜效果时出错: {str(e)}")
                    processed_frame = frame
            else:
                processed_frame = frame

            # 如果正在录制，保存帧
            if self.recording:
                # 将处理后的帧放大到原始大小
                full_size_frame = cv2.resize(processed_frame, (640, 480))
                self.out.write(full_size_frame)

            # 转换为Qt图像格式并显示
            h, w, ch = processed_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(scaled_pixmap)

            # 显示 FPS
            current_time = time.time()
            if hasattr(self, 'last_time') and current_time - self.last_time > 1.0:
                self.fps = getattr(self, 'frame_count', 0)
                self.frame_count = 0
                self.last_time = current_time
            self.frame_count = getattr(self, 'frame_count', 0) + 1
            self.fps_label.setText(f"FPS: {getattr(self, 'fps', 0)}")

        except Exception as e:
            print(f"更新帧时出错: {str(e)}")
            # 不要自动关闭换脸开关
            # self.is_swapping = False
            # self.face_swap_switch.setValue(0)

    def closeEvent(self, event):
        # 释放资源
        self.cap.release()
        if self.out is not None:
            self.out.release()
        event.accept()

    def update_parameters(self):
        # 获取当前源图像路径
        current_source = "无源图像"
        if hasattr(self, 'source_face') and self.source_face is not None:
            # 如果是预设图像，显示预设图像路径
            for preset_path in self.preset_images:
                if os.path.exists(preset_path):
                    try:
                        img = cv2.imread(preset_path)
                        if img is not None:
                            faces = self.face_analyzer.get(img)
                            if faces and np.array_equal(faces[0].embedding, self.source_face.embedding):
                                current_source = preset_path
                                break
                    except:
                        continue

        # 获取滑动条的值
        face_width = self.slider1.value()
        eye_length = self.slider2.value()
        eye_height = self.slider3.value()

        # 在终端输出信息
        print("\n当前参数状态：")
        print(f"源图像路径: {current_source}")
        print(f"脸宽参数: {face_width}")
        print(f"眼长参数: {eye_length}")
        print(f"眼高参数: {eye_height}")


import cv2
import dlib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def load_image(path):
    """加载并验证图像"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法加载图像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hist_match(source, template):
    """直方图匹配实现（CPU版本）"""
    src_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    tgt_hist = cv2.calcHist([template], [0], None, [256], [0, 256])
    src_cdf = np.cumsum(src_hist)
    tgt_cdf = np.cumsum(tgt_hist)
    src_cdf_normalized = src_cdf / src_cdf[-1]
    tgt_cdf_normalized = tgt_cdf / tgt_cdf[-1]
    lut = np.interp(src_cdf_normalized, tgt_cdf_normalized, np.arange(256))
    return cv2.LUT(source, lut.astype(np.uint8))


def adjust_lighting(src, target):
    """光照一致性调整（CPU版本）"""
    result = np.zeros_like(src)
    for i in range(3):
        result[:, :, i] = hist_match(src[:, :, i], target[:, :, i])
    return result


def enlarge_eyes(img, landmarks, scale_x=1.0, scale_y=1.0):
    """改进的大眼效果（CPU版本）"""
    result = img.copy()

    for eye_points in [range(36, 42), range(42, 48)]:
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        center = np.mean(points, axis=0).astype(int)

        x_radius = int(np.max(np.abs(points[:, 0] - center[0])) * 2.5)
        y_radius = int(np.max(np.abs(points[:, 1] - center[1])) * 2.5)

        h, w = img.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        dx = X - center[0]
        dy = Y - center[1]
        dist = np.sqrt((dx / x_radius) ** 2 + (dy / y_radius) ** 2)

        pupil_radius = 0.3
        mask = (dist <= 1.0) & (dist > pupil_radius)

        strength_x = np.zeros_like(dist)
        strength_y = np.zeros_like(dist)
        strength_x[mask] = (1 - (dist[mask] - pupil_radius) / (1.0 - pupil_radius)) * (scale_x - 1)
        strength_y[mask] = (1 - (dist[mask] - pupil_radius) / (1.0 - pupil_radius)) * (scale_y - 1)

        map_x = (center[0] + dx * (1 + strength_x)).astype(np.float32)
        map_y = (center[1] + dy * (1 + strength_y)).astype(np.float32)

        warped = cv2.remap(result, map_x, map_y, cv2.INTER_LANCZOS4)

        blend_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(blend_mask,
                    center=tuple(center),
                    axes=(x_radius, y_radius),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=1.0,
                    thickness=-1)

        blend_mask = cv2.GaussianBlur(blend_mask, (51, 51), min(x_radius, y_radius) / 3)
        blend_mask = np.dstack([blend_mask] * 3)

        result = (warped * blend_mask + result * (1 - blend_mask)).astype(np.uint8)

    return result


def create_mls_grid(shape, src_points, dst_points):
    """优化的MLS网格变形映射（CPU版本）"""
    h, w = shape[:2]
    Y, X = np.indices((h, w))

    grid_points = np.stack([X, Y], axis=-1).reshape(-1, 2).astype(np.float32)

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    weights = np.zeros((len(grid_points), len(src_points)), dtype=np.float32)
    for i, p in enumerate(src_points):
        diff = grid_points - p
        dist = np.sqrt(np.sum(diff ** 2, axis=1)) + 1e-8
        weights[:, i] = 1 / dist

    weights /= weights.sum(axis=1, keepdims=True)

    delta = dst_points - src_points
    grid_x = X + np.sum(weights * delta[:, 0], axis=1).reshape(h, w)
    grid_y = Y + np.sum(weights * delta[:, 1], axis=1).reshape(h, w)

    return np.dstack((grid_x, grid_y))


def slim_face(img, landmarks, strength=0.3):
    """改进的瘦脸效果（CPU版本）"""
    jaw_src = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
    center_x = img.shape[1] // 2

    jaw_dst = [(x - (x - center_x) * strength, y) for x, y in jaw_src]

    grid = create_mls_grid(img.shape, jaw_src, jaw_dst)

    result = cv2.remap(img, grid[:, :, 0].astype(np.float32),
                       grid[:, :, 1].astype(np.float32),
                       cv2.INTER_LANCZOS4)

    return result


def process_image(img, landmarks, face_strength, eye_scale_x, eye_scale_y):
    """处理图像的主函数（CPU版本）"""
    try:
        result = img.copy()

        # 打印所有特征点，用于调试
        print("所有特征点:", landmarks)

        # 确保landmarks是numpy数组
        landmarks = np.array(landmarks)
        if landmarks.size == 0:
            print("特征点为空")
            return img

        # 打印特征点形状
        print("特征点形状:", landmarks.shape)

        # 根据特征点形状调整处理方式
        if landmarks.shape[0] == 5:  # 如果是5点特征点
            # 使用5点特征点进行眼睛处理
            left_eye = landmarks[0]  # 左眼中心
            right_eye = landmarks[1]  # 右眼中心
            nose = landmarks[2]  # 鼻子
            left_mouth = landmarks[3]  # 左嘴角
            right_mouth = landmarks[4]  # 右嘴角

            # 计算面部中心点
            eye_center = (left_eye + right_eye) / 2  # 两眼中心点
            face_center = (eye_center + nose) / 2  # 面部中心点（眼睛中心点和鼻子的中点）

            # 计算脸宽
            face_width = np.linalg.norm(left_eye - right_eye) * 3.0

            # 创建变形网格
            h, w = img.shape[:2]
            Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            # 计算每个点到面部中心点的距离
            points = np.stack([X, Y], axis=-1)
            face_center_points = np.tile(face_center, (h, w, 1))
            vectors = points - face_center_points
            
            # 计算到面部中心点的距离
            dist = np.sqrt(np.sum(vectors ** 2, axis=-1))
            dist = dist / face_width  # 归一化距离
            
            # 创建变形掩码
            mask = dist < 1.2

            if np.any(mask):
                # 计算变形强度
                strength = np.zeros_like(dist)
                strength[mask] = (1 - dist[mask]) * face_strength * 1.5
                
                # 计算变形方向（从面部中心点向外）
                direction = vectors[mask] / (np.linalg.norm(vectors[mask], axis=-1, keepdims=True) + 1e-8)
                
                # 应用变形
                dx = direction[:, 0] * strength[mask] * 2.0
                dy = direction[:, 1] * strength[mask] * 0.5
                
                map_x = X[mask] + dx
                map_y = Y[mask] + dy

                # 确保坐标在有效范围内
                map_x = np.clip(map_x, 0, w-1)
                map_y = np.clip(map_y, 0, h-1)

                # 创建完整的映射
                full_map_x = X.copy()
                full_map_y = Y.copy()
                full_map_x[mask] = map_x
                full_map_y[mask] = map_y

                # 创建平滑过渡的掩码
                smooth_mask = np.zeros_like(dist, dtype=np.float32)
                smooth_mask[mask] = 1.0
                
                # 使用高斯模糊创建平滑过渡
                smooth_mask = cv2.GaussianBlur(smooth_mask, (51, 51), 15)
                
                # 应用变形
                warped = cv2.remap(result, full_map_x.astype(np.float32), full_map_y.astype(np.float32), cv2.INTER_LINEAR)
                
                # 使用平滑掩码进行混合
                smooth_mask = np.dstack([smooth_mask] * 3)
                result = (warped * smooth_mask + result * (1 - smooth_mask)).astype(np.uint8)

            # 计算眼睛半径
            eye_radius = np.linalg.norm(left_eye - right_eye) * 0.3

            # 处理左眼
            center = left_eye
            x_radius = y_radius = eye_radius

            # 计算变形强度
            dx = X - center[0]
            dy = Y - center[1]
            dist = np.sqrt((dx / x_radius) ** 2 + (dy / y_radius) ** 2)

            pupil_radius = 0.2
            mask = (dist <= 1.0) & (dist > pupil_radius)

            if np.any(mask):
                strength_x = np.zeros_like(dist)
                strength_y = np.zeros_like(dist)
                
                mask_coef = (1 - (dist[mask] - pupil_radius) / (1.0 - pupil_radius))
                strength_x[mask] = mask_coef * (eye_scale_x - 1) * 2.0
                strength_y[mask] = mask_coef * (eye_scale_y - 1) * 2.0

                map_x = X + dx * strength_x
                map_y = Y + dy * strength_y

                map_x = np.clip(map_x, 0, w-1)
                map_y = np.clip(map_y, 0, h-1)

                # 创建眼睛区域的平滑掩码
                eye_mask = np.zeros_like(dist, dtype=np.float32)
                eye_mask[mask] = 1.0
                eye_mask = cv2.GaussianBlur(eye_mask, (31, 31), 10)
                
                # 应用眼睛变形
                warped = cv2.remap(result, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
                
                # 使用平滑掩码进行混合
                eye_mask = np.dstack([eye_mask] * 3)
                result = (warped * eye_mask + result * (1 - eye_mask)).astype(np.uint8)

            # 处理右眼
            center = right_eye
            dx = X - center[0]
            dy = Y - center[1]
            dist = np.sqrt((dx / x_radius) ** 2 + (dy / y_radius) ** 2)

            mask = (dist <= 1.0) & (dist > pupil_radius)

            if np.any(mask):
                strength_x = np.zeros_like(dist)
                strength_y = np.zeros_like(dist)
                
                mask_coef = (1 - (dist[mask] - pupil_radius) / (1.0 - pupil_radius))
                strength_x[mask] = mask_coef * (eye_scale_x - 1) * 2.0
                strength_y[mask] = mask_coef * (eye_scale_y - 1) * 2.0

                map_x = X + dx * strength_x
                map_y = Y + dy * strength_y

                map_x = np.clip(map_x, 0, w-1)
                map_y = np.clip(map_y, 0, h-1)

                # 创建眼睛区域的平滑掩码
                eye_mask = np.zeros_like(dist, dtype=np.float32)
                eye_mask[mask] = 1.0
                eye_mask = cv2.GaussianBlur(eye_mask, (31, 31), 10)
                
                # 应用眼睛变形
                warped = cv2.remap(result, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
                
                # 使用平滑掩码进行混合
                eye_mask = np.dstack([eye_mask] * 3)
                result = (warped * eye_mask + result * (1 - eye_mask)).astype(np.uint8)

        else:  # 如果是68点特征点
            # 使用原有的68点特征点处理方式
            jaw_src = landmarks[0:17]
            center_x = img.shape[1] // 2
            jaw_dst = np.array([(x - (x - center_x) * face_strength * 1.5, y) for x, y in jaw_src])

            h, w = img.shape[:2]
            Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid_points = np.stack([X, Y], axis=-1).reshape(-1, 2)

            weights = np.zeros((len(grid_points), len(jaw_src)))
            for i, p in enumerate(jaw_src):
                diff = grid_points - p
                dist = np.sqrt(np.sum(diff ** 2, axis=1)) + 1e-8
                weights[:, i] = 1 / dist

            weights /= weights.sum(axis=1, keepdims=True)

            delta = jaw_dst - jaw_src
            grid_x = X + np.sum(weights * delta[:, 0], axis=1).reshape(h, w)
            grid_y = Y + np.sum(weights * delta[:, 1], axis=1).reshape(h, w)

            # 创建下颌线区域的平滑掩码
            jaw_mask = np.zeros((h, w), dtype=np.float32)
            for i in range(len(jaw_src)-1):
                pt1 = tuple(map(int, jaw_src[i]))
                pt2 = tuple(map(int, jaw_src[i+1]))
                cv2.line(jaw_mask, pt1, pt2, 1.0, 20)
            jaw_mask = cv2.GaussianBlur(jaw_mask, (51, 51), 15)
            
            # 应用变形
            warped = cv2.remap(result, grid_x.astype(np.float32), grid_y.astype(np.float32), cv2.INTER_LINEAR)
            
            # 使用平滑掩码进行混合
            jaw_mask = np.dstack([jaw_mask] * 3)
            result = (warped * jaw_mask + result * (1 - jaw_mask)).astype(np.uint8)

            # 处理眼睛
            eye_count = 0
            left_eye_indices = [36, 37, 38, 39, 40, 41]
            right_eye_indices = [42, 43, 44, 45, 46, 47]
            
            for eye_indices in [left_eye_indices, right_eye_indices]:
                try:
                    eye_points = landmarks[eye_indices]
                    
                    if len(eye_points) < 6 or not np.all(np.isfinite(eye_points)):
                        continue

                    center = np.mean(eye_points, axis=0)
                    if not np.all(np.isfinite(center)):
                        continue

                    x_diffs = eye_points[:, 0] - center[0]
                    y_diffs = eye_points[:, 1] - center[1]
                    
                    if not np.all(np.isfinite(x_diffs)) or not np.all(np.isfinite(y_diffs)):
                        continue

                    x_radius = float(np.max(np.abs(x_diffs))) * 3.0
                    y_radius = float(np.max(np.abs(y_diffs))) * 3.0

                    if x_radius < 1.0 or y_radius < 1.0:
                        x_radius = max(x_radius, 1.0)
                        y_radius = max(y_radius, 1.0)

                    dx = X - center[0]
                    dy = Y - center[1]
                    dist = np.sqrt((dx / x_radius) ** 2 + (dy / y_radius) ** 2)

                    pupil_radius = 0.2
                    mask = (dist <= 1.0) & (dist > pupil_radius)

                    if not np.any(mask):
                        continue

                    strength_x = np.zeros_like(dist)
                    strength_y = np.zeros_like(dist)
                    
                    mask_coef = (1 - (dist[mask] - pupil_radius) / (1.0 - pupil_radius))
                    strength_x[mask] = mask_coef * (eye_scale_x - 1) * 2.0
                    strength_y[mask] = mask_coef * (eye_scale_y - 1) * 2.0

                    map_x = X + dx * strength_x
                    map_y = Y + dy * strength_y

                    map_x = np.clip(map_x, 0, w-1)
                    map_y = np.clip(map_y, 0, h-1)

                    # 创建眼睛区域的平滑掩码
                    eye_mask = np.zeros_like(dist, dtype=np.float32)
                    eye_mask[mask] = 1.0
                    eye_mask = cv2.GaussianBlur(eye_mask, (31, 31), 10)
                    
                    # 应用眼睛变形
                    warped = cv2.remap(result, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
                    
                    # 使用平滑掩码进行混合
                    eye_mask = np.dstack([eye_mask] * 3)
                    result = (warped * eye_mask + result * (1 - eye_mask)).astype(np.uint8)
                    eye_count += 1

                except Exception as e:
                    print(f"处理眼睛时出错: {str(e)}")
                    continue

            print(f"成功处理 {eye_count} 个眼睛")

        return result
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return img


def process_video_stream(video_frame, face_strength, eye_scale_x, eye_scale_y):
    """处理实时视频流的函数（CPU版本）"""
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = video_frame

        faces = detector(frame_rgb, 1)
        if len(faces) == 0:
            return video_frame

        landmarks = predictor(frame_rgb, faces[0])
        processed_frame = process_image(frame_rgb, landmarks, face_strength, eye_scale_x, eye_scale_y)

        if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        return processed_frame

    except Exception as e:
        print(f"处理视频帧时出错: {str(e)}")
        return video_frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec_())







