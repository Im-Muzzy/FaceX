# FaceX 🎭 实时换脸应用

## 项目简介 📖
FaceX 是一个基于 Python 和 PyQt5 开发的实时换脸应用程序，利用 InsightFace 深度学习框架实现高质量的人脸替换效果。该应用支持摄像头实时换脸、截图、录制视频等功能，提供了简洁直观的用户界面。

---

## 功能特点 🚀

- 实时摄像头人脸替换 🎥
- 预设人脸模板选择 🖼️
- 自定义人脸图像导入 💾
- 截图保存功能 📸
- 视频录制功能 🎞️
- 人脸参数微调（FaceX 2.0版本） 🔧
- GPU 加速支持 ⚡

---

## 版本对比 🆚

### FaceX 1.0
- 基础的实时换脸功能
- 简洁的用户界面
- 预设人脸模板
- 截图和录制功能
- GPU提升处理性能

### FaceX 2.0
- 增强的人脸检测稳定性
- 添加人脸平滑处理
- 增加面部参数微调功能
- 优化的用户界面和提示信息

---
# 📸 展示模块

为了更直观地展示 FaceX 的功能与改进，我们提供了 **图形界面截图** 和 **换脸效果对比**。

---

## 🧩 界面展示

### 💻 FaceX 1.0 界面（基础版）

![FaceX 1.0 界面](https://yourimagehosting.com/path/to/FaceX1.0_ui.png)

📌 特点：
- 简洁明了的操作面板
- 支持摄像头实时换脸
- 内置预设人脸模板
- 基础截图与录屏功能

---

### 🚀 FaceX 2.0 界面（增强版）

![FaceX 2.0 界面](https://yourimagehosting.com/path/to/FaceX2.0_ui.png)

📌 升级亮点：
- 新增面部参数滑动条，支持微调
- 更流畅的界面交互体验
- 更清晰的状态提示信息
- 支持 GPU 加速状态显示

---

## 🎭 换脸结果展示

为了让使用者清楚看到 FaceX 的换脸效果，以下是不同场景下的换脸前后对比。

### 🎞️ 视频换脸动图演示
-FaceX1.0

<p align="center">
  <img src="https://github.com/Im-Muzzy/Im-Muzzy/blob/main/images/changeFaceResult.gif" alt="实时换脸动图" width="600"/>
</p>

-FaceX2.0
<p align="center">
  <img src="https://yourimagehosting.com/path/to/faceswap-demo.gif" alt="实时换脸动图" width="600"/>
</p>
---

## 安装要求 🛠️

### 环境依赖
- Python 3.7+
- CUDA 支持的 GPU（推荐）
- 摄像头设备

### 必要库
```plaintext
opencv-python
numpy
insightface
PyQt5
onnxruntime-gpu
```

---

## 使用说明 📚

1. **启动应用程序**
- 版本1：
```bash
python FaceX1.0.py
```
- 版本2
```bash
python FaceX2.0.py
```

2. **操作指南**
   - 点击图像作为换脸目标，滑动换脸开关
   - 可点击图片下方按钮从本地文件夹中更换换脸目标
   - 点击视频下方按钮，进行截图和录屏
   - 可以通过调节滑动条对面部进行微调（FaceX 2.0版本）

---

## 技术实现 🤖

FaceX 应用基于以下技术实现：

- **InsightFace** 深度学习框架用于人脸检测和替换
- **ONNX Runtime** 提供 GPU 加速
- **PyQt5** 构建图形用户界面
- **OpenCV** 处理图像和视频流

---

## 算法概述 🧠

FaceX 使用了基于深度学习的人脸检测与替换技术，主要包括以下四个核心模块：

---

### 1. 人脸检测算法 👀

人脸检测是换脸过程的第一步。我们使用 **InsightFace** 框架中的 `FaceAnalysis` 模块实现高效准确的人脸定位。

#### 初始化人脸检测器（Python 示例）:

```python
def init_face_analyzer(det_size=(640, 640)):
    # 初始化人脸检测器，使用 GPU
    analyzer = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    analyzer.prepare(ctx_id=0, det_size=det_size)
    return analyzer
```

📌 **功能说明：**
- 基于深度学习模型进行人脸检测。
- 返回人脸的边界框（bounding box）和关键点信息（landmarks）。

---

### 2. 人脸特征提取 📷

在检测到人脸后，我们需要提取其特征表示，用于后续的换脸操作。

#### 加载源图像并提取人脸特征（Python 示例）:

```python
def load_source_face(img_path, analyzer):
    # 加载源图像并提取人脸
    source_img = cv2.imread(img_path)
    if source_img is None:
        raise FileNotFoundError(f"无法加载源图像: {img_path}")
    faces = analyzer.get(source_img)
    if not faces:
        raise ValueError("未在源图像中检测到人脸")
    return faces[0]
```

📌 **功能说明：**
- 提取人脸的特征向量（embedding）。
- 该特征用于后续人脸替换时的匹配与融合。

---

### 3. 人脸替换算法 🔄

核心换脸算法基于 InsightFace 的 `inswapper` 模型实现，负责将源人脸“贴”到目标人脸上。

#### 实现换脸逻辑（Python 示例）:

```python
def swap_faces_in_frame(frame, analyzer, swapper, source_face):
    # 在帧中进行人脸替换
    target_faces = analyzer.get(frame)
    if target_faces:
        target_face = target_faces[0]
        # 使用 GPU 进行人脸替换
        swapped_frame = swapper.get(
            frame, target_face, source_face, paste_back=True
        )
        return swapped_frame
    return frame
```

---

### 4. 平滑处理算法 (FaceX 2.0) 📈

为了提升视频换脸的稳定性，在 FaceX 2.0 中引入了平滑处理机制，减少帧间抖动。

#### 添加平滑处理逻辑（Python 示例）:

```python
# 添加平滑处理
self.face_history = []       # 存储最近的人脸检测结果
self.max_history = 10        # 增加历史记录长度
self.landmark_history = []   # 存储最近的特征点
self.smooth_factor = 0.85    # 增加平滑因子
```

---

### 📌 总结

| 模块             | 技术/工具           | 功能描述                         |
|------------------|---------------------|----------------------------------|
| 人脸检测         | InsightFace          | 定位人脸并提取关键点             |
| 特征提取         | FaceAnalysis         | 获取人脸特征向量                 |
| 换脸算法         | InSwapper 模型       | 替换人脸并融合表情               |
| 平滑处理（2.0） | 历史帧加权平均       | 减少视频帧间抖动，提升视觉稳定性 |

---


## 项目结构 🗂️

```plaintext
FaceX/
├── FaceX1.0.py       # 基础版本
├── FaceX2.0.py       # 增强版本
├── pictures/         # 预设人脸图像
│   ├── img.png
│   ├── img_2.png
│   ├── img_3.png
│   └── img_4.png
└── README.md         # 项目说明文档
```


---

## 许可证 📄

本项目采用 MIT 许可证。

---

## 致谢 🙏

- **InsightFace** 项目提供的深度学习模型
- **PyQt5** 提供的 GUI 开发框架
- 所有测试和提供反馈的用户

---

## 联系方式 📧

如有问题或建议，请通过 [GitHub Issues](https://github.com/Im-Muzzy/FaceX/issues) 提交。

