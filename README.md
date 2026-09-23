# 面向生活场景的文字检测与识别系统

本科毕业设计项目。基于 PaddleOCR（PP-OCR 预训练模型）实现自然场景图片中的文字检测与识别，支持图片上传、结果可视化，并在公开数据集（CTW1500 / ICDAR 2015）上完成对比实验。不训练新模型，只做应用集成 + 公开数据集对比实验。

## 环境

- Python 3.11（conda 环境 `scene-text`，解释器 `D:\miniconda3\envs\scene-text\python.exe`）
- PaddleOCR 3.7.0 / PaddlePaddle 3.3.1（CPU）/ OpenCV 4.10（opencv-contrib-python）/ Gradio 6.28
- 版本全部锁定，见 `requirements.txt`

## 安装

```bash
D:/miniconda3/envs/scene-text/python.exe -m pip install -r requirements.txt
```

首次运行会自动下载 PP-OCR 预训练模型到 `~/.paddlex`，需联网。

## 使用方式

### 命令行单图识别

```bash
cd E:\project\毕业设计\scene-text-ocr

# 直接识别
D:/miniconda3/envs/scene-text/python.exe main.py data/samples/test.jpg

# 先预处理再识别（去噪 + 对比度增强 + 锐化）
D:/miniconda3/envs/scene-text/python.exe main.py data/samples/test.jpg --preprocess
```

参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `image` | 输入图片路径（位置参数） | 必填 |
| `--lang` | 识别语言 | `ch` |
| `--device` | 推理设备 | `cpu`（可传 `gpu`） |
| `--preprocess` | 识别前先做图像预处理 | 关闭 |

### Gradio 网页界面（阶段 2 开发中）

```bash
D:/miniconda3/envs/scene-text/python.exe app.py
```

## 图像预处理

模块：`src/preprocess/enhancer.py`，类 `ImageEnhancer`。各功能独立开关，默认开启「去噪 + 对比度增强 + 锐化」。

| 功能 | 说明 | 默认 |
| --- | --- | --- |
| 灰度化 | BGR → 灰度 | 关 |
| 去噪 | 中值 / 高斯 | 开 |
| 对比度增强 | CLAHE | 开 |
| 锐化 | 拉普拉斯 | 开 |
| 二值化 | Otsu / 自适应 | 关 |
| 倾斜校正 | 最小外接矩形旋转 | 关 |
| 尺寸归一化 | 长边缩放 | 关 |

## 目录

- `src/preprocess/` 图像预处理（去噪、纠偏、增强，可开关用于消融实验）
- `src/detector/` 文字检测（预留）
- `src/recognizer/` 文字识别（预留）
- `src/pipeline/` 完整识别管线（`SceneTextOCR` 封装）
- `src/evaluator/` 数据集批量评估与指标计算（预留）
- `src/ui/` Gradio 网页界面（预留）
- `data/` 样张、公开数据集、实验结果（已 gitignore）

## 开发进度

- [x] 阶段 0：最小闭环——单图输入，输出识别文本
- [x] 阶段 1：图像预处理模块
- [ ] 阶段 2：Gradio 界面
- [ ] 阶段 3：CTW1500 数据集评估与消融实验
- [ ] 阶段 4：实验图表与论文素材
