# 面向生活场景的文字检测与识别系统

本科毕业设计项目。基于 PaddleOCR（PP-OCR 预训练模型）实现自然场景图片中的文字检测与识别，支持图片上传、摄像头采集、结果可视化，并在公开数据集（CTW1500 / ICDAR 2015）上完成对比实验。

## 环境

- Python 3.11（conda 环境 `scene-text`，位于 `D:\miniconda3\envs\scene-text`）
- PaddleOCR 3.7.0 / PaddlePaddle 3.3.1（CPU）/ OpenCV 4.10 / Gradio 6.28

## 使用方式

在 VSCode 中选择解释器 `D:\miniconda3\envs\scene-text\python.exe` 后运行：

```bash
python src/ui/app.py          # 启动 Gradio 网页界面
python main.py --image xxx.jpg   # 命令行单图识别
```

## 目录

- `src/preprocess/` 图像预处理（去噪、纠偏、增强，可开关用于消融实验）
- `src/detector/` 文字检测（封装 PaddleOCR 检测模块）
- `src/recognizer/` 文字识别（封装 PaddleOCR 识别模块）
- `src/pipeline/` 完整识别管线
- `src/evaluator/` 数据集批量评估与指标计算
- `src/ui/` Gradio 网页界面
- `data/` 样张、公开数据集、实验结果

## 待办（开发阶段）

- [ ] 阶段 0：最小闭环——单图输入，输出识别文本
- [ ] 阶段 1：图像预处理模块
- [ ] 阶段 2：Gradio 界面
- [ ] 阶段 3：CTW1500 数据集评估与消融实验
- [ ] 阶段 4：实验图表与论文素材
