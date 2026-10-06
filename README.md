# 面向生活场景的文字检测与识别系统

本科毕业设计项目。基于 PaddleOCR（PP-OCR 预训练模型）实现自然场景图片中的文字检测与识别，支持图片上传、结果可视化，并在公开数据集 ICDAR 2015 上完成对比实验。不训练新模型，只做应用集成 + 公开数据集对比实验。

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

（CLI 默认 PP-OCRv6 small 模型，与界面一致；评估基准 medium 由 `evaluate.py` 承担。）

```bash
cd E:\project\scene-text-ocr

# 直接识别
D:/miniconda3/envs/scene-text/python.exe main.py data/samples/test.jpg

# 先预处理再识别（显式全开配置；消融实验显示清晰图下有害，仅作对比实验用）
D:/miniconda3/envs/scene-text/python.exe main.py data/samples/test.jpg --preprocess
```

参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `image` | 输入图片路径（位置参数） | 必填 |
| `--device` | 推理设备 | `cpu`（可传 `gpu`） |
| `--preprocess` | 识别前先做预处理（显式全开，仅对比实验用） | 关闭 |

### Gradio 网页界面

```bash
D:/miniconda3/envs/scene-text/python.exe app.py
```

## 测试

纯函数单元测试（编辑距离、标注解析、框 IoU/匹配、行合并），不加载模型：

```bash
D:/miniconda3/envs/scene-text/python.exe -m unittest discover -s tests -p "test_*.py"
```

## 图像预处理

模块：`src/preprocess/enhancer.py`，类 `ImageEnhancer`。各功能独立开关。

消融实验结论（ICDAR2015 200 张单行图，见 `docs/evaluation_report.md` 第七节）：
- 清晰图：预处理整体有害（全开 -30 字符点），**默认应关闭**
- 算子适用域：去噪（+16.9，噪声图）、锐化（+5.6，模糊图）有效；CLAHE（−13.1）与倾斜校正（−1.3）有害
- ✅ **「小字放大 +55.4」已重做并推翻**：原实验在长边中位数仅 54px 的单行裁剪图上 ÷2 再放大到 400px，
  相对原图净放大 **+641%**，收益无法与「就是喂了张大图」分离。改用**整图 + 四臂 + 分离 det/rec + 部署值 800**
  重做（`evaluate_small_text.py`，100 张）后：放大到 800 只让 det 召回 **+2.9 点**、rec **+0.8 点**，
  **且两项都不如直接把图还原到原尺寸**（0.3589 vs 0.3373 / 0.9007 vs 0.8897）。
  **不应再把它写成"最强的正收益算子"。** 机制：检测器 `limit_side_len=960/max` 只缩不放，
  该算子的真实作用是把低于上限的小图抬到上限附近，一到上限即无额外收益。
  详见 `data/results/small_text_ablation.md` 与报告第七节更正块
- ⚠️ **「倾斜校正 −39.5」是 bug 伪影**：原 `deskew` 的角度估计因 `minAreaRect` 的模 90° 歧义
  而**根本无效**（带符号误差均值 −59.5°，即按无意义角度乱转）。已换成投影轮廓方差法（误差 −0.05°），
  重测后官方消融为**轻微有害 −1.3 点**（受控实验 −1.8 点；识别模型自身对 ±12° 已有容差）

Gradio 界面只暴露正收益算子，默认全不勾：**去噪 / 锐化 / 小字放大**（后者待重做后确认）。

| 功能 | 说明 | 界面默认 |
| --- | --- | --- |
| 小字放大 | 长边不足时放大 | 关（可勾选，收益待重做确认） |
| 去噪 | 中值 / 高斯 | 关（可勾选） |
| 锐化 | 拉普拉斯 | 关（可勾选） |
| 对比度增强 | CLAHE | 不暴露（实测有害 −13.1） |
| 倾斜校正 | 投影轮廓方差法估角 | 不暴露（实测轻微有害 −1.3） |
| 灰度化 | BGR → 灰度 | 关 |
| 二值化 | Otsu / 自适应 | 关 |
| 尺寸归一化 | 长边缩放 | 关 |

## 目录

- `main.py` 命令行单图识别；`app.py` Gradio 网页界面
- `config.yaml` 系统配置（模型 / 推理 / 预处理 / 数据路径）；`src/config.py` 配置加载
- `src/preprocess/` 图像预处理（`ImageEnhancer`，8 算子可开关）
- `src/detector/` 文字检测（`TextDetector`）；`src/recognizer/` 文字识别（`TextRecognizer`）
- `src/pipeline/` 检测 + 识别编排（`SceneTextOCR`，含排序 / 透视裁剪 / 缓存）
- `src/evaluator/` 评估模块（metrics / parsing / runner）
- `evaluate.py` 评估 CLI；`evaluate_ablation.py` 预处理消融；`evaluate_linelevel.py` 行级口径评估
- `make_figures.py` 论文图表生成；`docs/evaluation_report.md` 评估总报告；`docs/figures/` 图表
- `tests/` 单元测试；`data/` 样张、公开数据集、实验结果（已 gitignore）

## 开发进度

- [x] 阶段 0：最小闭环——单图输入，输出识别文本
- [x] 阶段 1：图像预处理模块
- [x] 阶段 2：Gradio 界面
- [x] 阶段 3：ICDAR2015 数据集评估与消融实验
- [x] 阶段 4：实验图表与论文素材
