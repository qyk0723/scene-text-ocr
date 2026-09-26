# 项目交接文档

> 更新日期：2026-09-26
> 仓库：https://github.com/qyk0723/scene-text-ocr （main 分支）

## 一、项目现状

《面向生活场景的文字检测与识别系统》本科毕业设计项目。技术栈 Python 3.11 + PaddleOCR（PP-OCR 预训练）+ OpenCV + Gradio。**不训练模型**，只做应用集成 + ICDAR2015 对比实验。

**阶段 0~4 全部完成**：

- 阶段 0：最小闭环（`SceneTextOCR` 管线 + `main.py` CLI）
- 阶段 1：图像预处理模块（`ImageEnhancer`，8 算子独立开关）
- 阶段 2：Gradio 界面（用户自改的蓝白布局，放大/复制按钮，勾选式预处理）
- 阶段 3：ICDAR2015 全量评估（检测 500 张 + 识别 2074 张）+ 评估报告
- 阶段 4：论文图表（`docs/figures/`：指标、耗时、模型对比、预处理对比、消融、样例、界面截图）

剩余工作：论文写作 + 用户重截 `gradio_ui.png`（旧图状态栏耗时是 medium 的，与现系统 small 不一致）。

环境：Windows 11，conda 环境 `scene-text`，解释器 `D:\miniconda3\envs\scene-text\python.exe`。

## 二、评估结论速览（详见 docs/evaluation_report.md）

| 指标 | medium（评估基准） | small（系统部署） |
| --- | --- | --- |
| 检测 F1（500 张） | 36.84% | 32.70% |
| 识别字符 / 行级（2074 张） | 82.89% / 71.50% | 75.86% / 63.69% |
| test.jpg 整图耗时（CPU） | 115.6s | 21.7s |

检测召回率偏低的原因（已定性，非阈值问题）：**词级 GT vs 行级检测**的粒度错配。

预处理消融总结论（报告第七节）：清晰图预处理有害（全开 -20 字符点）；算子适用域——小字放大 +44.4（最强）、去噪 +15.7（噪声图）、锐化 +3.7（模糊图）、CLAHE 中性、倾斜校正 ±12° 内有害（模型自身有容差）。UI 只暴露三个正收益算子，默认全不勾。

## 三、关键参数配置（踩坑记录，勿回退）

### PaddleOCR 3.7 初始化（`src/pipeline/ocr_pipeline.py`）

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `device` | `"cpu"` | 3.x 已移除 `use_gpu` |
| `enable_mkldnn` | `False` | Paddle 3.3 MKLDNN 与 PIR 执行器不兼容 |
| `use_doc_orientation_classify` / `use_doc_unwarping` / `use_textline_orientation` | `False` | 文档模型，场景文字用不到；`use_angle_cls` 与 `use_textline_orientation` 互斥且已废弃 |
| `text_detection_model_name` / `text_recognition_model_name` | UI 传 small，评估默认 medium | v6 无 mobile 档，档位名 small/tiny |
| `text_recognition_batch_size` | 默认不传 | 实测 CPU 拼批无提速 |

### `ocr()` 返回格式（3.x）

字典列表：`raw[0]["dt_polys"]`（检测框）、`raw[0]["rec_texts"]`、`raw[0]["rec_scores"]`。

### 结果缓存

`SceneTextOCR.run_detailed` 按图片内容 hash 缓存结果（LRU 32 条），同图二次识别 <0.1s。UI 与 evaluate.py 共用。

## 四、已知问题 / 注意事项

1. **OpenCV 冲突**：环境同时装了 `opencv-python==5.0.0.93` 与 `opencv-contrib-python==4.10.0.84`，二者都提供 cv2 会互相覆盖；requirements.txt 只锁 contrib 版。
2. **推理速度**：small 模型整图 ~22s（test.jpg 39 行），清晰图识别 0.536s/行。历史基线 medium 68.9s 在 9-25 后变 115s 未完全查明（重启后 115s，电源已确认高性能），评估数值不受影响（指标与速度无关）。
3. **倾斜校正实现局限**：深色背景 Otsu 反相 bug 已修复；残余角度估计偏差源自 minAreaRect 对真实字符分布的拟合，论文按「±12° 内有害、模型自身有容差」如实写。
4. **网络代理**：git 需 Watt Toolkit 代理 `127.0.0.1:26561` + 本仓库 `http.sslVerify=false`（中间人解密）。关代理后恢复 `git config http.sslVerify true`。
5. **token 安全**：推送用过的 PAT 已明文暴露过，建议撤销换新。

## 五、文件地图

| 文件 | 作用 |
| --- | --- |
| `main.py` | CLI 单图识别（默认 small 模型，与界面一致；`--preprocess` 显式全开，仅对比实验用） |
| `app.py` | Gradio 界面（用户所有，改动前先沟通） |
| `evaluate.py` | ICDAR2015 检测/识别评估（`--task det/rec`、`--limit`、`--ignore-case`） |
| `evaluate_ablation.py` | 预处理消融（`--mode clean/degraded/lowcontrast/blur/skew/small`） |
| `make_figures.py` | 论文图表生成（常量改数值后重跑） |
| `src/pipeline/ocr_pipeline.py` | SceneTextOCR（模型名/阈值/批大小/缓存） |
| `src/preprocess/enhancer.py` | ImageEnhancer（8 算子） |
| `docs/evaluation_report.md` | 评估总报告（含 medium vs small、消融） |
| `docs/figures/` | 论文图表 PNG |

## 六、论文写作要点提示

1. 检测指标低是「词级 GT vs 行级检测」结构性原因，有阈值实验佐证。
2. medium vs small 速度/精度权衡（5.3 倍提速换 -1.3 字符点 / -5 F1 点）。
3. 预处理消融是核心实验亮点：各算子适用域证据表 + 「默认关闭、按需启用」结论。
4. 系统功能（界面截图、可视化样例）用 small 模型输出，与部署一致。
