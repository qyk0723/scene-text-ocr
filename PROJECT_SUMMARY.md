# 项目交接文档

> 更新日期：2026-09-22
> 仓库：https://github.com/qyk0723/scene-text-ocr （main 分支）

## 一、项目现状

《面向生活场景的文字检测与识别系统》本科毕业设计项目。技术栈 Python 3.11 + PaddleOCR（PP-OCR 预训练）+ OpenCV + Gradio。**不训练模型**，只做应用集成 + 公开数据集对比实验。

当前进度：**阶段 0（最小闭环）与阶段 1（图像预处理）已完成并推送 GitHub**。阶段 2（Gradio 界面）未开始。

- 环境：Windows 11，conda 环境 `scene-text`，解释器 `D:\miniconda3\envs\scene-text\python.exe`
- 依赖版本已锁定于 `requirements.txt`
- 数据目录 `data/` 已 gitignore（样张、公开数据集、实验结果均不入库）

## 二、已完成功能

### 阶段 0：最小识别闭环

- `src/pipeline/ocr_pipeline.py`：类 `SceneTextOCR`，懒加载 PaddleOCR，`run(image)` 支持图片路径或 BGR numpy 数组，返回 `(文本框列表, 文本列表, 耗时)`。
- `main.py`：命令行入口，`python main.py <图片> [--lang] [--device] [--preprocess]`，打印检出文本与耗时。

### 阶段 1：图像预处理

- `src/preprocess/enhancer.py`：类 `ImageEnhancer`，7 项预处理，独立开关：
  - 灰度化、去噪（中值/高斯）、对比度增强（CLAHE）、锐化（拉普拉斯）
  - 二值化（Otsu/自适应）、倾斜校正（最小外接矩形旋转）、尺寸归一化（长边缩放）
  - 默认开启：去噪 + 对比度增强 + 锐化
- `main.py --preprocess`：先预处理再识别，分别打印预处理耗时与识别耗时。

## 三、关键参数配置

### PaddleOCR 3.7 初始化参数（已踩坑，勿回退）

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `device` | `"cpu"` | 3.x 已移除 `use_gpu` |
| `enable_mkldnn` | `False` | 3.3 的 MKLDNN 与 PIR 执行器不兼容，开启报 `onednn_instruction.cc` 错 |
| `use_doc_orientation_classify` | `False` | 文档方向分类，场景文字用不到 |
| `use_doc_unwarping` | `False` | UVDoc 文档矫正，场景文字用不到 |
| `use_textline_orientation` | `False` | 文本行方向，与 `use_angle_cls` 互斥；`use_angle_cls` 为旧参数已删除 |

### `ocr()` 返回格式（3.x 与 2.x 不同）

3.x 的 `ocr()` 返回字典列表：`raw[0]` 是 dict，文本框取 `raw[0]["dt_polys"]`，文本取 `raw[0]["rec_texts"]`。不再是 2.x 的 `[box, (text, score)]` 嵌套列表。

### 图像预处理默认项

`ImageEnhancer()` 默认：`denoise=True`（medianBlur 3×3）、`contrast=True`（CLAHE 2.0/8×8）、`sharpen=True`，其余关。

## 四、已知问题 / 注意事项

1. **OpenCV 冲突**：环境里同时装了 `opencv-python==5.0.0.93` 和 `opencv-contrib-python==4.10.0.84`，二者都提供 `cv2` 会互相覆盖。PaddleOCR 的 `OCRResult` 依赖 contrib 版，`requirements.txt` 只锁定 contrib 版，勿再装 opencv-python。
2. **CPU 推理慢**：39 行文本约 68–72s，主要耗时在 det+rec 推理，文档模型已关（只省 ~3.5s）。后续如需提速可换 PP-OCRv6 mobile 模型或上 GPU。
3. **网络代理**：本机 GitHub 直连被墙，git 需走 Watt Toolkit 本地代理 `127.0.0.1:26561`（`http.proxy`/`https.proxy` 已设），且因中间人解密需 `http.sslVerify=false`（仅本仓库）。关掉 Watt Toolkit 后需 `git config http.sslVerify true` 恢复。
4. **token 安全**：推送用过的两个 PAT 已明文暴露，建议尽快撤销换新。

## 五、下一步计划（阶段 2：Gradio 界面）

1. `src/ui/app.py`：Gradio 界面，图片上传 + 预处理开关 + 识别结果可视化（文本框画在图上）+ 耗时展示。
2. 复用 `SceneTextOCR` 与 `ImageEnhancer`，不重复造轮子。
3. 结果可视化可用 `raw[0]["rec_polys"]` + `rec_texts` 画框，或直接用 `OCRResult._to_img()`（需 opencv-contrib-python）。

## 六、阶段 3 / 4 预告

- 阶段 3：CTW1500 数据集批量评估（`src/evaluator/`），检测 + 识别指标，消融实验（开关各预处理项对比）。
- 阶段 4：实验图表（matplotlib）+ 论文素材。
