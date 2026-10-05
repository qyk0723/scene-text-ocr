# 项目交接文档

> 更新日期：2026-09-28
> 仓库：https://github.com/qyk0723/scene-text-ocr （main 分支）

## 一、项目现状

《面向生活场景的文字检测与识别系统》本科毕业设计项目。技术栈 Python 3.11 + PaddleOCR（PP-OCR 预训练）+ OpenCV + Gradio。**不训练模型**，只做应用集成 + ICDAR2015 对比实验。

**阶段 0~4 全部完成**：

- 阶段 0：最小闭环（`SceneTextOCR` 管线 + `main.py` CLI）
- 阶段 1：图像预处理模块（`ImageEnhancer`，8 算子独立开关）
- 阶段 2：Gradio 界面（用户自改的蓝白布局，放大/复制按钮，勾选式预处理）
- 阶段 3：ICDAR2015 全量评估（检测 500 张 + 识别 2074 张）+ 评估报告
- 阶段 4：论文图表（`docs/figures/`：指标、耗时、模型对比、预处理对比、消融、样例、界面截图）

剩余工作：论文写作。

环境：Windows 11，conda 环境 `scene-text`，解释器 `D:\miniconda3\envs\scene-text\python.exe`。

## 二、评估结论速览（详见 docs/evaluation_report.md）

| 指标 | medium（评估基准） | small（系统部署） |
| --- | --- | --- |
| 检测 F1（500 张） | 36.43% | 30.62% |
| 识别字符 / 行级（2074 张） | 91.28% / 80.62% | 89.66% / 77.05% |
| test.jpg 整图耗时（CPU） | 82.0s | 11.1s |

检测 F1 偏低：**主因难例漏检**（IoU 0.3 下行级召回仍仅 44%，约 56% 文字行无检测框），**次因输出粒度介于词/行之间**（IoU 0.3 下词级 P 88.3% vs 行级 P 56.2%）；阈值与单一粒度口径均非主因（报告第五节排除法 + IoU 敏感性分析）。

预处理消融总结论（报告第七节）：清晰图预处理有害（全开 -30 字符点）；算子适用域——小字放大 +55.4（最强，作用于检测侧）、去噪 +16.9（噪声图）、锐化 +5.6（模糊图）、CLAHE 与倾斜校正有害（识别模型对 ±12° 内倾斜有容差）。UI 只暴露三个正收益算子，默认全不勾。

## 三、关键配置与踩坑记录（勿回退）

### 配置源（config.yaml）

模型 / 设备 / 检测阈值 / 数据路径集中在根目录 `config.yaml`，`src/config.py` 加载。命令行参数可覆盖（如 `evaluate.py --model medium`）。

### 推理引擎（detector / recognizer）

检测、识别拆分为独立模块，各用 `paddlex.create_model(模型名, engine_config={"run_mode": "paddle"})`。

| 要点 | 说明 |
| --- | --- |
| `run_mode="paddle"` | 等价旧 `enable_mkldnn=False`：Paddle 3.3 MKLDNN 与 PIR 执行器不兼容，必须关 |
| 文档模型 | 不再加载（UVDoc / 方向分类 / 文本行方向），纯 det+rec，比一体化管线更快 |
| det 返回 | `dt_polys`（检测框）、`dt_scores` |
| rec 返回 | `rec_text`、`rec_score`（注意是**单数**，与一体化管线的 `rec_texts` 不同） |
| 排序/裁剪 | `src/pipeline/geometry.py` 复刻 PaddleOCR 的 `SortQuadBoxes` + `get_minarea_rect_crop` |

### 结果缓存

`SceneTextOCR.run_detailed` 按图片内容 hash 缓存结果（LRU 32 条），同图二次识别 <0.1s。

## 四、已知问题 / 注意事项

1. **OpenCV 冲突**：环境同时装了 `opencv-python==5.0.0.93` 与 `opencv-contrib-python==4.10.0.84`，二者都提供 cv2 会互相覆盖；requirements.txt 只锁 contrib 版。
2. **推理速度**：small 整图 ~11s（test.jpg 39 行）、单行识别 0.111s；medium 整图 ~82s、单行 0.877s。连续多次识别会触发 CPU 热节流（单次 ~43s），间隔测试恢复，非代码问题。
3. **倾斜校正实现局限**：深色背景 Otsu 反相 bug 已修复；残余角度估计偏差源自 minAreaRect 对真实字符分布的拟合，论文按「±12° 内有害、模型自身有容差」如实写。
4. **网络代理**：git 需 Watt Toolkit 代理 `127.0.0.1:26561` + 本仓库 `http.sslVerify=false`（中间人解密）。Watt Toolkit 开系统代理会拦截 localhost 导致 Gradio 启动 404，app.py 已设 `NO_PROXY` 绕过（或 Windows 代理勾「对本地地址不使用代理服务器」）。
5. **token 安全**：旧 PAT 已于 2026-09-28 撤销，git 已配 `credential.helper manager`（凭据存 Windows 凭据管理器，推送不再贴 token）。

## 五、文件地图

| 文件 | 作用 |
| --- | --- |
| `config.yaml` / `src/config.py` | 系统配置 + 加载 |
| `main.py` | CLI 单图识别（模型默认读 config；`--preprocess` 显式全开，仅对比实验用） |
| `app.py` | Gradio 界面（用户所有，改动前先沟通） |
| `evaluate.py` | 评估 CLI（`--task`、`--limit`、`--model medium/small`） |
| `evaluate_ablation.py` | 预处理消融（`--mode clean/degraded/lowcontrast/blur/skew/small`） |
| `evaluate_linelevel.py` | 行级口径检测评估 |
| `make_figures.py` | 论文图表生成（常量改数值后重跑） |
| `src/pipeline/ocr_pipeline.py` | SceneTextOCR 编排（检测→排序→裁剪→识别→缓存） |
| `src/pipeline/geometry.py` | 框排序 + 透视裁剪 |
| `src/detector/` / `src/recognizer/` | TextDetector / TextRecognizer |
| `src/evaluator/` | 评估指标 / 解析 / 执行 |
| `src/preprocess/enhancer.py` | ImageEnhancer（8 算子） |
| `tests/` | 单元测试（纯函数，不加载模型） |
| `docs/evaluation_report.md` | 评估总报告（含 medium vs small、消融） |
| `docs/figures/` | 论文图表 PNG |

## 六、论文写作要点提示

1. 检测指标低：主因难例漏检、次因输出粒度介于词/行之间，用排除法 + IoU 敏感性分析论证（阈值、单一粒度口径均已排除，见报告第五节）。
2. medium vs small 权衡（7.4 倍提速换识别字符 -1.6 点 / 检测 F1 -5.8 点，全量口径；识别两者接近、检测差距明显）。
3. 预处理消融是核心实验亮点：各算子适用域证据表 + 「默认关闭、按需启用」结论。
4. 系统功能（界面截图、可视化样例）用 small 模型输出，与部署一致。
