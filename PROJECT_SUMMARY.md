# 项目交接文档

> 更新日期：2026-10-06
> 仓库：https://github.com/qyk0723/scene-text-ocr （main 分支）

## 一、项目现状

《面向生活场景的文字检测与识别系统》本科毕业设计项目。技术栈 Python 3.11 + PaddleOCR（PP-OCR 预训练）+ OpenCV + Gradio。**不训练模型**，只做应用集成 + ICDAR2015 对比实验。

环境：Windows 11，conda 环境 `scene-text`，解释器 `D:\miniconda3\envs\scene-text\python.exe`。

### 1.1 阶段进度

功能阶段 0~4 均已完成：

- 阶段 0：最小闭环（`SceneTextOCR` 管线 + `main.py` CLI）
- 阶段 1：图像预处理模块（`ImageEnhancer`，8 算子独立开关）
- 阶段 2：Gradio 界面（用户自改的蓝白布局，放大/复制按钮、勾选式预处理）
- 阶段 3：ICDAR2015 全量评估（检测 500 张 + 识别 2074 张）+ 评估报告
- 阶段 4：论文图表（`docs/figures/`）

### 1.2 ⚠️ 2026-10-06 审计后的真实状态（**接手请先读这一节**）

2026-10-06 做了一次全项目审计（**`docs/PROJECT_AUDIT.md`**，31 项问题 + 逐轮实测记录），
结论是**"阶段 0~4 完成、只剩写作"这个判断过于乐观**：代码里有多处真实缺陷，
且若干已发表数字建立在**非标准评测口径**上。审计在 `main` 上已落实的改进：

| 类别 | 已完成 |
| --- | --- |
| **真实缺陷修复** | 非 ASCII 路径读图失败（`main.py` 传绝对路径必失败）、`crop_box` 退化框静默返回整图、`render_report` 硬编码 IoU、报告标签把 det-only 标成 det+rec、`preprocess_compare` 的 no-op 代码回归 |
| **评测口径** | 新增 ICDAR2015 官方 **do-not-care 双口径**（legacy/deteval 并存，一次推理同时算出）；`evaluate.py` 报告两套都报，**以 deteval 为主** |
| **新增指标** | **端到端（整图）系统指标** `evaluate_end2end.py`（词级+行级）+ `--dump-lines` 错误分类；`--sample-seed` 随机抽样 |
| **可复现性** | `--dump-pred` 预测框+置信度落盘；图表数值改为从 **`metrics/`** 单一数据源读取（`tools/gen_metrics.py` 从产物解析生成，缺键报错）；统一 UTF-8 输出；消融耗时口径修正 |
| **环境** | 卸载 `opencv-python`，只留 `opencv-contrib-python==4.10.0.84` |
| **测试** | 单测 23 → **49 项**，全部通过 |

**✅ medium 的 deteval 全量重跑已完成（2026-10-06 23:11，500 张，耗时约 3 小时）**：
**deteval P 0.5115 / R 0.4299 / F1 0.4672**；同一次运行算出的 legacy 为 0.6137/0.2591/0.3643，
与库内旧产物 `eval_medium_det_final.md` **逐位相同**（可复现性交叉验证）。
两张检测类图表（`detection_metrics.png`、`model_compare_accuracy.png`）已重新生成并核对过数值。

**尚未完成、且需要你决定或授权的**：

1. `docs/evaluation_report.md` §五 的结论按 deteval 重写（medium 数字已就位，可以动笔了）。
2. **P1-3 预处理消融重做**：脚本 `evaluate_small_text.py` 已写好并冒烟通过
   （整图、四臂、分离 det-only/rec-only、用部署值 800），**尚未跑正式实验**——
   它需要独占 CPU 才能让耗时列可信。同时 `ablation_skew.md` 需用修复后的 deskew 重跑。
3. 审计列出的其余事项：结果产物入库（P1-15，**已完成**，见 `.gitignore`）、
   `deskew` 修复（P2-19，**已完成**）、日志编码（P2-20，**已完成**）。

**剩余工作**：论文写作 + 上述第 1、2 项。

## 二、评估结论速览（详见 docs/evaluation_report.md）

**检测按「两套口径都报、deteval 为主」呈现**（deteval = ICDAR2015 官方做法，`###` 的 do-not-care 框不计入召回分母）：

| 指标 | medium（评估基准） | small（系统部署） |
| --- | --- | --- |
| **检测 F1（deteval，官方口径 / 论文主口径）** | **46.72%** | **42.21%** |
| 检测 P / R（deteval） | 51.15% / 42.99% | 48.18% / 37.55% |
| 检测 F1（legacy，历史口径，不可对外比较） | 36.43% | 30.62% |
| 识别字符 / 行级（2074 张） | 91.28% / 80.62% | 89.66% / 77.05% |
| test.jpg 整图耗时（CPU） | 82.0s | 11.1s |

> ⚠️ **口径影响（small 全量 500 张实测）**：换用官方口径后 **F1 30.62% → 42.21%（+11.6 点）、
> 召回 20.98% → 37.55%（近乎翻倍）**；medium 同向（36.43% → 46.72%，+10.3 点）。
> ICDAR2015 检测集的 5230 个 GT 框中 **3153 个（60.3%）转写为 `###`**。
> 因此 legacy 口径的检测指标**不可与已发表 IC15 结果比较**，只能作历史对照。
> 另：即便用 deteval，本项目的匹配方案（IoU≥0.5 贪心一对一）与官方 deteval
> （面积重叠 + 显式忽略 + 一对多 + H-mean）仍不同，**仍不可逐字对比**。
>
> medium 的 legacy 数字与库内旧产物 `eval_medium_det_final.md` **逐位相同**，
> 说明该配置的检测结果可复现（但耗时不可复现：本次 13.555 s/张 vs 旧记录 11.26 s/张）。

**检测 F1 偏低的成因（按官方口径重述）**：
1. **难例漏检确实存在，但远没有 legacy 口径显示的严重**——排除 do-not-care 后，同口径行级召回上界是 **82.8%** 而非 41.6%。原文"约 58% 文字行无检测框"中的"文字行"有 **71.5% 其实完全由 do-not-care 区域构成**。
2. **输出粒度介于词/行之间（现有因果证据）**：把 `unclip_ratio` 从 1.4 调到 1.0（框更紧）F1 上升、调到 1.8（框更松）精确率与召回率**同时**下降，两个口径下单调次序一致。但 100 张子集上单个差值**未达显著**，论文只能写趋势。
3. **阈值/参数不是主因**（三组独立证据）：`box_thresh` 0.35–0.7 波动 ±2 点、`thresh` 0.15/0.30 ∓1 点、`unclip_ratio` +1.8/−5.8 点；而**换口径是 +11.6 点**。

> 报告第五节的原文结论基于 legacy 口径，**已加口径更正块标注需重写**。详见 `docs/evaluation_report.md` 与 `docs/PROJECT_AUDIT.md`。

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

1. **OpenCV 冲突（已解决，2026-10-06）**：环境曾同时装了 `opencv-python==5.0.0.93` 与 `opencv-contrib-python==4.10.0.84`。两个包**提供同一个 `cv2` 目录**，后装的会覆盖先装的，不报错、只是静默换版本——属未锁定环境的隐患。
   **现已卸载 `opencv-python`，只保留 `opencv-contrib-python==4.10.0.84`**（与 requirements.txt 一致）。校验：`cv2.__version__ == 4.10.0`、CLAHE 与 `intersectConvexConvex` 等功能正常、42 项单测通过、真实推理正常。
   ⚠️ **操作提醒**：直接 `pip uninstall opencv-python` **会带走 contrib 也需要的共享文件**，导致 `cv2` 损坏（表现为 `module 'cv2' has no attribute '__version__'`）。必须随后执行
   `pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84` 修复。本次即如此处理。
   **切勿再安装 `opencv-python`**。
2. **推理速度**：small 整图 ~11s（test.jpg 39 行）、单行识别 0.111s；medium 整图 ~82s、单行 0.877s。**但耗时波动极大**——同一操作（small 整图 test.jpg）实测见过 11.1s 与 29.0s 的差异，39 张单行图串行也见过 3.89s 与 5.27s（差 35%）。引用耗时数字前应预热 + 重复取中位数；连续多次识别会触发 CPU 热节流，非代码问题。
3. **倾斜校正（实现已修，结论也变了）**：深色背景 Otsu 反相 bug 早已修复；更根本的是
   **角度估计原本完全无效**——`cv2.minAreaRect` 的角度按"宽高谁更长"而模 90° 歧义，
   实测带符号误差均值 **−59.5°**，即按无意义角度乱转。已换成投影轮廓方差法（误差 −0.05°）。
   重测后结论从「−39.5 点有害」改为「**轻微有害 −1.8 点，且识别模型自身对 ±12° 已有容差**」，
   论文按后者写。详见 `docs/PROJECT_AUDIT.md` P2-19 与 `docs/evaluation_report.md` 第七节更正块。
   ⚠️ **「小字放大 +55.4」已重做并推翻**（`evaluate_small_text.py`，整图/四臂/det-rec 分离/部署值 800）：
   放大到 800 只让 det 召回 +2.9 点、rec +0.8 点，**且两项都不如直接把图还原到原尺寸**——
   不应再写成"最强正收益算子"。见 `data/results/small_text_ablation.md`。
   P1-3 与 P2-19 至此均已闭环。
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
| `docs/evaluation_report.md` | 评估总报告（含 medium vs small、消融、端到端 §3.1/§5.1/§5.2） |
| `docs/figures/` | 论文图表 PNG |
| `docs/PROJECT_AUDIT.md` | **项目审计与待办清单**（31 项 + 逐轮实测记录 + 自纠） |
| `docs/literature_comparison.md` | 文献对比表模板与填写规范 |
| `evaluate_end2end.py` | **端到端（整图）系统评测** CLI |
| `metrics/` | **图表数值的单一数据源**（JSON，由 `tools/gen_metrics.py` 从产物解析生成） |
| `tools/gen_metrics.py` | 从 `data/results/` 产物生成 `metrics/*.json` |
| `tools/check_consistency.py` | **一致性校验**：文档数字 ↔ `metrics/` ↔ 产物（改完数字请跑一次） |

## 六、论文写作要点提示

> 2026-10-06 更新：检测已改为「**两套口径都报、deteval（官方口径）为主**」，
> 下列要点已按新口径重述。

1. **检测指标的论述方式需要改**：原「主因难例漏检、次因粒度」的论证建立在 legacy 口径上，
   证据强度被高估——排除 do-not-care 后行级召回上界是 **82.8%** 而非 41.6%。
   更强的写法是三条实测证据：
   - **端到端错误分类**：漏读的 129 行里 **61% 是「有框但 IoU 未达 0.5」**（41.9% 落在 0.3–0.5 区间），
     即**检出了但几何/粒度对不上**，不是模型没看见；真漏检只占 39%。
   - **粒度有了因果证据**：`unclip_ratio` 1.4→1.0 时 F1 升、1.4→1.8 时精确率与召回率**同时**下降，
     两个口径下单调次序一致（但 100 张上单个差值未达显著，只能写趋势）。
   - **分层曲线**：召回率随框高强烈单调（<15px 仅 **14.4%**，≥60px 达 **47.9%**），
     结合"输入被缩到 0.75 倍"构成完整链条。
   ⚠️ 反面必须写：假阳性里只有 28% 是 <25px 的碎框，**72% 来自 ≥25px 的框**，不能说"只在小字上出错"。
2. medium vs small 权衡（**deteval 口径**）：small 换来约 **4.9×** 检测提速
   （13.555 → 2.779 s/张），代价是识别字符准确率 **−1.6 点**（91.28% → 89.66%）、
   检测 F1 **−4.5 点**（46.72% → 42.21%）。**即识别几乎不损失、检测损失明显** ——
   与「瓶颈在检测」的端到端结论一致，是选 small 部署的核心理由。
   （legacy 口径下检测 F1 差 −5.8 点，方向相同、幅度略大。）
3. 预处理消融是核心实验亮点：各算子适用域证据表 + 「默认关闭、按需启用」。
   ⚠️ 引用耗时列时**必须与「检出框数」一起看**——每个检出框都要跑一次识别，
   一个把检测弄坏的配置（如清晰图下的"全开"）会因为框变少而显得很快。见 `evaluate_ablation.py` 的口径说明。
4. 系统功能（界面截图、可视化样例）用 small 模型输出，与部署一致。
5. **文献对比表**：见 `docs/literature_comparison.md`（含列定义、核对清单与候选行）。
   本项目是**零样本** + 匹配方案与官方 deteval 不同，**只能作量级参照、不可逐项对比**。
6. **端到端系统指标**（新增，最能反映用户体验）：100 张子集上行级 F1 **33.59%**、词级 29.82%，
   而 rec-only 是 **89.66%** 字符准确率 → **瓶颈在检测，不在识别**。见报告 §3.1 与 §5.2。
