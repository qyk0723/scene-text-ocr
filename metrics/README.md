# metrics/ —— 图表数值的单一数据源

这里存放 `docs/figures/` 全部图表的**数值来源**。绘图脚本 `make_figures.py`
不再包含任何硬编码数字，而是从这里读取；缺键会立刻抛出带路径的错误。

**为什么要有这个目录**：审计（`docs/PROJECT_AUDIT.md` 的 P1-16）发现绘图脚本把每个
数字写成了字面量，与 `data/results/` 的产物没有任何程序化关联。后果已经真实发生——
同一个 small 检测 F1，图表里是 30.62、另一个产物文件里是 23.53，**没有任何机制能发现**。

## 重新生成

```bash
python tools/gen_metrics.py
```

该脚本把可解析的数值**从产物中解出**（不手工转录）。生成后可用下面的方式自查：
重新运行 `make_figures.py` 的相关函数，产出的 PNG 应与仓库中的版本一致
（本目录首次建立时已验证：10 张图里 9 张逐字节相同，唯一差异见下）。

## 文件与来源

| 文件 | 内容 | 来源产物 |
| --- | --- | --- |
| `detection.json` | medium / small 的检测 P·R·F1，**两套口径**（legacy 与 deteval） | `eval_medium_det_final.md`、`eval_small_det_deteval.md` |
| `recognition.json` | medium / small 的识别字符·行级准确率与单行耗时 | `eval_medium_final.md`、`eval_small_v2.md` |
| `linelevel.json` | 词级/行级 F1 × IoU 0.3/0.5/0.7，medium 与 small | `linelevel_medium_iou.md`、`linelevel_small_iou.md` |
| `ablation.json` | 6 种图像条件下各预处理配置的准确率、检出框数、耗时 | `ablation_*.md` |
| `timing.json` | 有产物支撑的耗时（识别单行、检测整图） | 同 `recognition` / `detection` |
| `manual.json` | **没有任何产物支撑**的历史数值，见下 | 无 |

`detection.json` 的 small 项带一个 `legacy_crosscheck` 字段：它来自**另一个独立产物**
（`eval_small_v2.md`），与主来源数值逐位相同——用于交叉验证解析没有取错行。

## ⚠️ `manual.json`：这些数字没有证据

`manual.json` 里的数值是历史手工记录，**审计已指出它们不可复现**：

- `preprocess_whole_image_s` = 0.10：仅存在于绘图脚本的历史字面量中，无任何产物。
- `test_jpg_end_to_end_s`：medium 82.0 / small 11.1，同样无产物；
  而同一操作实测见过 **11.1s 与 29.0s**，波动极大。

**在重测之前，不要把这三个数当作精确值写进论文。** 它们被集中放在这里，就是为了
让"哪些数字没有证据"一眼可见，而不是埋在绘图代码里。

## 检测指标口径（重要）

`detection.json` 同时保存两套口径：

- **legacy**：全部 5230 个 GT 框计入召回分母（仓库历史口径，**不可与已发表结果比较**）；
- **deteval**：转写为 `###` 的 3153 个框视为 do-not-care，不计入分母、命中它的预测被忽略
  （ICDAR2015 官方做法）。

`make_figures.py` 顶部有一个开关：

```python
DETECTION_PROTOCOL = "legacy"   # 改成 "deteval" 即切换全部检测图的口径
```

论文最终采用哪一套由 `docs/PROJECT_AUDIT.md` 的 P0-1 决定；切换只需改这一行。

## 已知差异说明

**`timing.png` 与仓库旧版本的差异**：旧版绘图脚本写的是四舍五入后的 `2.78`，
现在直接从产物读得 `2.779`。两者渲染出来**只有 412 个像素不同（占 0.031%）**，
是柱状图边缘的亚像素差异，不是数据变化。保留产物原值以保证"图由数据生成"可复现。

**`linelevel.json` 的子集不是随机的**：它来自"字典序前 30 张"，审计实测该子集文本密度
偏高（行数处于全量重采样分布的第 94.6 百分位）。口径本身正确，但子集代表性偏弱。
