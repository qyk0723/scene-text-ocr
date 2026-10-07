# P1-8 medium 档基线：100 张 deteval（medium@960/max，与 1280/max 同批同序）

- 样本：--limit 100 --sample-seed 20261006（与 small A/B/C、medium@1280 同一批图）
- 模型：PP-OCRv6 **medium** det；**模型自带默认** 960/max（未覆写长边限制）
- 工具：	ools/eval_det_resumable.py（可续跑，逐张检查点，口径与 evaluate.py 同源）

| 指标 | 值 |
| --- | --- |
| deteval 精确率 | 0.5160 |
| deteval 召回率 | 0.4617 |
| **deteval F1** | **0.4874** |
| TP / FP / FN | 193 / 181 / 225 |
| 平均耗时 | 11.129 s/张（**空闲机器**） |

⚠️ 该 100 张子集对 medium 偏乐观：0.4874 vs 已发表 500 张全量 0.4672（+2.02 点）。
因此只可用于**同一样本内**与 1280/max 比较；引用绝对水平请用 500 张全量。
