# P1-8 medium 档：100 张 deteval 复测（medium@1280/max vs 已发表 medium@960/max）

- 样本：--limit 100 --sample-seed 20261006（与 small 的 A/B/C 档同一批图、同一顺序）
- 模型：PP-OCRv6 **medium** det；limit_side_len=1280、limit_type=max（不下采样）
- **100/100 张**完成（第 1–44 张由 evaluate.py 跑，第 45–100 张由
  	ools/eval_det_resumable.py 可续跑脚本补齐——原因见该脚本文档头：
  该轮 medium 跑到第 44 张被外部杀掉且 evaluate.py 只在全跑完时写报告）

| 指标 | 值 |
| --- | --- |
| deteval 精确率 | 0.5013 |
| deteval 召回率 | 0.4545 |
| **deteval F1** | **0.4768** |
| TP / FP / FN | 190 / 189 / 228 |

> 口径与 evaluate.py 严格一致：同一套 parse_det_gt_flagged + match_boxes +
> match_boxes_dnc + prf_from_counts，IoU 0.5 贪心一对一，### 为 do-not-care。
