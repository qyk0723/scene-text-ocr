# 数据集来源与溯源

> 建立日期：2026-10-07（审计 §待办：系统完善 的 A4 项）
> 目的：ICDAR2015 是**外部第三方数据**，本科毕设正文/附录需要写清"数据从哪来、怎么核对"。
> 本文只记录**可核实的事实**；无法核实的部分显式标注为"待确认"，不代填。
> 注：本文件位于 `data/` 下（该目录被 `.gitignore` 的 `data/*` 排除），
> 属于**显式入库的证据文件**，需要 `git add -f` 才能提交。

## 一、ICDAR2015（本项目唯一的评估数据集）

| 项 | 内容 |
| --- | --- |
| 全称 | ICDAR 2015 Robust Reading Competition — Challenge 4: **Incidental Scene Text** |
| 官方站点 | <https://rrc.cvc.uab.es/?ch=4>（Robust Reading Competition 门户） |
| 竞赛总览 | <https://rrc.cvc.uab.es/files/Robust_Reading_2015_v02.pdf> |
| 应引用的论文 | Karatzas et al., *ICDAR 2015 competition on Robust Reading*, ICDAR 2015, DOI [10.1109/ICDAR.2015.7333942](https://doi.org/10.1109/ICDAR.2015.7333942) |
| 本地路径 | `data/icdar2015/`（**已被 `.gitignore` 的 `data/*` 排除，不入库**） |
| 用途 | 检测 500 张全量评测 + 识别 2074 张行级评测；预处理消融的 200 张单行图取自识别集 |

### 本地内容与规模（2026-10-07 实测，可复现）

| 子集 | 内容 | 计数 |
| --- | --- | --- |
| `detection/test/` | 评估用检测集（imgs + gt） | 500 个标注文件、**5230 个文本框** |
| `detection/test/gt/*.txt` | 标注 | `###` 无效区（do-not-care）**3153** 个 / 真实文本 **2077** 个 |
| `detection/train/` | 训练集 | 2000 个文件，**当前代码零引用**（审计 P2-23：留作 P1-12 的留出集或归档） |
| `recognition/test.txt` | 识别清单 | 2074 行 |
| `recognition/test/` | 单行裁剪图 | 供 `evaluate.py --task rec` 与消融使用 |

### 原版性核对（说明"这就是官方原版，不是被改写过的版本"）

审计 P0-1 曾担心本地 GT 被"剥离成统一 `###`"。实测**并非如此**，证据两条：

1. **两类标注并存**：5230 个框里 3153 个是 `###`、2077 个是真实转写文本
   （`Please`、`residential` 等）。若被统一剥离，不会还有 2077 条真实文本。
2. **保留了官方著名的拼写错误**：`gt_img_100.txt` 内含 `Refishing`
   （官方原版特征，社区广泛引用）——实测为 `True`。

复现命令（项目根目录）：

```powershell
D:\miniconda3\envs\scene-text\python.exe -c "
from pathlib import Path
gt = Path('data/icdar2015/detection/test/gt')
dnc = real = 0
for f in gt.glob('*.txt'):
    for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
        p = line.split(',')
        if len(p) >= 9:
            dnc += p[8].strip() == '###'; real += p[8].strip() != '###'
print('do-not-care', dnc, '/ real', real)
print('Refishing in gt_img_100:', 'Refishing' in (gt/'gt_img_100.txt').read_text(encoding='utf-8', errors='replace'))
"
```

### ⚠️ 使用条款与署名（**待人工确认，勿凭空写**）

- 官方站点对数据下载有**使用条款/注册流程**；ICDAR 系数据集的通行约束是
  **仅限学术研究、不得再分发**，但**具体措辞以下载时同意的条款为准**——
  本项目没有保留下载记录（无日期、无当时的条款快照），因此本文**不代填许可原文**。
- **论文里建议的写法**（只写已核实的事实）：注明数据名称与版本（ICDAR2015 Incidental
  Scene Text）、给出官方站点与本文件所述的引用论文，并说明"数据未随代码分发"。
- **不要在论文中声称具体许可措辞**，除非能找回当时的下载条款页面。
- 若需再分发或商用，必须回到官方站点重新确认条款。

## 二、被删除的无关文件（2026-10-07，审计 P2-23）

`data/icdar2015/detection/{train,test}.json`（11.6 MB + 5.1 MB）已删除：

- 内容是 PaddleX 数据集清单，`data_root` 指向 **`D:\dataset\icdar2015\...`**，
  **该路径在本机已不存在**（实测 `Test-Path D:\dataset` 为 False）；
- 全仓库代码 **零引用**（grep `train.json`/`test.json` 无命中，仅审计文档提到）；
- 属孤儿文件，删除不影响任何评测（`data/*` 本就不入库）。

保留未动的项（审计原文建议与事实不符，已在 `docs/PROJECT_AUDIT.md` 更正）：

| 对象 | 审计原文建议 | 实际处理 |
| --- | --- | --- |
| `data/ctw1500/` | "删除或真的启用" | **该目录已不存在**（本轮实测），无需处理 |
| `.idea/` `.vscode/` `.claude/` | "提交前清掉" | **已被 `.gitignore` 忽略**（不会入库），且是开发者本地配置；按用户决定**保留** |
| `detection/train/` | "留作留出集或归档" | 保留（未来若做留出集评估会用到） |