# 统一的运行入口：强制 UTF-8，避免日志出现混合编码。
#
# 背景（审计 P2-20）：`data/results/logs/` 下 10/12 个日志**单个文件内混了 UTF-8 与 GBK**
# ——脚本自身的中文输出是 UTF-8，但交错的 Windows/PowerShell 错误文本用 OEM 代码页，
# 于是任何 UTF-8 读取器读到中途就会失败，原始证据对工具链不可读。已按行救回 26 行中文
# （零丢失），并把原文件备份到 `data/results/logs_original_encoding/`。
#
# 今后请用本脚本跑评测，不要再直接 `python xxx.py`：
#
#   pwsh scripts/run.ps1 evaluate.py --task det --model small
#   pwsh scripts/run.ps1 evaluate.py --task rec --model small 2>&1 |
#       Tee-Object -FilePath data/results/logs/rec_small.log
#
# 记录日志时优先用 `Tee-Object`（PowerShell 侧统一 UTF-8），不要用 `>` 重定向
# ——后者会按 UTF-16LE 写文件，反而引入新的编码问题。

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "0"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Warning "设置控制台编码失败（不影响 Python 侧的 UTF-8）：$_"
}

$python = "D:\miniconda3\envs\scene-text\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "找不到解释器 $python —— 请确认 conda 环境 scene-text 存在"
    exit 1
}

if ($args.Count -lt 1) {
    Write-Host "用法: pwsh scripts/run.ps1 <脚本> [参数...]"
    Write-Host "例:   pwsh scripts/run.ps1 evaluate.py --task det --model small"
    exit 1
}

& $python -u @args
exit $LASTEXITCODE
