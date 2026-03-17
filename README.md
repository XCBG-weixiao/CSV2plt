# CSV2plt

把一份 CSV 自动做基础分析，并输出一组默认分析图到 `artifacts/`。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 快速开始（使用示例数据）

```bash
python scripts/profile_csv.py --input data/sample.csv --out artifacts/profile.md --json artifacts/profile.json
python scripts/plot_csv.py --input data/sample.csv --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md
```

生成的产物：
- `artifacts/profile.md`：数据画像（概览/缺失值/列角色推断）
- `artifacts/profile.json`：画像的结构化数据
- `artifacts/report.md`：分析汇总与复现命令
- `artifacts/plots/*.png`：默认图集

## 用你自己的 CSV

把 `<CSV_PATH>` 换成你的路径即可：

```bash
python scripts/profile_csv.py --input "<CSV_PATH>" --out artifacts/profile.md --json artifacts/profile.json
python scripts/plot_csv.py --input "<CSV_PATH>" --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md
```

常见可选参数：
- `profile_csv.py`
  - `--delimiter`：手动指定分隔符（例如 `,` `;` `\\t` `|`）
  - `--encoding`：手动指定编码（例如 `utf-8` `gbk`）
  - `--nrows`：限制读取行数，快速预览
- `plot_csv.py`
  - `--max_corr_cols`：相关性热力图最多使用多少数值列（默认 30）
  - `--max_hist_cols`：直方图/箱线图最多多少数值列（默认 12）
  - `--max_pair_cols`：散点矩阵最多多少数值列（默认 5）
  - `--pair_sample_rows`：散点矩阵最多抽样多少行（默认 5000）

## 项目级 Skill（Cursor）

本项目自带一个可复用 Skill：`.cursor/skills/csv2plt/SKILL.md`。\n
你在 Cursor 里对话提到 “CSV/数据分析/出图/matplotlib/seaborn/相关性/缺失值”等时，它会按工作流引导并执行：\n
1) 获取 CSV 输入与读取参数（可自动探测）\n
2) 生成画像 `artifacts/profile.md/json`\n
3) 生成图集 `artifacts/plots/` + 汇总 `artifacts/report.md`\n

