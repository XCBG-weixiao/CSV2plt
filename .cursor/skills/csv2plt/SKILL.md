---
name: csv2plt
description: Automates CSV profiling and plotting (missingness, distributions, boxplots, correlation heatmap, sampled scatter). Use when the user mentions CSV, pandas, matplotlib, seaborn, data analysis, profiling, or plotting charts from tabular data.
---

# csv2plt (CSV → 分析 → 出图)

## 适用场景（触发信号）
- 用户提到：CSV、pandas、matplotlib、seaborn、数据分析、画像(profile)、出图、plt、相关性、缺失值、直方图、箱线图、热力图。

## 你要做什么（高层目标）
把一份 CSV 自动跑通：
1) 读取与参数探测（编码/分隔符/坏行策略）
2) 数据画像（概要 + 关键统计 + 列类型推断）
3) 自动基础分析图集输出到 `artifacts/`
4) 汇总报告 `artifacts/report.md`（包含关键发现与复现命令）

## 工作流（严格按顺序执行）

### Step 0: 获取输入
向用户确认/收集（若用户没给就给默认值并继续）：
- CSV 路径（相对或绝对）
- 分隔符（未知则自动探测）
- 编码（未知则自动探测；常见：utf-8/utf-8-sig/gbk）
- 是否需要抽样（默认：大于 200k 行时自动抽样用于散点/两两散点）

如果用户没有提供 CSV 文件路径：
- 引导用户把 CSV 放到项目内（例如 `data/`），或直接粘贴一小段样例（前 50 行）以便先验证流程。

### Step 1: 安装依赖（只在缺依赖时）
运行：
- `python -m pip install -r requirements.txt`

### Step 2: 生成画像（profile）
运行：
- `python scripts/profile_csv.py --input "<CSV_PATH>" --out artifacts/profile.md --json artifacts/profile.json`

你要检查的关键输出：
- `artifacts/profile.json` 中是否包含：`shape`、`columns`、`numeric_columns`、`datetime_columns`、`categorical_columns`、`read_params`

### Step 3: 自动基础出图
运行：
- `python scripts/plot_csv.py --input "<CSV_PATH>" --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md`

默认图集（自动基础包）应包含：
- 缺失值条形图（按缺失率排序）
- 数值列直方图（多子图）
- 数值列箱线图（多子图）
- 相关性热力图（Pearson；列过多时截断）
- 抽样两两散点/散点矩阵（限制列数与行数，避免过重）
- 1 张“总览图”（关键指标文字 + 缺失值/相关性摘要）

### Step 4: 输出汇总给用户
给用户一个简短总结（不需要贴所有代码/图片）：
- 关键发现（缺失最严重列、疑似异常/离群、最高相关性对）
- 产物位置：`artifacts/profile.md`、`artifacts/report.md`、`artifacts/plots/`
- 复现命令（从 report.md 复制）

## 失败回路（必须执行）

### CSV 读取失败（编码/分隔符/坏行）
优先策略：
- 自动探测分隔符（用采样 + `csv.Sniffer`）
- 编码候选依次尝试：`utf-8-sig` → `utf-8` → `gbk` → `latin1`
- 对坏行：优先 `on_bad_lines="skip"`（pandas>=1.3）

若仍失败：
- 把最终尝试过的 read_csv 参数写入输出（profile/report）
- 给出最小复现命令（含 `--delimiter`/`--encoding`）并提示用户提供：前 20 行原始内容 + 报错堆栈

### 数据太大导致出图很慢/爆内存
降级策略（自动执行并在 report 中说明）：
- 两两散点只取最多 5 个数值列
- 行数抽样到 5k（或用户指定）
- 相关性只取前 N 个数值列（默认 30）

## 可选增强：MCP（不要求）
如果以后你想把“读写文件/运行 Python”也交给 MCP 工具链，可以考虑（在 Cursor 的 MCP 配置里）添加：\n
- filesystem：`@modelcontextprotocol/server-filesystem`\n
- python repl / safe python executor（用于在对话中直接跑 pandas/matplotlib）\n
然后将本工作流中的 Step 2/3 改为直接调用 MCP 工具执行相同命令与产物路径。

