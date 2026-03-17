# 数据分析报告

## 1. 需求复述
- **用户需求**：按离职科室统计离职人数并排序，前两名柱状图用红色；各科室每个月的折线图走势

## 2. 数据字段与口径
- **离职人数**：以记录数计数（每行=1位离职人员）
- **按月**：由 `离职时间` 解析日期后派生 `离职月份`（YYYY-MM）再聚合

## 3. 复现方式

```bash
python scripts/profile_csv.py --input "data/turnover_sample.csv" --out artifacts/profile.md --json artifacts/profile.json
python scripts/plot_csv.py --input "data/turnover_sample.csv" --profile artifacts/profile.json --request "按离职科室统计离职人数并排序，前两名柱状图用红色；各科室每个月的折线图走势" --outdir artifacts/plots --report artifacts/report.md
```

## 4. 数据读取参数
- **encoding**: `utf-8-sig`
- **delimiter**: `,`

## 5. 字段映射与预处理
- **分组列**：`离职科室`
- **时间列**：`离职时间`
- **月份列（派生）**：`离职月份`

## 6. 图表清单
- **柱状图（排序）**：按 `离职科室` 统计离职人数并降序排序
  - Top2 使用颜色：`red`
- **折线图（按月趋势）**：`离职时间` → `离职月份`，按 `离职科室` 分组按月统计
  - 分组过多时默认 Top10 + Others

## 7. 关键发现
- **离职科室 Top1**：研发（8）
- **离职高峰月份**：2025-03（6）

## 8. 产物（图像与文件）
- `artifacts/plots/bar_rank_col_38f40894.png`
- `artifacts/plots/line_monthly_col_38f40894.png`
