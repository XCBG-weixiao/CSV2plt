# CSV Analysis Report

## Repro

```bash
python scripts/profile_csv.py --input "data/sample.csv" --out artifacts/profile.md --json artifacts/profile.json
python scripts/plot_csv.py --input "data/sample.csv" --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md
```

## Read params
- **encoding**: `utf-8-sig`
- **delimiter**: `,`

## Artifacts
- `artifacts/plots/missingness.png`
- `artifacts/plots/numeric_histograms.png`
- `artifacts/plots/numeric_boxplots.png`
- `artifacts/plots/correlation_heatmap.png`
- `artifacts/plots/scatter_matrix.png`
- `artifacts/plots/overview.png`

## Top correlations (absolute)
- **value_a** vs **value_b**: 0.991
- **id** vs **value_b**: -0.360
- **id** vs **value_a**: -0.318

## Inferred column roles
- **numeric_columns** (3): id, value_a, value_b
- **datetime_columns** (1): ts
- **categorical_columns** (2): group, flag
