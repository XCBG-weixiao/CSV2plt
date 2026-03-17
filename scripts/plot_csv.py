from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ReadCsvParams, choose_read_params, ensure_dir, safe_filename, top_abs_correlations, write_text

from nl2spec import build_spec


def read_csv_forgiving(
    path: str,
    delimiter: Optional[str],
    encoding: Optional[str],
    nrows: Optional[int] = None,
) -> tuple[pd.DataFrame, ReadCsvParams]:
    read_params = choose_read_params(path, delimiter=delimiter, encoding=encoding)
    encodings = [read_params.encoding] + [e for e in ["utf-8-sig", "utf-8", "gbk", "latin1"] if e != read_params.encoding]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            try:
                df = pd.read_csv(
                    path,
                    sep=read_params.delimiter,
                    encoding=enc,
                    on_bad_lines="skip",
                    nrows=nrows,
                    low_memory=False,
                )
            except TypeError:
                # pandas<1.3 compatibility
                df = pd.read_csv(
                    path,
                    sep=read_params.delimiter,
                    encoding=enc,
                    error_bad_lines=False,  # type: ignore
                    warn_bad_lines=False,  # type: ignore
                    nrows=nrows,
                )
            return df, ReadCsvParams(encoding=enc, delimiter=read_params.delimiter)
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def load_profile(profile_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not profile_path:
        return None
    p = Path(profile_path)
    if not p.exists():
        return None
    import json

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_spec(spec_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not spec_path:
        return None
    p = Path(spec_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.shape[0] <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def plot_missingness(df: pd.DataFrame, outpath: Path, top_n: int = 40) -> None:
    missing_rate = df.isna().mean().sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > 0].head(top_n)

    plt.figure(figsize=(10, max(3, 0.25 * len(missing_rate))))
    if len(missing_rate) == 0:
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")
    else:
        sns.barplot(x=missing_rate.values, y=missing_rate.index, orient="h")
        plt.xlabel("Missing rate")
        plt.ylabel("Column")
        plt.title("Missingness (top columns)")
        plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str], outpath: Path, max_cols: int = 12) -> None:
    cols = numeric_cols[:max_cols]
    if not cols:
        return
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols):
        s = df[c].dropna()
        ax.hist(s, bins="auto", color="#4C78A8", alpha=0.85)
        ax.set_title(c)
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Numeric distributions (histograms)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame, numeric_cols: List[str], outpath: Path, max_cols: int = 12) -> None:
    cols = numeric_cols[:max_cols]
    if not cols:
        return
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols):
        s = df[c].dropna()
        sns.boxplot(x=s, ax=ax, color="#F58518")
        ax.set_title(c)
        ax.set_xlabel("")
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Numeric boxplots", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, numeric_cols: List[str], outpath: Path, max_cols: int = 30) -> Optional[pd.DataFrame]:
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    plt.figure(figsize=(min(14, 0.5 * len(cols) + 4), min(12, 0.5 * len(cols) + 4)))
    sns.heatmap(corr, cmap="vlag", center=0.0, square=False)
    plt.title("Correlation heatmap (Pearson)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return corr


def plot_scatter_matrix(df: pd.DataFrame, numeric_cols: List[str], outpath: Path, max_cols: int = 5, sample_rows: int = 5000) -> None:
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return
    d = _maybe_sample(df[cols], sample_rows).dropna()
    if d.shape[0] == 0:
        return
    # seaborn pairplot can be heavy; keep it small.
    g = sns.pairplot(d, corner=True, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
    g.fig.suptitle("Sampled scatter matrix", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(outpath, dpi=160)
    plt.close(g.fig)


def plot_overview(df: pd.DataFrame, profile: Optional[Dict[str, Any]], corr: Optional[pd.DataFrame], outpath: Path) -> None:
    rows, cols = df.shape
    missing_total = float(df.isna().mean().mean()) if cols else 0.0

    fig = plt.figure(figsize=(12, 7))
    ax_text = fig.add_subplot(2, 2, 1)
    ax_text.axis("off")

    lines = [
        "CSV Overview",
        "",
        f"Rows: {rows}",
        f"Columns: {cols}",
        f"Overall missingness (avg over columns): {missing_total*100:.2f}%",
    ]
    if profile and "read_params" in profile:
        rp = profile["read_params"]
        lines.append(f"Encoding: {rp.get('encoding')}")
        lines.append(f"Delimiter: {rp.get('delimiter')}")

    ax_text.text(0, 1, "\n".join(lines), va="top", fontsize=11)

    ax_miss = fig.add_subplot(2, 2, 2)
    miss = df.isna().mean().sort_values(ascending=False).head(15)
    sns.barplot(x=miss.values, y=miss.index, orient="h", ax=ax_miss)
    ax_miss.set_title("Missingness (top 15)")
    ax_miss.set_xlim(0, 1)
    ax_miss.set_xlabel("")
    ax_miss.set_ylabel("")

    ax_corr = fig.add_subplot(2, 1, 2)
    ax_corr.axis("off")
    if corr is not None and corr.shape[0] >= 2:
        tops = top_abs_correlations(corr, k=10)
        if tops:
            text = ["Top absolute correlations:"]
            for a, b, v in tops:
                text.append(f"- {a} vs {b}: {v:.3f}")
            ax_corr.text(0, 1, "\n".join(text), va="top", fontsize=11)
        else:
            ax_corr.text(0, 1, "No correlations available.", va="top", fontsize=11)
    else:
        ax_corr.text(0, 1, "Not enough numeric columns for correlations.", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _derive_month_column(df: pd.DataFrame, time_col: str, month_col: str) -> pd.DataFrame:
    d = df.copy()
    if time_col not in d.columns:
        return d
    dt = pd.to_datetime(d[time_col], errors="coerce", infer_datetime_format=True)
    d[month_col] = dt.dt.to_period("M").astype(str)
    return d


def _plot_bar_rank(
    df: pd.DataFrame,
    dimension: str,
    outpath: Path,
    highlight_top_k: Optional[int],
    highlight_color: Optional[str],
    title: str,
) -> pd.DataFrame:
    g = df.groupby(dimension).size().sort_values(ascending=False)
    plot_df = g.reset_index()
    plot_df.columns = [dimension, "count"]

    colors = None
    if highlight_top_k and highlight_color:
        colors = [highlight_color if i < highlight_top_k else "#4C78A8" for i in range(len(plot_df))]

    plt.figure(figsize=(max(8, 0.35 * len(plot_df) + 3), 5))
    ax = sns.barplot(x="count", y=dimension, data=plot_df, orient="h", palette=colors if colors else None)
    ax.set_title(title)
    ax.set_xlabel("离职人数（记录数）")
    ax.set_ylabel(dimension)

    max_count = float(plot_df["count"].max()) if not plot_df.empty else 0.0
    for p in ax.patches:
        w = float(p.get_width())
        ax.text(w + max(0.5, 0.01 * max_count), p.get_y() + p.get_height() / 2, f"{int(w)}", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return plot_df


def _plot_line_monthly(
    df: pd.DataFrame,
    time_col: str,
    month_col: str,
    dimension: str,
    outpath: Path,
    top_n: int = 10,
    others_bucket: bool = True,
    title: str = "",
) -> pd.DataFrame:
    d = _derive_month_column(df, time_col=time_col, month_col=month_col)
    d = d.dropna(subset=[month_col])
    counts = d.groupby([month_col, dimension]).size().reset_index(name="count")
    if counts.empty:
        return counts

    totals = counts.groupby(dimension)["count"].sum().sort_values(ascending=False)
    top_groups = list(totals.head(top_n).index)
    if others_bucket and len(totals) > top_n:
        counts[dimension] = counts[dimension].where(counts[dimension].isin(top_groups), other="Others")
        counts = counts.groupby([month_col, dimension])["count"].sum().reset_index()

    pivot = counts.pivot(index=month_col, columns=dimension, values="count").fillna(0.0).sort_index()

    plt.figure(figsize=(12, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col].values, marker="o", linewidth=1.8, label=str(col))
    plt.title(title or f"按月离职人数趋势（按 {dimension} 分组）")
    plt.xlabel("离职月份")
    plt.ylabel("离职人数（记录数）")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    long_df = pivot.reset_index().melt(id_vars=[month_col], var_name=dimension, value_name="count")
    return long_df


def build_report(
    input_csv: str,
    outdir: Path,
    plots: List[Path],
    profile: Optional[Dict[str, Any]],
    corr: Optional[pd.DataFrame],
    read_params: ReadCsvParams,
    sample_notes: List[str],
    request: Optional[str] = None,
    spec: Optional[Dict[str, Any]] = None,
    findings: Optional[Dict[str, Any]] = None,
) -> str:
    plot_rel = [p.relative_to(outdir.parent) if p.is_absolute() else p for p in plots]
    lines: List[str] = []
    lines.append("# 数据分析报告")
    lines.append("")

    if request or spec:
        lines.append("## 1. 需求复述")
        if request:
            lines.append(f"- **用户需求**：{request}")
        if spec and spec.get("warnings"):
            lines.append("- **解析警告/假设**：")
            for w in spec["warnings"]:
                lines.append(f"  - {w}")
        lines.append("")

    lines.append("## 2. 数据字段与口径")
    lines.append("- **离职人数**：以记录数计数（每行=1位离职人员）")
    if spec and spec.get("mapping", {}).get("time_column"):
        tc = spec["mapping"]["time_column"]
        mc = spec["mapping"].get("month_column", "离职月份")
        lines.append(f"- **按月**：由 `{tc}` 解析日期后派生 `{mc}`（YYYY-MM）再聚合")
    lines.append("")

    lines.append("## 3. 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append(f'python scripts/profile_csv.py --input "{input_csv}" --out artifacts/profile.md --json artifacts/profile.json')
    if request:
        lines.append(
            f'python scripts/plot_csv.py --input "{input_csv}" --profile artifacts/profile.json --request "{request}" --outdir artifacts/plots --report artifacts/report.md'
        )
    else:
        lines.append(f'python scripts/plot_csv.py --input "{input_csv}" --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md')
    lines.append("```")
    lines.append("")

    lines.append("## 4. 数据读取参数")
    lines.append(f"- **encoding**: `{read_params.encoding}`")
    lines.append(f"- **delimiter**: `{read_params.delimiter}`")
    lines.append("")

    if spec:
        lines.append("## 5. 字段映射与预处理")
        mp = spec.get("mapping", {})
        lines.append(f"- **分组列**：`{mp.get('group_column')}`")
        if mp.get("time_column"):
            lines.append(f"- **时间列**：`{mp.get('time_column')}`")
            lines.append(f"- **月份列（派生）**：`{mp.get('month_column', '离职月份')}`")
        lines.append("")

        lines.append("## 6. 图表清单")
        for ch in spec.get("charts", []):
            ctype = ch.get("type")
            if ctype == "bar_rank":
                dim = ch.get("dimension")
                st = ch.get("style", {}) or {}
                lines.append(f"- **柱状图（排序）**：按 `{dim}` 统计离职人数并降序排序")
                if st.get("highlight_top_k") and st.get("highlight_color"):
                    lines.append(f"  - Top{st['highlight_top_k']} 使用颜色：`{st['highlight_color']}`")
            elif ctype == "line_monthly":
                dim = ch.get("dimension")
                tc = ch.get("time_column")
                mc = ch.get("month_column")
                lim = ch.get("limits", {}) or {}
                lines.append(f"- **折线图（按月趋势）**：`{tc}` → `{mc}`，按 `{dim}` 分组按月统计")
                if lim.get("top_n"):
                    lines.append(f"  - 分组过多时默认 Top{lim['top_n']} + Others")
        lines.append("")

    if findings:
        lines.append("## 7. 关键发现")
        for k, v in findings.items():
            lines.append(f"- **{k}**：{v}")
        lines.append("")

    if sample_notes:
        lines.append("## 附：性能与降级说明")
        for n in sample_notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## 8. 产物（图像与文件）")
    for p in plot_rel:
        lines.append(f"- `{p.as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV")
    ap.add_argument("--profile", default=None, help="Path to artifacts/profile.json (optional)")
    ap.add_argument("--request", default=None, help="Chinese natural language request (optional)")
    ap.add_argument("--spec", default=None, help="ChartSpec JSON path (optional; overrides --request)")
    ap.add_argument("--outdir", default="artifacts/plots", help="Directory to write plots")
    ap.add_argument("--report", default="artifacts/report.md", help="Report markdown path")
    ap.add_argument("--spec_out", default="artifacts/spec.json", help="Where to write spec when using --request")
    ap.add_argument("--delimiter", default=None, help="Override delimiter")
    ap.add_argument("--encoding", default=None, help="Override encoding")
    ap.add_argument("--max_corr_cols", type=int, default=30, help="Max numeric columns for correlation heatmap")
    ap.add_argument("--max_hist_cols", type=int, default=12, help="Max numeric columns for histogram/boxplot grids")
    ap.add_argument("--max_pair_cols", type=int, default=5, help="Max numeric columns for scatter matrix")
    ap.add_argument("--pair_sample_rows", type=int, default=5000, help="Row sample size for scatter matrix")
    ap.add_argument("--mode", default="auto", choices=["auto", "base", "spec"], help="auto=spec if provided else base")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    profile = load_profile(args.profile)
    spec = load_spec(args.spec)

    df, read_params = read_csv_forgiving(args.input, args.delimiter, args.encoding)
    df.columns = [str(c) for c in df.columns]

    mode = args.mode
    if mode == "auto":
        mode = "spec" if (spec is not None or args.request) else "base"

    if mode == "spec" and spec is None and args.request:
        if not profile:
            raise SystemExit("When using --request, please also provide --profile artifacts/profile.json")
        spec = build_spec(args.request, profile, out_path=args.spec_out)

    numeric_cols = infer_numeric_columns(df)
    sample_notes: List[str] = []
    if len(numeric_cols) > args.max_corr_cols:
        sample_notes.append(f"Correlation heatmap truncated to first {args.max_corr_cols} numeric columns.")
    if len(numeric_cols) > args.max_hist_cols:
        sample_notes.append(f"Histogram/boxplot grids truncated to first {args.max_hist_cols} numeric columns.")
    if df.shape[0] > args.pair_sample_rows:
        sample_notes.append(f"Scatter matrix sampled to {args.pair_sample_rows} rows (random_state=42).")
    if len(numeric_cols) > args.max_pair_cols:
        sample_notes.append(f"Scatter matrix truncated to first {args.max_pair_cols} numeric columns.")

    plots: List[Path] = []
    findings: Dict[str, Any] = {}
    corr: Optional[pd.DataFrame] = None

    if mode == "base":
        p_missing = outdir / "missingness.png"
        plot_missingness(df, p_missing)
        plots.append(p_missing)

        if numeric_cols:
            p_hist = outdir / "numeric_histograms.png"
            plot_numeric_histograms(df, numeric_cols, p_hist, max_cols=args.max_hist_cols)
            plots.append(p_hist)

            p_box = outdir / "numeric_boxplots.png"
            plot_numeric_boxplots(df, numeric_cols, p_box, max_cols=args.max_hist_cols)
            plots.append(p_box)

            p_corr = outdir / "correlation_heatmap.png"
            corr = plot_corr_heatmap(df, numeric_cols, p_corr, max_cols=args.max_corr_cols)
            if corr is not None:
                plots.append(p_corr)
        else:
            corr = None

        p_pair = outdir / "scatter_matrix.png"
        try:
            plot_scatter_matrix(df, numeric_cols, p_pair, max_cols=args.max_pair_cols, sample_rows=args.pair_sample_rows)
            if p_pair.exists():
                plots.append(p_pair)
        except Exception:
            pass

        p_overview = outdir / "overview.png"
        plot_overview(df, profile=profile, corr=corr, outpath=p_overview)
        plots.append(p_overview)
    else:
        if not spec:
            raise SystemExit("Spec mode requires --spec or --request (and --profile when using --request).")
        for ch in spec.get("charts", []):
            ctype = ch.get("type")
            if ctype == "bar_rank":
                dim = ch.get("dimension")
                if not dim or dim not in df.columns:
                    continue
                st = ch.get("style", {}) or {}
                outp = outdir / f"bar_rank_{safe_filename(dim)}.png"
                ranked = _plot_bar_rank(
                    df,
                    dimension=dim,
                    outpath=outp,
                    highlight_top_k=st.get("highlight_top_k"),
                    highlight_color=st.get("highlight_color"),
                    title=ch.get("title") or f"按 {dim} 统计离职人数（降序）",
                )
                if outp.exists():
                    plots.append(outp)
                if not ranked.empty:
                    findings[f"{dim} Top1"] = f"{ranked.iloc[0][dim]}（{int(ranked.iloc[0]['count'])}）"
            elif ctype == "line_monthly":
                dim = ch.get("dimension")
                tc = ch.get("time_column")
                mc = ch.get("month_column") or "离职月份"
                if not dim or dim not in df.columns or not tc or tc not in df.columns:
                    continue
                lim = ch.get("limits", {}) or {}
                outp = outdir / f"line_monthly_{safe_filename(dim)}.png"
                series_df = _plot_line_monthly(
                    df,
                    time_col=tc,
                    month_col=mc,
                    dimension=dim,
                    outpath=outp,
                    top_n=int(lim.get("top_n", 10)),
                    others_bucket=bool(lim.get("others_bucket", True)),
                    title=ch.get("title") or f"按月离职人数趋势（按 {dim} 分组）",
                )
                if outp.exists():
                    plots.append(outp)
                if not series_df.empty:
                    peak = series_df.groupby(mc)["count"].sum().sort_values(ascending=False).head(1)
                    if len(peak) == 1:
                        findings["离职高峰月份"] = f"{peak.index[0]}（{int(peak.iloc[0])}）"

    report_text = build_report(
        input_csv=args.input,
        outdir=outdir,
        plots=plots,
        profile=profile,
        corr=corr,
        read_params=read_params,
        sample_notes=sample_notes,
        request=args.request,
        spec=spec,
        findings=findings or None,
    )
    write_text(args.report, report_text)
    print(f"Wrote plots to {outdir} and report to {args.report}")


if __name__ == "__main__":
    main()

