from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    ReadCsvParams,
    choose_read_params,
    ensure_dir,
    json_dump,
    safe_filename,
    top_abs_correlations,
    write_text,
)


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


def build_report(
    input_csv: str,
    outdir: Path,
    plots: List[Path],
    profile: Optional[Dict[str, Any]],
    corr: Optional[pd.DataFrame],
    read_params: ReadCsvParams,
    sample_notes: List[str],
) -> str:
    plot_rel = [p.relative_to(outdir.parent) if p.is_absolute() else p for p in plots]
    lines: List[str] = []
    lines.append("# CSV Analysis Report")
    lines.append("")
    lines.append("## Repro")
    lines.append("")
    lines.append("```bash")
    lines.append(f'python scripts/profile_csv.py --input "{input_csv}" --out artifacts/profile.md --json artifacts/profile.json')
    lines.append(f'python scripts/plot_csv.py --input "{input_csv}" --profile artifacts/profile.json --outdir artifacts/plots --report artifacts/report.md')
    lines.append("```")
    lines.append("")
    lines.append("## Read params")
    lines.append(f"- **encoding**: `{read_params.encoding}`")
    lines.append(f"- **delimiter**: `{read_params.delimiter}`")
    lines.append("")
    if sample_notes:
        lines.append("## Performance notes")
        for n in sample_notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("## Artifacts")
    for p in plot_rel:
        lines.append(f"- `{p.as_posix()}`")
    lines.append("")
    if corr is not None and corr.shape[0] >= 2:
        tops = top_abs_correlations(corr, k=10)
        if tops:
            lines.append("## Top correlations (absolute)")
            for a, b, v in tops:
                lines.append(f"- **{a}** vs **{b}**: {v:.3f}")
            lines.append("")
    if profile:
        lines.append("## Inferred column roles")
        for k in ("numeric_columns", "datetime_columns", "categorical_columns"):
            vals = profile.get(k, [])
            lines.append(f"- **{k}** ({len(vals)}): {', '.join(map(str, vals[:50]))}{' ...' if len(vals) > 50 else ''}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV")
    ap.add_argument("--profile", default=None, help="Path to artifacts/profile.json (optional)")
    ap.add_argument("--outdir", default="artifacts/plots", help="Directory to write plots")
    ap.add_argument("--report", default="artifacts/report.md", help="Report markdown path")
    ap.add_argument("--delimiter", default=None, help="Override delimiter")
    ap.add_argument("--encoding", default=None, help="Override encoding")
    ap.add_argument("--max_corr_cols", type=int, default=30, help="Max numeric columns for correlation heatmap")
    ap.add_argument("--max_hist_cols", type=int, default=12, help="Max numeric columns for histogram/boxplot grids")
    ap.add_argument("--max_pair_cols", type=int, default=5, help="Max numeric columns for scatter matrix")
    ap.add_argument("--pair_sample_rows", type=int, default=5000, help="Row sample size for scatter matrix")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    profile = load_profile(args.profile)

    df, read_params = read_csv_forgiving(args.input, args.delimiter, args.encoding)
    df.columns = [str(c) for c in df.columns]

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
        # Pairplot can fail on corner cases; continue without blocking the pipeline.
        pass

    p_overview = outdir / "overview.png"
    plot_overview(df, profile=profile, corr=corr, outpath=p_overview)
    plots.append(p_overview)

    report_text = build_report(
        input_csv=args.input,
        outdir=outdir,
        plots=plots,
        profile=profile,
        corr=corr,
        read_params=read_params,
        sample_notes=sample_notes,
    )
    write_text(args.report, report_text)
    print(f"Wrote plots to {outdir} and report to {args.report}")


if __name__ == "__main__":
    main()

