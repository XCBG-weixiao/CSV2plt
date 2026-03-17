from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import ReadCsvParams, choose_read_params, fmt_pct, json_dump, write_text


def infer_column_roles(df: pd.DataFrame, max_categories: int = 50) -> Dict[str, List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Datetime inference: try parse object columns with reasonable success rate.
    datetime_cols: List[str] = []
    for c in df.columns:
        if c in numeric_cols:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            if non_null.empty:
                continue
            sample = non_null.astype(str).head(5000)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True, utc=False)
            ok = parsed.notna().mean()
            if ok >= 0.8:
                datetime_cols.append(c)

    categorical_cols: List[str] = []
    for c in df.columns:
        if c in numeric_cols or c in datetime_cols:
            continue
        s = df[c]
        non_null = s.dropna()
        if non_null.empty:
            categorical_cols.append(c)
            continue
        nunique = non_null.nunique(dropna=True)
        if nunique <= max_categories:
            categorical_cols.append(c)

    return {
        "numeric_columns": numeric_cols,
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical_cols,
    }


def build_profile(df: pd.DataFrame, read_params: ReadCsvParams, input_path: str) -> Dict[str, Any]:
    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

    columns: List[Dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        non_null = s.notna().sum()
        missing = int(df.shape[0] - non_null)
        nunique = int(s.nunique(dropna=True)) if df.shape[0] else 0
        col_info: Dict[str, Any] = {
            "name": str(c),
            "dtype": str(s.dtype),
            "non_null": int(non_null),
            "missing": missing,
            "missing_rate": float(missing / df.shape[0]) if df.shape[0] else 0.0,
            "nunique": nunique,
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            col_info["numeric"] = {k: (None if pd.isna(v) else float(v)) for k, v in desc.to_dict().items()}
        columns.append(col_info)

    roles = infer_column_roles(df)
    profile: Dict[str, Any] = {
        "input": {"path": input_path},
        "shape": shape,
        "read_params": {"encoding": read_params.encoding, "delimiter": read_params.delimiter},
        "columns": columns,
        **roles,
    }
    return profile


def profile_to_markdown(profile: Dict[str, Any]) -> str:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    rp = profile["read_params"]
    numeric_cols = profile.get("numeric_columns", [])
    datetime_cols = profile.get("datetime_columns", [])
    categorical_cols = profile.get("categorical_columns", [])

    # Top missing columns
    col_rows = profile["columns"]
    top_missing = sorted(col_rows, key=lambda r: r["missing_rate"], reverse=True)[: min(15, len(col_rows))]

    lines: List[str] = []
    lines.append("# CSV Profile")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Rows**: {rows}")
    lines.append(f"- **Columns**: {cols}")
    lines.append(f"- **Encoding**: `{rp['encoding']}`")
    lines.append(f"- **Delimiter**: `{rp['delimiter']}`")
    lines.append("")
    lines.append("## Column roles (inferred)")
    lines.append(f"- **Numeric** ({len(numeric_cols)}): {', '.join(map(str, numeric_cols[:50]))}{' ...' if len(numeric_cols) > 50 else ''}")
    lines.append(f"- **Datetime** ({len(datetime_cols)}): {', '.join(map(str, datetime_cols[:50]))}{' ...' if len(datetime_cols) > 50 else ''}")
    lines.append(f"- **Categorical** ({len(categorical_cols)}): {', '.join(map(str, categorical_cols[:50]))}{' ...' if len(categorical_cols) > 50 else ''}")
    lines.append("")
    lines.append("## Missingness (top)")
    lines.append("")
    lines.append("| column | missing | missing_rate | nunique | dtype |")
    lines.append("|---|---:|---:|---:|---|")
    for r in top_missing:
        lines.append(
            f"| {r['name']} | {r['missing']} | {fmt_pct(r['missing_rate'])} | {r['nunique']} | {r['dtype']} |"
        )
    lines.append("")
    return "\n".join(lines)


def read_csv_forgiving(
    path: str,
    delimiter: Optional[str],
    encoding: Optional[str],
    nrows: Optional[int],
) -> tuple[pd.DataFrame, ReadCsvParams]:
    read_params = choose_read_params(path, delimiter=delimiter, encoding=encoding)

    last_err: Optional[Exception] = None
    encodings = [read_params.encoding] + [e for e in ["utf-8-sig", "utf-8", "gbk", "latin1"] if e != read_params.encoding]
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
            continue

    assert last_err is not None
    raise last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV")
    ap.add_argument("--out", default="artifacts/profile.md", help="Output markdown path")
    ap.add_argument("--json", dest="json_out", default="artifacts/profile.json", help="Output JSON path")
    ap.add_argument("--delimiter", default=None, help="Override delimiter (e.g. , ; \\t |)")
    ap.add_argument("--encoding", default=None, help="Override encoding (e.g. utf-8, gbk)")
    ap.add_argument("--nrows", type=int, default=None, help="Optional row limit for faster profiling")
    args = ap.parse_args()

    df, read_params = read_csv_forgiving(args.input, args.delimiter, args.encoding, args.nrows)

    # Normalize column names to strings (matplotlib/markdown friendliness).
    df.columns = [str(c) for c in df.columns]

    profile = build_profile(df, read_params=read_params, input_path=str(Path(args.input)))
    json_dump(args.json_out, profile)
    write_text(args.out, profile_to_markdown(profile))

    print(f"Wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()

