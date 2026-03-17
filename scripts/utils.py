from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ENCODING_CANDIDATES: List[str] = ["utf-8-sig", "utf-8", "gbk", "latin1"]


@dataclass(frozen=True)
class ReadCsvParams:
    encoding: str
    delimiter: str


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text_sample(path: str | Path, max_bytes: int = 64 * 1024) -> bytes:
    with open(path, "rb") as f:
        return f.read(max_bytes)


def sniff_delimiter(sample_bytes: bytes) -> str:
    # Try to decode as utf-8-ish for delimiter sniffing; fall back to commas.
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            sample_text = sample_bytes.decode(enc, errors="replace")
            break
        except Exception:
            continue
    else:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Heuristic fallback: pick the delimiter with the highest average count.
        lines = [ln for ln in sample_text.splitlines() if ln.strip()][:20]
        if not lines:
            return ","
        candidates = [",", ";", "\t", "|"]
        best = ","
        best_score = -1.0
        for d in candidates:
            counts = [ln.count(d) for ln in lines]
            score = sum(counts) / len(counts)
            if score > best_score:
                best_score = score
                best = d
        return best


def choose_read_params(csv_path: str | Path, delimiter: Optional[str] = None, encoding: Optional[str] = None) -> ReadCsvParams:
    sample = read_text_sample(csv_path)
    delim = delimiter or sniff_delimiter(sample)
    enc = encoding or _sniff_encoding(sample) or ENCODING_CANDIDATES[0]
    return ReadCsvParams(encoding=enc, delimiter=delim)


def _sniff_encoding(sample_bytes: bytes) -> Optional[str]:
    # Quick validity checks for common encodings.
    for enc in ENCODING_CANDIDATES:
        try:
            sample_bytes.decode(enc)
            return enc
        except Exception:
            continue
    return None


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x * 100:.2f}%"


def safe_filename(name: str) -> str:
    """
    Cross-platform safe filename.

    - Keep ASCII letters/digits and a small set of separators.
    - Replace everything else (including CJK) with '_' so Windows terminals/filesystems
      won't produce mojibake.
    - Append a short hash to keep uniqueness.
    """
    import re
    import zlib

    raw = (name or "").strip()
    if not raw:
        raw = "col"

    # Replace Windows-forbidden characters first.
    bad = '<>:"/\\|?*'
    cleaned = "".join("_" if c in bad else c for c in raw)

    # Keep ASCII alnum + _-. ; replace others with underscore.
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "col"

    h = zlib.adler32(raw.encode("utf-8")) & 0xFFFFFFFF
    return f"{cleaned}_{h:08x}"


def top_abs_correlations(corr_df, k: int = 10) -> List[Tuple[str, str, float]]:
    # corr_df is a pandas DataFrame.
    cols = list(corr_df.columns)
    out: List[Tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_df.iloc[i, j]
            if v is None or (hasattr(v, "__float__") and (math.isnan(float(v)))):
                continue
            out.append((cols[i], cols[j], float(v)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out[:k]

