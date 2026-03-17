from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TIME_KEYWORDS = ["时间", "日期", "月", "月份", "每月", "按月", "逐月"]


def load_profile(profile_path: str | Path) -> Dict[str, Any]:
    p = Path(profile_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    t = text.strip()
    t = t.replace("：", ":").replace("，", ",").replace("；", ";")
    return t


def _extract_int_after(text: str, patterns: List[str]) -> Optional[int]:
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue
    return None


def _contains_any(text: str, kws: List[str]) -> bool:
    return any(k in text for k in kws)


@dataclass(frozen=True)
class MappingResult:
    time_col: Optional[str]
    group_col: Optional[str]
    month_col_name: str
    warnings: List[str]


def infer_mapping(user_request: str, profile: Dict[str, Any]) -> MappingResult:
    req = _normalize(user_request)

    cols = [c["name"] for c in profile.get("columns", []) if "name" in c]
    datetime_cols = profile.get("datetime_columns", []) or []
    categorical_cols = profile.get("categorical_columns", []) or []

    # Heuristic keyword → column mapping (Chinese HR turnover dataset defaults)
    keyword_to_candidates = [
        (["科室", "部门"], ["离职科室", "科室", "部门"]),
        (["职位", "岗位"], ["离职的职位", "职位", "岗位"]),
        (["学历"], ["学历"]),
        (["离职时间", "时间", "日期"], ["离职时间", "日期", "时间"]),
    ]

    def pick_by_candidates(cands: List[str]) -> Optional[str]:
        for cand in cands:
            if cand in cols:
                return cand
        return None

    group_col: Optional[str] = None
    for kws, cands in keyword_to_candidates[:3]:
        if _contains_any(req, kws):
            group_col = pick_by_candidates(cands)
            if group_col:
                break

    # If still unknown, choose a reasonable categorical column
    if not group_col:
        for preferred in ["离职科室", "离职的职位", "学历"]:
            if preferred in cols:
                group_col = preferred
                break
    if not group_col and categorical_cols:
        group_col = categorical_cols[0]

    time_col: Optional[str] = None
    if _contains_any(req, DEFAULT_TIME_KEYWORDS) or "折线" in req or "趋势" in req:
        time_col = pick_by_candidates(["离职时间", "日期", "时间"]) or (datetime_cols[0] if datetime_cols else None)

    warnings: List[str] = []
    if not group_col:
        warnings.append("未能确定分组维度列（科室/职位/学历）。请在需求中明确或检查 CSV 列名。")
    if (_contains_any(req, DEFAULT_TIME_KEYWORDS) or "折线" in req or "趋势" in req) and not time_col:
        warnings.append("需求包含按时间/按月趋势，但未能识别时间列。请确认 CSV 中存在日期/时间列（如“离职时间”）。")

    return MappingResult(time_col=time_col, group_col=group_col, month_col_name="离职月份", warnings=warnings)


def build_spec(
    user_request: str,
    profile: Dict[str, Any],
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    req = _normalize(user_request)
    mapping = infer_mapping(req, profile)

    charts: List[Dict[str, Any]] = []

    # TopK highlight rule (e.g., 前两名/前2名/Top2)
    topk = _extract_int_after(req, patterns=[r"前(\d+)名", r"top\s*(\d+)", r"Top\s*(\d+)", r"前(\d+)个"])
    highlight_color = None
    if "红" in req:
        highlight_color = "red"

    wants_rank_bar = _contains_any(req, ["排序", "排名", "柱状图", "条形图", "Top", "top"]) or (topk is not None)
    wants_monthly_line = _contains_any(req, ["每月", "按月", "逐月", "折线", "趋势", "走势", "折线图"])

    if wants_rank_bar:
        charts.append(
            {
                "type": "bar_rank",
                "dimension": mapping.group_col,
                "metric": {"type": "count_rows"},
                "sort": {"by": "metric", "order": "desc"},
                "style": {
                    "highlight_top_k": topk or 2 if highlight_color else topk,
                    "highlight_color": highlight_color,
                },
                "title": "按维度排序（离职人数）",
            }
        )

    if wants_monthly_line:
        charts.append(
            {
                "type": "line_monthly",
                "time_column": mapping.time_col,
                "month_column": mapping.month_col_name,
                "dimension": mapping.group_col,
                "metric": {"type": "count_rows"},
                "limits": {"top_n": 10, "others_bucket": True},
                "title": "按月趋势（离职人数）",
            }
        )

    if not charts:
        charts.append(
            {
                "type": "bar_rank",
                "dimension": mapping.group_col,
                "metric": {"type": "count_rows"},
                "sort": {"by": "metric", "order": "desc"},
                "style": {"highlight_top_k": None, "highlight_color": None},
                "title": "按维度排序（离职人数）",
            }
        )
        mapping_warnings = mapping.warnings + ["未能从需求中识别具体图表类型，已默认生成排序柱状图。"]
    else:
        mapping_warnings = mapping.warnings

    spec: Dict[str, Any] = {
        "version": 1,
        "request": req,
        "mapping": {
            "group_column": mapping.group_col,
            "time_column": mapping.time_col,
            "month_column": mapping.month_col_name,
        },
        "charts": charts,
        "warnings": mapping_warnings,
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

    return spec


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to artifacts/profile.json")
    ap.add_argument("--request", required=True, help="User request (Chinese natural language)")
    ap.add_argument("--out", default="artifacts/spec.json", help="Output spec path")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    spec = build_spec(args.request, profile, out_path=args.out)
    print(f"Wrote {args.out} with {len(spec.get('charts', []))} chart(s)")


if __name__ == "__main__":
    main()

