from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd


@dataclass
class IssueRecord:
    key: str
    summary: str
    status: str
    created: Optional[pd.Timestamp]
    updated: Optional[pd.Timestamp]
    resolutiondate: Optional[pd.Timestamp]
    assignee: Optional[str]
    reporter: Optional[str]
    labels: List[str]


def normalize_issues(raw_issues: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[IssueRecord] = []
    for issue in raw_issues:
        fields = issue.get("fields", {})
        assignee = fields.get("assignee")
        reporter = fields.get("reporter")
        labels = fields.get("labels") or []

        record = IssueRecord(
            key=issue.get("key", ""),
            summary=fields.get("summary", "") or "",
            status=(fields.get("status") or {}).get("name", "") or "",
            created=_safe_parse_ts(fields.get("created")),
            updated=_safe_parse_ts(fields.get("updated")),
            resolutiondate=_safe_parse_ts(fields.get("resolutiondate")),
            assignee=(assignee or {}).get("displayName") if assignee else None,
            reporter=(reporter or {}).get("displayName") if reporter else None,
            labels=labels,
        )
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "key",
                "summary",
                "status",
                "created",
                "updated",
                "resolutiondate",
                "assignee",
                "reporter",
                "labels",
            ]
        )

    df = pd.DataFrame([r.__dict__ for r in records])

    # 确保时间列为统一的无时区 datetime，避免 tz-aware / tz-naive 混用导致报错
    for col in ["created", "updated", "resolutiondate"]:
        if col in df.columns:
            series = pd.to_datetime(df[col], errors="coerce", utc=True)
            # 去掉时区信息，统一为本地无时区时间
            df[col] = series.dt.tz_convert(None)

    if "created" in df.columns:
        df["created_date"] = df["created"].dt.date
    else:
        df["created_date"] = pd.NaT

    if "resolutiondate" in df.columns:
        df["resolved_date"] = df["resolutiondate"].dt.date
        df["cycle_time_days"] = (
            (df["resolutiondate"] - df["created"]).dt.total_seconds() / 86400
        )
    else:
        df["resolved_date"] = pd.NaT
        df["cycle_time_days"] = pd.NA

    return df


def _safe_parse_ts(value: Any) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return None
