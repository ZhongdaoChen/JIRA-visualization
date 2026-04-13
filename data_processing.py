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
    duedate: Optional[pd.Timestamp]
    assignee: Optional[str]
    assignee_name: Optional[str]   # JIRA login name (email in this instance)
    reporter: Optional[str]
    reporter_name: Optional[str]   # JIRA login name (email in this instance)
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
            created=_safe_parse_date(fields.get("created")),
            updated=_safe_parse_date(fields.get("updated")),
            resolutiondate=_safe_parse_date(fields.get("resolutiondate")),
            duedate=_safe_parse_date(fields.get("duedate")),
            assignee=(assignee or {}).get("displayName") if assignee else None,
            assignee_name=(assignee or {}).get("name") if assignee else None,
            reporter=(reporter or {}).get("displayName") if reporter else None,
            reporter_name=(reporter or {}).get("name") if reporter else None,
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
                "duedate",
                "assignee",
                "assignee_name",
                "reporter",
                "reporter_name",
                "labels",
            ]
        )

    df = pd.DataFrame([r.__dict__ for r in records])

    # 所有时间列统一转换为 date 类型（只保留日期部分）
    for col in ["created", "updated", "resolutiondate", "duedate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    if "created" in df.columns:
        df["created_date"] = df["created"]
    else:
        df["created_date"] = pd.NaT

    if "resolutiondate" in df.columns:
        df["resolved_date"] = df["resolutiondate"]
        # cycle_time_days 只保留整数天数（向下取整）
        df["cycle_time_days"] = (
            (pd.to_datetime(df["resolutiondate"]) - pd.to_datetime(df["created"]))
            .dt.total_seconds()
            .apply(lambda x: int(x // 86400) if pd.notna(x) else None)
        )
    else:
        df["resolved_date"] = pd.NaT
        df["cycle_time_days"] = None

    return df


def _safe_parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse date/datetime value and return date only (no time component)."""
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


@dataclass
class KPIResult:
    """KPI 指标计算结果"""
    total_count: int  # 总 issue 数
    resolved_count: int  # 已解决数
    closed_count: int  # 已关闭数
    resolution_rate: float  # 解决率 (0-100)
    avg_cycle_days: Optional[float]  # 平均解决周期
    median_cycle_days: Optional[float]  # 中位数解决周期
    min_cycle_days: Optional[float]  # 最短周期
    max_cycle_days: Optional[float]  # 最长周期


def calculate_kpis(df: pd.DataFrame, status_column: str = "status") -> KPIResult:
    """
    根据选中的 issue 计算 KPI 指标

    Args:
        df: 包含 issue 数据的 DataFrame（已包含 cycle_time_days 列）
        status_column: 状态列名

    Returns:
        KPIResult 包含所有计算好的指标
    """
    total_count = len(df)

    # 已解决：有 resolutiondate 的
    resolved_count = df["resolutiondate"].notna().sum() if "resolutiondate" in df.columns else 0

    # 已关闭：status 为 Closed/Done/Resolved 等
    closed_statuses = {"closed", "done", "resolved", "已关闭", "已完成", "已解决"}
    closed_count = 0
    if status_column in df.columns:
        closed_count = df[status_column].apply(
            lambda x: str(x).lower() in closed_statuses if x else False
        ).sum()

    # 解决率
    resolution_rate = (resolved_count / total_count * 100) if total_count > 0 else 0.0

    # 周期统计（只针对已解决的 issue）
    cycle_days = df.get("cycle_time_days")
    if cycle_days is not None:
        cycle_days_valid = cycle_days.dropna()
        if len(cycle_days_valid) > 0:
            avg_cycle_days = float(cycle_days_valid.mean())
            median_cycle_days = float(cycle_days_valid.median())
            min_cycle_days = float(cycle_days_valid.min())
            max_cycle_days = float(cycle_days_valid.max())
        else:
            avg_cycle_days = median_cycle_days = min_cycle_days = max_cycle_days = None
    else:
        avg_cycle_days = median_cycle_days = min_cycle_days = max_cycle_days = None

    return KPIResult(
        total_count=total_count,
        resolved_count=resolved_count,
        closed_count=closed_count,
        resolution_rate=resolution_rate,
        avg_cycle_days=avg_cycle_days,
        median_cycle_days=median_cycle_days,
        min_cycle_days=min_cycle_days,
        max_cycle_days=max_cycle_days,
    )
