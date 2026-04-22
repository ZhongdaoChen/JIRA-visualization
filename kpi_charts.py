"""
KPI 图表绘制模块
使用 Plotly 绘制专业的、交互性强的 KPI 可视化图表
"""
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_status_distribution_chart(df: pd.DataFrame, status_column: str = "status") -> go.Figure:
    """
    状态分布柱状图
    """
    status_counts = df[status_column].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    status_counts = status_counts.sort_values("count", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=status_counts["status"],
        y=status_counts["count"],
        marker=dict(
            color=status_counts["count"],
            showscale=True,
            colorscale="Viridis"
        ),
        text=status_counts["count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>数量：%{y}<extra></extra>"
    ))

    fig.update_layout(
        title="状态分布",
        xaxis_title="状态",
        yaxis_title="数量",
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(tickangle=-45)

    return fig


def create_cycle_time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    解决周期分布直方图
    """
    cycle_days = df["cycle_time_days"].dropna()

    if len(cycle_days) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无已解决的 Issue",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="解决周期分布",
            height=400,
            xaxis=dict(title="周期 (天)"),
            yaxis=dict(title="数量"),
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cycle_days,
        nbinsx=20,
        marker=dict(color="steelblue"),
        hovertemplate="周期：%{x:.1f}天<br>数量：%{y}<extra></extra>"
    ))

    # 添加平均线和中位线
    avg = cycle_days.mean()
    median = cycle_days.median()

    fig.add_vline(x=avg, line_dash="dash", line_color="red",
                  annotation_text=f"平均：{avg:.1f}天", annotation_position="top")
    fig.add_vline(x=median, line_dash="dash", line_color="green",
                  annotation_text=f"中位数：{median:.1f}天", annotation_position="bottom")

    fig.update_layout(
        title="解决周期分布",
        xaxis_title="周期 (天)",
        yaxis_title="数量",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        bargap=0.1,
    )

    return fig


def create_cycle_time_by_assignee_chart(df: pd.DataFrame,
                                         assignee_column: str = "assignee",
                                         top_n: int = 15) -> go.Figure:
    """
    按 assignee 的平均解决时间柱状图
    """
    # 只统计已解决的 issue
    resolved_df = df[df["cycle_time_days"].notna()].copy()

    if len(resolved_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无已解决的 Issue",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f"Top {top_n} 平均解决时间 - Assignee",
            height=400,
        )
        return fig

    # 按 assignee 分组计算平均周期
    assignee_stats = resolved_df.groupby(assignee_column).agg({
        "cycle_time_days": ["mean", "count", "std"]
    }).reset_index()
    assignee_stats.columns = ["assignee", "avg_days", "count", "std"]
    assignee_stats = assignee_stats[assignee_stats["count"] >= 1]  # 至少 1 个 issue
    assignee_stats = assignee_stats.sort_values("avg_days", ascending=False).head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assignee_stats["assignee"],
        y=assignee_stats["avg_days"],
        marker=dict(
            color=assignee_stats["avg_days"],
            colorscale="RdYlGn_r",  # 红色=慢，绿色=快
            showscale=True,
        ),
        error_y=dict(
            type="data",
            array=assignee_stats["std"],
            visible=True,
            width=0.5,
        ),
        text=assignee_stats["avg_days"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>平均周期：%{y:.1f}天<br>标准差：%{error_y.array:.1f}<br>Issue 数：" +
                      assignee_stats["count"].apply(lambda x: str(int(x))) + "<extra></extra>"
    ))

    fig.update_layout(
        title=f"Top {top_n} 平均解决时间 - Assignee",
        xaxis_title="Assignee",
        yaxis_title="平均周期 (天)",
        height=450,
        margin=dict(l=60, r=40, t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45)

    return fig


def create_cycle_time_by_label_chart(df: pd.DataFrame,
                                      label_column: str = "labels",
                                      top_n: int = 15) -> go.Figure:
    """
    按 label 分类的平均解决时间柱状图
    """
    # 只统计已解决的 issue
    resolved_df = df[df["cycle_time_days"].notna()].copy()

    if len(resolved_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无已解决的 Issue",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f"Top {top_n} 平均解决时间 - Label",
            height=400,
        )
        return fig

    # 展开 labels 列（每个 issue 可能有多个 label）
    label_stats = []
    for _, row in resolved_df.iterrows():
        labels = row.get(label_column, [])
        if isinstance(labels, list):
            for label in labels:
                if label:
                    label_stats.append({"label": label, "cycle_days": row["cycle_time_days"]})

    if not label_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无 Label 数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f"Top {top_n} 平均解决时间 - Label",
            height=400,
        )
        return fig

    label_df = pd.DataFrame(label_stats)
    label_agg = label_df.groupby("label").agg({"cycle_days": ["mean", "count", "std"]}).reset_index()
    label_agg.columns = ["label", "avg_days", "count", "std"]
    label_agg = label_agg[label_agg["count"] >= 1].sort_values("avg_days", ascending=False).head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=label_agg["label"],
        y=label_agg["avg_days"],
        marker=dict(
            color=label_agg["avg_days"],
            colorscale="RdYlGn_r",
            showscale=True,
        ),
        text=label_agg["avg_days"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>平均周期：%{y:.1f}天<br>Issue 数：%{customdata}<extra></extra>",
        customdata=label_agg["count"]
    ))

    fig.update_layout(
        title=f"Top {top_n} 平均解决时间 - Label",
        xaxis_title="Label",
        yaxis_title="平均周期 (天)",
        height=450,
        margin=dict(l=60, r=40, t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45)

    return fig


def create_trend_chart(df: pd.DataFrame,
                        created_column: str = "created",
                        resolved_column: str = "resolutiondate") -> go.Figure:
    """
    创建 vs 解决趋势图（按月）
    """
    # 转换日期列
    df = df.copy()

    # 创建日期统计
    if created_column in df.columns:
        df["_created_month"] = pd.to_datetime(df[created_column], errors="coerce").dt.to_period("M")
        created_monthly = df.groupby("_created_month").size().reset_index(name="created_count")
        created_monthly["_month"] = created_monthly["_created_month"].apply(lambda x: x.to_timestamp())
    else:
        created_monthly = pd.DataFrame()

    # 解决日期统计
    if resolved_column in df.columns:
        df["_resolved_month"] = pd.to_datetime(df[resolved_column], errors="coerce").dt.to_period("M")
        resolved_monthly = df.groupby("_resolved_month").size().reset_index(name="resolved_count")
        resolved_monthly["_month"] = resolved_monthly["_resolved_month"].apply(lambda x: x.to_timestamp())
    else:
        resolved_monthly = pd.DataFrame()

    if len(created_monthly) == 0 and len(resolved_monthly) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无日期数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="创建 vs 解决趋势",
            height=400,
        )
        return fig

    # 合并数据
    all_months = pd.concat([
        created_monthly[["_month"]].drop_duplicates() if len(created_monthly) > 0 else pd.DataFrame(columns=["_month"]),
        resolved_monthly[["_month"]].drop_duplicates() if len(resolved_monthly) > 0 else pd.DataFrame(columns=["_month"])
    ]).drop_duplicates().sort_values("_month")

    trend_df = all_months.merge(created_monthly[["_month", "created_count"]], on="_month", how="left")
    trend_df = trend_df.merge(resolved_monthly[["_month", "resolved_count"]], on="_month", how="left")
    trend_df = trend_df.fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["_month"],
        y=trend_df["created_count"],
        mode="lines+markers",
        name="创建",
        line=dict(color="blue", width=2),
        marker=dict(size=8),
        hovertemplate="月份：%{x|%Y-%m}<br>创建数：%{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=trend_df["_month"],
        y=trend_df["resolved_count"],
        mode="lines+markers",
        name="解决",
        line=dict(color="green", width=2),
        marker=dict(size=8),
        hovertemplate="月份：%{x|%Y-%m}<br>解决数：%{y}<extra></extra>"
    ))

    fig.update_layout(
        title="创建 vs 解决趋势",
        xaxis_title="月份",
        yaxis_title="数量",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat="%Y-%m", tickangle=-45)

    return fig


def create_resolution_rate_by_status_chart(df: pd.DataFrame,
                                             status_column: str = "status") -> go.Figure:
    """
    按状态分类的解决率饼图/环形图
    """
    status_counts = df[status_column].value_counts()

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker=dict(
            colorscale="Set2",
        ),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>数量：%{value}<br>占比：%{percent}<extra></extra>"
    ))

    fig.update_layout(
        title="状态分布占比",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_kpi_cards(kpi_result) -> str:
    """
    生成 KPI 指标卡片的 HTML
    使用 Streamlit 的 metric 格式
    """
    # 返回一个字典，供 Streamlit 的 st.metric 使用
    cards = {
        "总 Issue 数": {"value": kpi_result.total_count, "delta": None},
        "已解决数": {"value": kpi_result.resolved_count, "delta": None},
        "已关闭数": {"value": kpi_result.closed_count, "delta": None},
        "解决率": {"value": f"{kpi_result.resolution_rate:.1f}%", "delta": None},
        "平均周期": {"value": f"{kpi_result.avg_cycle_days:.1f}天" if kpi_result.avg_cycle_days else "N/A", "delta": None},
        "Critical/High": {"value": kpi_result.critical_high_count, "delta": None},
        "Critical/High 修复率": {"value": f"{kpi_result.critical_high_fix_rate:.1f}%", "delta": None},
        "Overdue": {"value": kpi_result.overdue_count, "delta": None},
    }
    return cards


def render_kpi_dashboard(df: pd.DataFrame) -> dict:
    """
    渲染完整的 KPI 仪表盘

    Args:
        df: 选中的 issue 数据（包含 cycle_time_days 等字段）

    Returns:
        包含所有图表的字典
    """
    from data_processing import calculate_kpis

    # 计算 KPI 指标
    kpi_result = calculate_kpis(df)

    # 生成 KPI 卡片数据
    cards = create_kpi_cards(kpi_result)

    # 生成图表
    charts = {
        "status_distribution": create_status_distribution_chart(df),
        "cycle_time_distribution": create_cycle_time_distribution_chart(df),
        "cycle_time_by_assignee": create_cycle_time_by_assignee_chart(df),
        "cycle_time_by_label": create_cycle_time_by_label_chart(df),
        "trend": create_trend_chart(df),
        "resolution_rate_pie": create_resolution_rate_by_status_chart(df),
    }

    return {
        "kpi_result": kpi_result,
        "cards": cards,
        "charts": charts,
    }
