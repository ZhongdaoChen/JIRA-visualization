"""
Flask 图表服务
接收 Streamlit 传来的 DataFrame JSON，用 ECharts 渲染图表并返回完整 HTML。
启动方式：python chart_server.py（默认监听 127.0.0.1:5050）
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ── ECharts CDN ──────────────────────────────────────────────────────────────
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"

# ── 公共 HTML 模板 ─────────────────────────────────────────────────────────────
PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e1117; color: #fafafa; font-family: sans-serif; padding: 12px; }}
  .grid {{ display: grid; gap: 16px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .chart-box {{ background: #1a1c28; border-radius: 10px; padding: 12px; }}
  .chart-title {{ font-size: 13px; color: #aaa; margin-bottom: 6px; }}
  .chart {{ width: 100%; height: 300px; }}
  .chart-wide {{ width: 100%; height: 320px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }}
  .kpi-card {{ background: #1a1c28; border-radius: 10px; padding: 14px 10px; text-align: center; }}
  .kpi-label {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 22px; font-weight: bold; color: #fff; }}
  .kpi-value.rate {{ color: #4ade80; }}
</style>
</head>
<body>
<script src="{echarts_cdn}"></script>
{body}
<script>
{scripts}
</script>
</body>
</html>"""


def _df_from_request() -> pd.DataFrame:
    """从 POST JSON 中还原 DataFrame。"""
    data = request.get_json(force=True)
    df = pd.DataFrame(data.get("records", []))
    for col in ["created", "updated", "resolutiondate", "duedate", "created_date", "resolved_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "cycle_time_days" in df.columns:
        df["cycle_time_days"] = pd.to_numeric(df["cycle_time_days"], errors="coerce")
    return df


def _safe_list(series: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in series]


# ── 图表数据构建函数 ────────────────────────────────────────────────────────────

def _status_bar_option(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts()
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": counts.index.tolist(),
                  "axisLabel": {"color": "#ccc", "rotate": 30}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"}},
        "series": [{
            "type": "bar",
            "data": counts.values.tolist(),
            "itemStyle": {"color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [{"offset": 0, "color": "#6366f1"}, {"offset": 1, "color": "#3b82f6"}]
            }},
            "label": {"show": True, "position": "top", "color": "#fff"}
        }]
    }


def _cycle_histogram_option(df: pd.DataFrame) -> dict:
    valid = df["cycle_time_days"].dropna().tolist()
    if not valid:
        return {"title": {"text": "暂无已解决 Issue", "textStyle": {"color": "#888"}, "left": "center", "top": "center"}}

    avg = pd.Series(valid).mean()
    median = pd.Series(valid).median()

    # 手动分 20 个 bin
    min_v, max_v = min(valid), max(valid)
    bin_size = max((max_v - min_v) / 20, 1)
    bins: dict[int, int] = {}
    for v in valid:
        b = int((v - min_v) / bin_size)
        bins[b] = bins.get(b, 0) + 1
    x_data = [round(min_v + i * bin_size) for i in sorted(bins)]
    y_data = [bins[i] for i in sorted(bins)]

    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x_data, "name": "天",
                  "axisLabel": {"color": "#ccc"}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"}},
        "series": [
            {"type": "bar", "data": y_data,
             "itemStyle": {"color": "#38bdf8"},
             "label": {"show": False}},
        ],
        "markLine": {
            "symbol": "none",
            "data": [
                {"xAxis": avg, "label": {"formatter": f"均值 {avg:.1f}d", "color": "#f87171"}, "lineStyle": {"color": "#f87171", "type": "dashed"}},
                {"xAxis": median, "label": {"formatter": f"中位 {median:.1f}d", "color": "#4ade80"}, "lineStyle": {"color": "#4ade80", "type": "dashed"}},
            ]
        }
    }


def _assignee_bar_option(df: pd.DataFrame, top_n: int = 15) -> dict:
    resolved = df[df["cycle_time_days"].notna()].copy()
    if resolved.empty:
        return {"title": {"text": "暂无已解决 Issue", "textStyle": {"color": "#888"}, "left": "center", "top": "center"}}
    agg = (resolved.groupby("assignee")["cycle_time_days"]
           .agg(["mean", "count"]).reset_index()
           .sort_values("mean", ascending=False).head(top_n))
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": agg["assignee"].tolist(),
                  "axisLabel": {"color": "#ccc", "rotate": 30}},
        "yAxis": {"type": "value", "name": "天", "axisLabel": {"color": "#ccc"}},
        "series": [{
            "type": "bar",
            "data": [round(v, 1) for v in agg["mean"]],
            "itemStyle": {"color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [{"offset": 0, "color": "#f87171"}, {"offset": 1, "color": "#fb923c"}]
            }},
            "label": {"show": True, "position": "top", "color": "#fff"}
        }]
    }


def _label_bar_option(df: pd.DataFrame, top_n: int = 15) -> dict:
    resolved = df[df["cycle_time_days"].notna()].copy()
    if resolved.empty:
        return {"title": {"text": "暂无已解决 Issue", "textStyle": {"color": "#888"}, "left": "center", "top": "center"}}
    rows = []
    for _, row in resolved.iterrows():
        for lbl in (row.get("labels") or []):
            if lbl:
                rows.append({"label": lbl, "days": row["cycle_time_days"]})
    if not rows:
        return {"title": {"text": "暂无 Label 数据", "textStyle": {"color": "#888"}, "left": "center", "top": "center"}}
    agg = (pd.DataFrame(rows).groupby("label")["days"]
           .mean().reset_index()
           .sort_values("days", ascending=False).head(top_n))
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": agg["label"].tolist(),
                  "axisLabel": {"color": "#ccc", "rotate": 30}},
        "yAxis": {"type": "value", "name": "天", "axisLabel": {"color": "#ccc"}},
        "series": [{
            "type": "bar",
            "data": [round(v, 1) for v in agg["days"]],
            "itemStyle": {"color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [{"offset": 0, "color": "#a78bfa"}, {"offset": 1, "color": "#818cf8"}]
            }},
            "label": {"show": True, "position": "top", "color": "#fff"}
        }]
    }


def _trend_line_option(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["_cm"] = pd.to_datetime(df["created"], errors="coerce").dt.to_period("M")
    df["_rm"] = pd.to_datetime(df["resolutiondate"], errors="coerce").dt.to_period("M")

    created_m = df.groupby("_cm").size().reset_index(name="c")
    resolved_m = df.groupby("_rm").size().reset_index(name="r")

    all_months = sorted(set(
        created_m["_cm"].dropna().tolist() + resolved_m["_rm"].dropna().tolist()
    ))
    labels = [str(m) for m in all_months]

    cm_map = dict(zip(created_m["_cm"], created_m["c"]))
    rm_map = dict(zip(resolved_m["_rm"], resolved_m["r"]))
    c_data = [int(cm_map.get(m, 0)) for m in all_months]
    r_data = [int(rm_map.get(m, 0)) for m in all_months]

    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["创建", "解决"], "textStyle": {"color": "#ccc"}},
        "xAxis": {"type": "category", "data": labels,
                  "axisLabel": {"color": "#ccc", "rotate": 30}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"}},
        "series": [
            {"name": "创建", "type": "line", "data": c_data,
             "smooth": True, "lineStyle": {"color": "#38bdf8", "width": 2},
             "itemStyle": {"color": "#38bdf8"}},
            {"name": "解决", "type": "line", "data": r_data,
             "smooth": True, "lineStyle": {"color": "#4ade80", "width": 2},
             "itemStyle": {"color": "#4ade80"}},
        ]
    }


def _pie_option(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts()
    palette = ["#6366f1", "#38bdf8", "#4ade80", "#fb923c", "#f87171",
               "#a78bfa", "#facc15", "#34d399", "#f472b6", "#94a3b8"]
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": "#ccc"}},
        "series": [{
            "type": "pie", "radius": ["35%", "65%"],
            "data": [{"name": k, "value": int(v)} for k, v in zip(counts.index, counts.values)],
            "itemStyle": {"borderRadius": 4},
            "label": {"color": "#ccc"},
            "color": palette
        }]
    }


# ── 路由 ──────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/charts", methods=["POST"])
def charts():
    """主接口：接收 DataFrame JSON，返回完整图表 HTML。"""
    df = _df_from_request()

    charts_def = [
        ("status_bar",   "状态分布",            _status_bar_option(df),      "chart"),
        ("cycle_hist",   "解决周期分布",         _cycle_histogram_option(df), "chart"),
        ("assignee_bar", "Assignee 平均解决时间", _assignee_bar_option(df),   "chart"),
        ("label_bar",    "Label 平均解决时间",    _label_bar_option(df),      "chart"),
        ("trend_line",   "创建 vs 解决趋势",      _trend_line_option(df),     "chart-wide"),
        ("pie_chart",    "状态占比",             _pie_option(df),             "chart"),
    ]

    divs = ""
    scripts = ""

    # 第一行：2 列
    divs += '<div class="grid grid-2">'
    for cid, title, option, cls in charts_def[:2]:
        divs += f'<div class="chart-box"><div class="chart-title">{title}</div><div id="{cid}" class="{cls}"></div></div>'
        scripts += f'echarts.init(document.getElementById("{cid}"), "dark").setOption({json.dumps(option, ensure_ascii=False)});\n'
    divs += '</div>'

    # 第二行：2 列
    divs += '<div class="grid grid-2" style="margin-top:16px">'
    for cid, title, option, cls in charts_def[2:4]:
        divs += f'<div class="chart-box"><div class="chart-title">{title}</div><div id="{cid}" class="{cls}"></div></div>'
        scripts += f'echarts.init(document.getElementById("{cid}"), "dark").setOption({json.dumps(option, ensure_ascii=False)});\n'
    divs += '</div>'

    # 第三行：趋势图（全宽）+ 饼图（右侧）
    divs += '<div class="grid grid-2" style="margin-top:16px">'
    for cid, title, option, cls in charts_def[4:]:
        divs += f'<div class="chart-box"><div class="chart-title">{title}</div><div id="{cid}" class="{cls}"></div></div>'
        scripts += f'echarts.init(document.getElementById("{cid}"), "dark").setOption({json.dumps(option, ensure_ascii=False)});\n'
    divs += '</div>'

    html = PAGE_TEMPLATE.format(
        echarts_cdn=ECHARTS_CDN,
        body=divs,
        scripts=scripts,
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
