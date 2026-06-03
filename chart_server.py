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

# ── 公共调色板 ─────────────────────────────────────────────────────────────────
PALETTE = {
    "blue":    "#38bdf8",
    "green":   "#4ade80",
    "purple":  "#a78bfa",
    "indigo":  "#6366f1",
    "orange":  "#fb923c",
    "red":     "#f87171",
    "amber":   "#f59e0b",
    "teal":    "#34d399",
    "pink":    "#f472b6",
    "slate":   "#94a3b8",
    "dark":    "#64748b",
    "yellow":  "#facc15",
}

# 通用饼图/分类图轮换顺序
PALETTE_CYCLE = [
    PALETTE["indigo"], PALETTE["blue"],  PALETTE["green"],
    PALETTE["orange"], PALETTE["red"],   PALETTE["purple"],
    PALETTE["yellow"], PALETTE["teal"],  PALETTE["pink"],
    PALETTE["slate"],
]

def _grad(top: str, bottom: str) -> dict:
    """返回 ECharts 竖向线性渐变 dict（top → bottom）。"""
    return {
        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
        "colorStops": [{"offset": 0, "color": top}, {"offset": 1, "color": bottom}],
    }

# ── AppSec 常量（与 app.py 保持同步）──────────────────────────────────────────
_APPSEC_COLORS_MAP = {
    "SAST":               PALETTE["purple"],
    "Pentest":            PALETTE["blue"],
    "BugBounty":          PALETTE["green"],
    "Container Security": PALETTE["amber"],
    "DAST":               PALETTE["orange"],
    "Wiz":                PALETTE["teal"],
    "Ad-hoc":             PALETTE["indigo"],
    "Other":              PALETTE["slate"],
}
_APPSEC_CAT_ORDER = ["SAST", "Pentest", "BugBounty", "Container Security", "DAST", "Wiz", "Ad-hoc", "Other"]
_APPSEC_STATUS_COLORS = {
    "Open":     PALETTE["blue"],
    "Accepted": PALETTE["green"],
    "Closed":   PALETTE["slate"],
    "Reopen":   PALETTE["red"],
    "Other":    PALETTE["orange"],
}
_APPSEC_STATUS_ORDER = ["Open", "Accepted", "Closed", "Reopen", "Other"]

# ── 公共 HTML 模板 ─────────────────────────────────────────────────────────────
PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; color: #fafafa; font-family: sans-serif; padding: 12px; }}
  .grid {{ display: grid; gap: 16px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .chart-box {{ background: transparent; border-radius: 10px; padding: 12px; }}
  .chart-title {{ font-size: 13px; color: #aaa; margin-bottom: 6px; }}
  .chart {{ width: 100%; height: 300px; }}
  .chart-wide {{ width: 100%; height: 320px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }}
  .kpi-card {{ background: transparent; border-radius: 10px; padding: 14px 10px; text-align: center; }}
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


def _classify_appsec_status(status_str: str) -> str:
    """Map raw JIRA status string to Open/Accepted/Closed/Reopen/Other.
    IMPORTANT: check 'reopen' before 'open' to avoid substring false match."""
    s = str(status_str or "").lower()
    if "reopen" in s:
        return "Reopen"
    if "accepted" in s:
        return "Accepted"
    if "closed" in s or "done" in s or "resolved" in s:
        return "Closed"
    if "open" in s or "to do" in s or "in progress" in s or "new" in s:
        return "Open"
    return "Other"


def _parse_request():
    """Returns (df, raw_data_dict). raw_data_dict may contain 'months' etc."""
    data = request.get_json(force=True)
    df = pd.DataFrame(data.get("records", []))
    for col in ["created", "updated", "resolutiondate", "duedate", "created_date", "resolved_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "cycle_time_days" in df.columns:
        df["cycle_time_days"] = pd.to_numeric(df["cycle_time_days"], errors="coerce")
    return df, data


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
            "itemStyle": {"color": _grad(PALETTE["indigo"], PALETTE["blue"])},
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
             "itemStyle": {"color": _grad(PALETTE["blue"], PALETTE["indigo"])},
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
            "itemStyle": {"color": _grad(PALETTE["red"], PALETTE["orange"])},
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
            "itemStyle": {"color": _grad(PALETTE["purple"], PALETTE["indigo"])},
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
             "smooth": True, "lineStyle": {"color": PALETTE["blue"], "width": 2},
             "itemStyle": {"color": PALETTE["blue"]}},
            {"name": "解决", "type": "line", "data": r_data,
             "smooth": True, "lineStyle": {"color": PALETTE["green"], "width": 2},
             "itemStyle": {"color": PALETTE["green"]}},
        ]
    }


def _pie_option(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts()
    palette = PALETTE_CYCLE
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




# ── AppSec ECharts 脚本构建函数 ─────────────────────────────────────────────


def _appsec_pie_scripts(cid: str, labels: list, values: list, colors: list,
                        ticket_map: dict, title: str, jira_base_url: str = "") -> str:
    """Return a JS IIFE that renders an ECharts donut pie.
    On hover, shows a parent-DOM overlay with clickable JIRA ticket links."""
    ticket_map_json = json.dumps(ticket_map, ensure_ascii=False)
    base_url = jira_base_url.rstrip("/")
    overlay_id = f"pie_ov_{cid}"
    option = {
        "backgroundColor": "transparent",
        "title": {"text": title, "left": "center", "textStyle": {"color": "#ccc", "fontSize": 13}},
        "tooltip": {"show": False},
        "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": "#ccc"}, "top": "middle"},
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "center": ["60%", "55%"],
            "data": [{"name": l, "value": v} for l, v in zip(labels, values)],
            "color": colors,
            "itemStyle": {"borderRadius": 4},
            "label": {"color": "#ccc", "formatter": "{b}: {c}"},
        }],
    }
    option_json = json.dumps(option, ensure_ascii=False)
    return f"""(function() {{
  var TICKET_MAP = {ticket_map_json};
  var BASE_URL = '{base_url}';
  var OID = '{overlay_id}';

  // Inject overlay + styles into parent document (same origin)
  var _ov = parent.document.getElementById(OID);
  if (!_ov) {{
    var _st = parent.document.createElement('style');
    _st.textContent = [
      '#{overlay_id}{{position:fixed;z-index:99999;display:none;',
      'background:rgba(15,23,42,0.97);border:1px solid rgba(148,163,184,0.3);',
      'border-radius:8px;padding:10px 14px;max-height:360px;overflow-y:auto;',
      'min-width:160px;max-width:290px;box-shadow:0 8px 32px rgba(0,0,0,0.55);',
      'pointer-events:auto;}}',
      '#{overlay_id} .pov-title{{font-size:12px;color:#94a3b8;margin-bottom:8px;font-weight:600;}}',
      '#{overlay_id} a{{display:block;font-size:12px;color:#93c5fd;text-decoration:none;padding:2px 0;white-space:nowrap;}}',
      '#{overlay_id} a:hover{{color:#60a5fa;text-decoration:underline;}}',
      '#{overlay_id} .pov-more{{font-size:11px;color:#64748b;margin-top:4px;}}'
    ].join('');
    parent.document.head.appendChild(_st);
    _ov = parent.document.createElement('div');
    _ov.id = OID;
    parent.document.body.appendChild(_ov);
  }}

  // Track mouse position inside iframe
  var _mx = 0, _my = 0;
  document.addEventListener('mousemove', function(e) {{ _mx = e.clientX; _my = e.clientY; }});

  var _hideTimer = null;
  function showOverlay(name, keys) {{
    if (_hideTimer) {{ clearTimeout(_hideTimer); _hideTimer = null; }}
    var MAX = 30;
    var html = '<div class="pov-title">' + name + ' (' + keys.length + ' 条)</div>';
    keys.slice(0, MAX).forEach(function(k) {{
      html += BASE_URL
        ? '<a href="' + BASE_URL + '/browse/' + k + '" target="_blank">' + k + '</a>'
        : '<span style="display:block;font-size:12px;color:#ccc;padding:2px 0">' + k + '</span>';
    }});
    if (keys.length > MAX) html += '<div class="pov-more">...以及另外 ' + (keys.length - MAX) + ' 条</div>';
    _ov.innerHTML = html;

    var r = window.frameElement ? window.frameElement.getBoundingClientRect() : {{left:0,top:0}};
    var absX = r.left + _mx, absY = r.top + _my;
    var W = parent.innerWidth || 1200, H = parent.innerHeight || 900;
    var L = absX + 14; if (L + 300 > W - 10) L = absX - 300 - 14;
    var T = absY + 14; if (T + 380 > H - 10) T = absY - 380 - 14;
    _ov.style.left = L + 'px'; _ov.style.top = T + 'px';
    _ov.style.display = 'block';
  }}
  function hideOverlay() {{
    _hideTimer = setTimeout(function() {{
      if (!_ov.matches(':hover')) _ov.style.display = 'none';
    }}, 200);
  }}
  _ov.addEventListener('mouseleave', function() {{ _ov.style.display = 'none'; }});

  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption({option_json});
  chart.on('mouseover', function(params) {{
    if (params.componentType !== 'series') return;
    var keys = TICKET_MAP[params.name] || [];
    if (!keys.length) return;
    showOverlay(params.name, keys);
  }});
  chart.on('mouseout', hideOverlay);
  chart.on('globalout', hideOverlay);
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""


def _appsec_service_bar_scripts(cid: str, df: pd.DataFrame) -> str:
    """Return a JS IIFE that renders a stacked bar (resolved vs unresolved) with fix rate labels."""
    if df.empty or "_service" not in df.columns:
        option = {"title": {"text": "暂无数据", "left": "center", "top": "center",
                            "textStyle": {"color": "#888"}}}
        return f"""(function() {{
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption({json.dumps(option)});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""
    active_cats = [c for c in _APPSEC_CAT_ORDER if c in df["_service"].values]
    resolved_vals = []
    unresolved_vals = []
    rate_labels = []
    bar_colors = []
    for cat in active_cats:
        sub = df[df["_service"] == cat]
        total = len(sub)
        res = int(sub["resolutiondate"].notna().sum()) if "resolutiondate" in df.columns else 0
        unres = total - res
        resolved_vals.append(res)
        unresolved_vals.append(unres)
        rate_labels.append(f"{res / total * 100:.0f}%" if total > 0 else "0%")
        bar_colors.append(_APPSEC_COLORS_MAP.get(cat, "#94a3b8"))

    rate_labels_json = json.dumps(rate_labels, ensure_ascii=False)
    option = {
        "backgroundColor": "transparent",
        "title": {"text": "各服务 Ticket 数量与修复率", "left": "center",
                  "textStyle": {"color": "#ccc", "fontSize": 13}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["已修复", "未修复"], "textStyle": {"color": "#ccc"},
                   "top": 30, "right": 10},
        "grid": {"left": "5%", "right": "5%", "bottom": "8%", "top": "18%", "containLabel": True},
        "xAxis": {"type": "category", "data": active_cats,
                  "axisLabel": {"color": "#ccc", "rotate": 20}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.1)"}}},
        "series": [
            {
                "name": "已修复",
                "type": "bar",
                "stack": "total",
                "data": [{"value": v, "itemStyle": {"color": bar_colors[i]}}
                         for i, v in enumerate(resolved_vals)],
                "label": {"show": False},
            },
            {
                "name": "未修复",
                "type": "bar",
                "stack": "total",
                "data": unresolved_vals,
                "itemStyle": {"color": PALETTE["dark"]},
                "label": {"show": True, "position": "top", "color": "#f8fafc", "fontSize": 13},
            },
        ],
    }
    option_json = json.dumps(option, ensure_ascii=False)
    return f"""(function() {{
  var rateLabels = {rate_labels_json};
  var opt = {option_json};
  opt.series[1].label.formatter = function(params) {{ return rateLabels[params.dataIndex]; }};
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption(opt);
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""


def _appsec_monthly_bar_scripts(cid: str, months: list, created_vals: list,
                                 resolved_vals: list) -> str:
    """Return a JS IIFE for a grouped bar chart of created vs resolved per month."""
    option = {
        "backgroundColor": "transparent",
        "title": {"text": "每月创建 vs 已解决 Tickets（近 3 个月）", "left": "center",
                  "textStyle": {"color": "#ccc", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["创建", "已解决"], "textStyle": {"color": "#ccc"}, "top": 30},
        "grid": {"left": "5%", "right": "5%", "bottom": "8%", "top": "18%", "containLabel": True},
        "xAxis": {"type": "category", "data": months,
                  "axisLabel": {"color": "#ccc", "rotate": 20}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.1)"}}},
        "series": [
            {"name": "创建", "type": "bar", "data": created_vals,
             "itemStyle": {"color": _grad(PALETTE["blue"], PALETTE["indigo"])},
             "label": {"show": True, "position": "top", "color": "#fff"}},
            {"name": "已解决", "type": "bar", "data": resolved_vals,
             "itemStyle": {"color": _grad(PALETTE["green"], PALETTE["teal"])},
             "label": {"show": True, "position": "top", "color": "#fff"}},
        ],
    }
    option_json = json.dumps(option, ensure_ascii=False)
    return f"""(function() {{
  var opt = {option_json};
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption(opt);
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""


def _appsec_monthly_stacked_scripts(cid: str, months: list, df: pd.DataFrame) -> str:
    """Return a JS IIFE for a stacked bar chart of service composition per month."""
    if df.empty or "_service" not in df.columns:
        option = {"title": {"text": "暂无数据", "left": "center", "top": "center",
                            "textStyle": {"color": "#888"}}}
        return f"""(function() {{
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption({json.dumps(option)});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""
    col = "_created_month"
    series = []
    for cat in _APPSEC_CAT_ORDER:
        if col not in df.columns:
            counts = [0] * len(months)
        else:
            sub = df[df["_service"] == cat]
            grp = sub.groupby(col).size()
            counts = [int(grp.get(m, 0)) for m in months]
        if sum(counts) == 0:
            continue
        series.append({
            "name": cat,
            "type": "bar",
            "stack": "total",
            "data": counts,
            "itemStyle": {"color": _APPSEC_COLORS_MAP.get(cat, "#94a3b8")},
        })
    if not series:
        option = {"title": {"text": "暂无数据", "left": "center", "top": "center",
                            "textStyle": {"color": "#888"}}}
        return f"""(function() {{
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption({json.dumps(option)});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""
    legend_data = [s["name"] for s in series]
    option = {
        "backgroundColor": "transparent",
        "title": {"text": "每月新建 Tickets 服务构成（近 6 个月）", "left": "center",
                  "textStyle": {"color": "#ccc", "fontSize": 13}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": legend_data, "textStyle": {"color": "#ccc"}, "top": 30},
        "grid": {"left": "5%", "right": "5%", "bottom": "8%", "top": "22%", "containLabel": True},
        "xAxis": {"type": "category", "data": months,
                  "axisLabel": {"color": "#ccc", "rotate": 20}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#ccc"},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.1)"}}},
        "series": series,
    }
    option_json = json.dumps(option, ensure_ascii=False)
    return f"""(function() {{
  var opt = {option_json};
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption(opt);
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""


def _appsec_monthly_heatmap_scripts(cid: str, months: list, df: pd.DataFrame) -> str:
    """Return a JS IIFE for a heatmap of service × month creation counts."""
    services = _APPSEC_CAT_ORDER
    col = "_created_month"
    heat_data = []
    if not df.empty and "_service" in df.columns and col in df.columns:
        for row_idx, cat in enumerate(services):
            for col_idx, m in enumerate(months):
                count = int(((df["_service"] == cat) & (df[col] == m)).sum())
                heat_data.append([col_idx, row_idx, count])
    max_val = max((d[2] for d in heat_data), default=1) or 1

    months_json = json.dumps(months, ensure_ascii=False)
    services_json = json.dumps(services, ensure_ascii=False)
    heat_data_json = json.dumps(heat_data)
    option = {
        "backgroundColor": "transparent",
        "title": {"text": "服务 × 月份 创建量热力图", "left": "center",
                  "textStyle": {"color": "#ccc", "fontSize": 13}},
        "grid": {"left": "20%", "right": "10%", "bottom": "10%", "top": "15%"},
        "xAxis": {"type": "category", "data": months,
                  "axisLabel": {"color": "#ccc", "rotate": 20},
                  "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": services,
                  "axisLabel": {"color": "#ccc"},
                  "splitArea": {"show": True}},
        "visualMap": {
            "min": 0, "max": max_val,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "1%",
            "inRange": {"color": ["#1e293b", PALETTE["blue"]]},
            "textStyle": {"color": "#ccc"},
        },
        "series": [{
            "type": "heatmap",
            "data": heat_data,
            "label": {"show": True, "color": "#fff"},
        }],
    }
    option_json = json.dumps(option, ensure_ascii=False)
    return f"""(function() {{
  var months = {months_json};
  var services = {services_json};
  var opt = {option_json};
  opt.tooltip = {{
    formatter: function(params) {{
      var m = months[params.data[0]];
      var s = services[params.data[1]];
      return s + ' / ' + m + '<br>创建数：' + params.data[2];
    }}
  }};
  var chart = echarts.init(document.getElementById('{cid}'), 'dark');
  chart.setOption(opt);
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
"""


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


@app.route("/appsec_service_charts", methods=["POST"])
def appsec_service_charts():
    """AppSec service pie + status pie + service stacked bar."""
    df, raw = _parse_request()
    jira_base_url = raw.get("jira_base_url", "")

    # Build service ticket_map
    svc_ticket_map: dict[str, list] = {c: [] for c in _APPSEC_CAT_ORDER}
    status_ticket_map: dict[str, list] = {s: [] for s in _APPSEC_STATUS_ORDER}

    for _, row in df.iterrows():
        key = str(row.get("key") or "")
        svc = str(row.get("_service") or "Other")
        if svc not in svc_ticket_map:
            svc = "Other"
        svc_ticket_map[svc].append(key)

        st_cat = _classify_appsec_status(str(row.get("status") or ""))
        status_ticket_map[st_cat].append(key)

    # Service pie data
    svc_active = [c for c in _APPSEC_CAT_ORDER if svc_ticket_map[c]]
    svc_values = [len(svc_ticket_map[c]) for c in svc_active]
    svc_colors = [_APPSEC_COLORS_MAP.get(c, "#94a3b8") for c in svc_active]

    # Status pie data
    st_active = [s for s in _APPSEC_STATUS_ORDER if status_ticket_map[s]]
    st_values = [len(status_ticket_map[s]) for s in st_active]
    st_colors = [_APPSEC_STATUS_COLORS.get(s, "#94a3b8") for s in st_active]

    scripts = (
        _appsec_pie_scripts("svc_pie", svc_active, svc_values, svc_colors,
                            {c: svc_ticket_map[c] for c in svc_active}, "服务类型分布", jira_base_url)
        + _appsec_pie_scripts("status_pie", st_active, st_values, st_colors,
                              {s: status_ticket_map[s] for s in st_active}, "修复状态分布", jira_base_url)
        + _appsec_service_bar_scripts("svc_bar", df)
    )

    body = (
        '<div class="grid grid-2">'
        '<div class="chart-box"><div class="chart-title">服务类型分布</div>'
        '<div id="svc_pie" style="width:100%;height:300px;"></div></div>'
        '<div class="chart-box"><div class="chart-title">修复状态分布</div>'
        '<div id="status_pie" style="width:100%;height:300px;"></div></div>'
        '</div>'
        '<div class="chart-box" style="margin-top:16px">'
        '<div class="chart-title">各服务 Ticket 数量与修复率</div>'
        '<div id="svc_bar" style="width:100%;height:350px;"></div>'
        '</div>'
    )

    html = PAGE_TEMPLATE.format(echarts_cdn=ECHARTS_CDN, body=body, scripts=scripts)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/appsec_monthly_charts", methods=["POST"])
def appsec_monthly_charts():
    """AppSec monthly: grouped bar + stacked by service + heatmap."""
    df, data = _parse_request()
    months = data.get("months", [])
    months_bar = data.get("months_bar", months)  # 创建 vs 已解决用近 3 个月

    # Compute _created_month if not already present
    if not df.empty and "created" in df.columns:
        df["_created_month"] = (
            pd.to_datetime(df["created"], errors="coerce")
            .dt.to_period("M").astype(str)
        )
    else:
        df["_created_month"] = ""

    if not df.empty and "resolutiondate" in df.columns:
        df["_resolved_month"] = (
            pd.to_datetime(df["resolutiondate"], errors="coerce")
            .dt.to_period("M").astype(str)
        )
    else:
        df["_resolved_month"] = ""

    df_created = df[df["_created_month"].isin(months)] if months else df
    df_resolved = df[df["_resolved_month"].isin(months)] if months else df

    if months_bar:
        created_vals = [int((df["_created_month"] == m).sum()) for m in months_bar]
        resolved_vals = [int((df["_resolved_month"] == m).sum()) for m in months_bar]
    else:
        created_vals = []
        resolved_vals = []

    scripts = (
        _appsec_monthly_bar_scripts("mon_bar", months_bar, created_vals, resolved_vals)
        + _appsec_monthly_stacked_scripts("mon_stacked", months, df_created)
        + _appsec_monthly_heatmap_scripts("mon_heat", months, df_created)
    )

    body = (
        '<div class="chart-box"><div class="chart-title">每月创建 vs 已解决 Tickets</div>'
        '<div id="mon_bar" style="width:100%;height:400px;"></div></div>'
        '<div class="chart-box" style="margin-top:16px">'
        '<div class="chart-title">每月新建 Tickets 服务构成</div>'
        '<div id="mon_stacked" style="width:100%;height:400px;"></div></div>'
        '<div class="chart-box" style="margin-top:16px">'
        '<div class="chart-title">服务 × 月份 创建量热力图</div>'
        '<div id="mon_heat" style="width:100%;height:420px;"></div></div>'
    )

    html = PAGE_TEMPLATE.format(echarts_cdn=ECHARTS_CDN, body=body, scripts=scripts)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
