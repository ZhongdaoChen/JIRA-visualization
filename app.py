import os
from typing import List, Optional

import plotly.express as px
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from baidu_llm import QwenClient, QwenConfig
from data_processing import normalize_issues
from jira_client import JiraClient, JiraConfig


load_dotenv()


def build_jql(
    project_keys: List[str],
    start_date: str | None,
    end_date: str | None,
    reporters: List[str],
    assignees: List[str],
    labels: List[str],
    statuses: List[str],
) -> str:
    clauses: List[str] = []

    if project_keys:
        projects = ",".join(project_keys)
        clauses.append(f"project in ({projects})")

    if start_date:
        clauses.append(f"created >= '{start_date}'")
    if end_date:
        clauses.append(f"created <= '{end_date}'")

    if reporters:
        reporter_str = ",".join([f'"{r}"' for r in reporters])
        clauses.append(f"reporter in ({reporter_str})")

    if assignees:
        assignee_str = ",".join([f'"{a}"' for a in assignees])
        clauses.append(f"assignee in ({assignee_str})")

    # Label 在 JQL 中只支持精确匹配，这里改为在应用层做模糊过滤，
    # 因此不在 JQL 中添加 labels 条件。

    if statuses:
        status_str = ",".join([f'"{s}"' for s in statuses])
        clauses.append(f"status in ({status_str})")

    if not clauses:
        return "ORDER BY created DESC"

    return " AND ".join(clauses) + " ORDER BY created DESC"


def sanitize_jql(jql: str) -> str:
    """
    清理 JQL 中的特殊字符，确保 JIRA API 可以正确解析。
    主要处理：相对日期表达式如 +1m、+30d、endOfMonth(+1M) 等转换为具体日期。
    """
    import re
    from datetime import datetime, timedelta

    # 处理 endOfMonth(+NM) 的模式（N 个月后的月末）
    def replace_endOfMonth_plus(match):
        months = int(match.group(1))
        # 计算 N 个月后的日期
        now = datetime.now()
        future_month = now.month + months
        future_year = now.year + (future_month - 1) // 12
        future_month = ((future_month - 1) % 12) + 1
        # 计算该月的最后一天
        if future_month == 12:
            next_month = datetime(future_year + 1, 1, 1)
        else:
            next_month = datetime(future_year, future_month + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day.strftime("%Y-%m-%d")

    # 处理 endOfMonth() 无参数（本月末）
    def replace_endOfMonth_empty(match):
        now = datetime.now()
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day.strftime("%Y-%m-%d")

    # 处理 +Nm（N 个月后）的模式
    def replace_plus_months(match):
        months = int(match.group(1))
        future_date = datetime.now() + timedelta(days=months * 30)
        return future_date.strftime("%Y-%m-%d")

    # 处理 +Nd（N 天后）的模式
    def replace_plus_days(match):
        days = int(match.group(1))
        future_date = datetime.now() + timedelta(days=days)
        return future_date.strftime("%Y-%m-%d")

    # 处理 +Nw（N 周后）的模式
    def replace_plus_weeks(match):
        weeks = int(match.group(1))
        future_date = datetime.now() + timedelta(weeks=weeks)
        return future_date.strftime("%Y-%m-%d")

    # 按顺序处理：先处理 endOfMonth(+NM)，再处理 endOfMonth()，最后处理其他
    jql = re.sub(r'endOfMonth\(\+(\d+)M\)', replace_endOfMonth_plus, jql)
    jql = re.sub(r'endOfMonth\(\)', replace_endOfMonth_empty, jql)
    jql = re.sub(r'\+(\d+)m', replace_plus_months, jql)
    jql = re.sub(r'\+(\d+)d', replace_plus_days, jql)
    jql = re.sub(r'\+(\d+)w', replace_plus_weeks, jql)

    return jql


def interpret_nl_command(description: str, default_project: Optional[str], step: int = 1) -> dict:
    """
    使用阿里云 Qwen（DashScope）将自然语言描述转换为 JQL 或操作指令。

    Args:
        description: 用户的自然语言描述
        default_project: 默认项目 key
        step: 步骤编号，1=只生成 JQL，2=只生成操作指令（不包含 JQL）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    model = os.getenv("QWEN_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY，无法使用自然语言筛选。")

    client = QwenClient(QwenConfig(api_key=api_key, model=model, base_url=base_url))

    if step == 1:
        # 第一步：只生成 JQL
        system_prompt = (
            "你是一个 JIRA 操作解释器，负责根据自然语言生成 JSON 指令。\n"
            "只输出 JSON，不要输出任何解释文字。\n"
            "=== 第一步：搜索模式 ===\n"
            "JSON 结构如下：\n"
            "{\n"
            '  "mode": "query",  // 固定为 query\n'
            '  "jql": "..."      // 一条完整的 JQL，用于选中 issue\n'
            "}\n"
            "如果没有指定 project，而给出了默认项目 key，则使用 project = <默认项目> 作为条件之一。\n"
            "时间范围请转换为 created 字段的 >= 和 <= 形式，使用具体日期（如 2026-01-01），不要使用 +、- 等相对日期符号。\n"
            "JQL 中的值如果包含特殊字符（如 +、空格等），必须用双引号包裹。\n"
            "当 reporter 或 assignee 使用邮箱或用户名（例如包含 @ 的值）时，JQL 中必须使用双引号包裹。\n"
        )
    else:
        # 第二步：只生成操作指令（不包含 JQL）
        system_prompt = (
            "你是一个 JIRA 操作解释器，负责根据自然语言生成 JSON 指令。\n"
            "只输出 JSON，不要输出任何解释文字。\n"
            "=== 第二步：操作模式 ===\n"
            "用户已经选择了一组 issue，现在需要根据描述生成操作指令。\n"
            "不要包含 jql 字段，因为结果集已经确定。\n"
            "JSON 结构如下：\n"
            "{\n"
            '  "mode": "update",               // 固定为 update\n'
            '  "fields": { ... },                // (可选) 用于 /rest/api/2/issue PUT 请求的 fields 对象\n'
            '  "add_watchers": [...],            // (可选) 要添加的 watcher 用户名/邮箱列表\n'
            '  "add_participants": [...]         // (可选) 要添加的 Additional Viewer (JSM) accountId 列表\n'
            '  "transition": {...},              // (可选) 状态转换配置：{"to": "目标状态名", "fields": {...}, "comment": "转换时的评论"}\n'
            "}\n"
            "\n=== fields 字段的正确格式示例 ===\n"
            "1. 添加/更新评论 (comment): {\"comment\": {\"add\": [{\"body\": \"这里是评论内容\"}]}}\n"
            "2. 修改摘要 (summary): {\"summary\": \"新的摘要内容\"}\n"
            "3. 修改描述 (description): {\"description\": \"新的描述内容\"}\n"
            "4. 修改标签 (labels): {\"labels\": [{\"add\": \"新标签\"}, {\"remove\": \"旧标签\"}]} 或 {\"labels\": [\"label1\", \"label2\"]}\n"
            "5. 修改优先级 (priority): {\"priority\": {\"name\": \"High\"}}\n"
            "6. 修改处理人 (assignee): {\"assignee\": {\"accountId\": \"5b10a2844c20165700ede94g\"}}\n"
            "7. 添加 Fix Version: {\"fixVersions\": [{\"add\": [{\"name\": \"v1.0\"}]}]}\n"
            "8. 添加 Version: {\"versions\": [{\"add\": [{\"name\": \"v2.0\"}]}]}\n"
            "9. 添加组件 (components): {\"components\": [{\"add\": [{\"name\": \"组件名\"}]}]}\n"
            "10. 自定义字段 (customfield_xxxxx): {\"customfield_10010\": \"值\"} 或 {\"customfield_10010\": {\"value\": \"选项值\"}}\n"
            "\n重要提示：\n"
            "- comment 字段使用嵌套结构：{\"comment\": {\"add\": [{\"body\": \"评论内容\"}]}}\n"
            "- 用户相关字段 (assignee) 使用 accountId，格式：{\"assignee\": {\"accountId\": \"xxx\"}}\n"
            "- 选项类字段 (priority, severity 等) 格式：{\"priority\": {\"name\": \"High\"}}\n"
            "- 数组类字段 (labels, versions, components 等) 使用 add/remove 操作或直接赋值数组\n"
            "- status 字段不能直接更新，需要通过工作流转换 (transition)，不要将其包含在 fields 中\n"
            "- reporter、created、updated、id、key、status 字段不能通过 API 直接更新，不要将其包含在 fields 中\n"
            "\n=== transition 状态转换示例 ===\n"
            "- 关闭 issue: {\"to\": \"Closed\"}\n"
            "- 解决 issue: {\"to\": \"Resolved\", \"fields\": {\"resolution\": {\"name\": \"Fixed\"}}}\n"
            "- 开始进行：{\"to\": \"In Progress\"}\n"
            "- 转换状态并添加评论：{\"to\": \"Done\", \"comment\": \"已完成修复\"}\n"
            "\n重要提示：\n"
            "- transition 的 \"to\" 字段必须是目标状态的名称（如 \"Closed\", \"Done\", \"In Progress\"）\n"
        )

    if step == 1:
        if default_project:
            user_prompt = (
                f"默认的项目 key 是 {default_project}。\n"
                f"用户的需求是：{description}\n"
                "请给出一条完整的 JQL。"
            )
        else:
            user_prompt = (
                f"用户的需求是：{description}\n"
                "请给出一条完整的 JQL。"
            )
    else:
        # 第二步：不需要 default_project，因为操作是基于已选中的 issue
        user_prompt = (
            f"用户已选中了一组 issue，现在的操作需求是：{description}\n"
            "请生成操作指令（不要包含 jql 字段）。"
        )

    raw = client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    import json  # local import to keep dependency minimal

    raw_str = (raw or "").strip()
    try:
        return json.loads(raw_str)
    except Exception:
        # 有些模型会在 JSON 外面包一层说明文字或 ```json 代码块，这里尝试抽取第一个 { 到最后一个 } 之间的内容重新解析
        start = raw_str.find("{")
        end = raw_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_str[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # 如果仍然无法解析，抛出原始内容，方便在 UI 中排查
        raise RuntimeError(f"Qwen 返回的不是合法 JSON：{raw_str}")


def execute_update_operations(client, raw_issues, fields_to_update, cmd, email_to_user):
    """
    执行批量更新操作（第二步）
    """
    # 从 fields 中提取 comment，因为 comment 需要用单独的 API
    comments_to_add = []
    if fields_to_update.get("comment"):
        comment_data = fields_to_update["comment"]
        # 支持格式：{"comment": {"add": [{"body": "..."}]}}
        if isinstance(comment_data, dict) and "add" in comment_data:
            comments_to_add = comment_data["add"]
        # 支持格式：{"comment": [{"body": "..."}]}
        elif isinstance(comment_data, list):
            comments_to_add = comment_data
        # 从 fields 中移除 comment，避免传给 update_issue
        fields_to_update = {k: v for k, v in fields_to_update.items() if k != "comment"}

    has_fields_update = bool(fields_to_update)
    has_watchers = bool(cmd.get("add_watchers"))
    has_participants = bool(cmd.get("add_participants"))
    has_transition = bool(cmd.get("transition"))
    has_comments = bool(comments_to_add)

    if not has_fields_update and not has_watchers and not has_participants and not has_transition and not has_comments:
        return False, "没有提供任何更新操作"

    # ==============================
    # 第一步：收集需要解析的邮箱
    # ==============================
    emails_to_resolve = set()

    # 检查 fields 中的 assignee
    if fields_to_update.get("assignee"):
        assignee_val = fields_to_update["assignee"]

        # 情况 1: assignee 是字符串（邮箱）
        if isinstance(assignee_val, str) and "@" in assignee_val:
            emails_to_resolve.add(assignee_val)
        # 情况 2: assignee 是对象，包含邮箱
        elif isinstance(assignee_val, dict):
            if "accountId" in assignee_val and "@" in str(assignee_val.get("accountId", "")):
                emails_to_resolve.add(assignee_val["accountId"])
            if "name" in assignee_val and "@" in str(assignee_val.get("name", "")):
                emails_to_resolve.add(assignee_val["name"])

    # 检查 watchers 中的邮箱
    if has_watchers:
        for w in cmd["add_watchers"]:
            if "@" in w:
                emails_to_resolve.add(w)

    # 检查 participants 中的邮箱
    if has_participants:
        for p in cmd["add_participants"]:
            if "@" in p:
                emails_to_resolve.add(p)

    # 执行邮箱到用户信息的转换
    if emails_to_resolve:
        with st.spinner(f"正在解析邮箱为用户信息：{', '.join(emails_to_resolve)}"):
            for email in emails_to_resolve:
                try:
                    user_info = client.get_user_info(email)
                    email_to_user[email] = user_info
                    st.success(f"邮箱 {email} → name: `{user_info.get('name')}`, accountId: `{user_info.get('accountId')}`, key: `{user_info.get('key')}`")
                except Exception as e:
                    st.error(f"解析邮箱 {email} 失败：{e}")
                    email_to_user[email] = None

    # 第二步：根据转换结果，修正 fields 和 watchers/participants
    # 修正 assignee - 使用从 API 获取的 name 字段（完整邮箱格式）
    if fields_to_update.get("assignee"):
        assignee_val = fields_to_update["assignee"]
        resolved_name = None

        # 情况 1: assignee 是字符串（邮箱）
        if isinstance(assignee_val, str) and "@" in assignee_val:
            resolved_name = email_to_user.get(assignee_val, {}).get("name") or assignee_val

        # 情况 2: assignee 是对象，包含 accountId 或 name 且是邮箱格式
        elif isinstance(assignee_val, dict):
            val_to_resolve = assignee_val.get("accountId") or assignee_val.get("name")
            if val_to_resolve and "@" in val_to_resolve:
                resolved_name = email_to_user.get(val_to_resolve, {}).get("name") or val_to_resolve

        if resolved_name:
            fields_to_update["assignee"] = {"name": resolved_name}

    # 修正 participants - 使用 name（兼容 JIRA Server）
    if has_participants:
        final_participants = []
        for p in cmd["add_participants"]:
            if "@" in p:
                user_info = email_to_user.get(p)
                if user_info:
                    participant_id = user_info.get("name") or user_info.get("key") or user_info.get("accountId")
                    if participant_id:
                        final_participants.append(participant_id)
            else:
                final_participants.append(p)
        cmd["add_participants"] = final_participants

    # ==============================
    # 第三步：执行更新操作
    # ==============================
    # 添加评论 (使用单独的 API)
    if has_comments:
        comment_success = []
        comment_failed = []
        with st.spinner(f"正在为 {len(raw_issues)} 条 Issue 添加评论..."):
            for issue in raw_issues:
                key = issue.get("key")
                if not key:
                    continue
                for c in comments_to_add:
                    try:
                        client.add_comment(key, c["body"])
                        comment_success.append({"key": key, "comment": c["body"]})
                    except Exception as exc:
                        comment_failed.append({
                            "key": key,
                            "comment": c["body"],
                            "error": str(exc),
                        })
        if comment_success:
            st.success(f"添加评论成功：{len(comment_success)} 条")
        if comment_failed:
            st.markdown("#### 添加评论失败详情")
            st.dataframe(comment_failed, use_container_width=True, hide_index=True)

    # 字段更新
    if has_fields_update:
        updated = 0
        failed = 0
        error_rows = []
        with st.spinner("正在批量更新 Issue 字段，请稍候..."):
            for issue in raw_issues:
                key = issue.get("key")
                if not key:
                    continue
                try:
                    client.update_issue(key, fields_to_update)
                    updated += 1
                except Exception as exc:
                    failed += 1
                    error_rows.append(
                        {
                            "key": key,
                            "error": str(exc),
                        }
                    )
        st.success(
            f"字段更新：共尝试更新 {updated + failed} 条 Issue，成功 {updated} 条，失败 {failed} 条。"
        )
        if error_rows:
            st.markdown("#### 字段更新失败详情")
            st.dataframe(error_rows, use_container_width=True, hide_index=True)

    # 添加 Watchers
    if has_watchers:
        resolved_watchers = []
        for w in cmd["add_watchers"]:
            if "@" in w and email_to_user.get(w):
                resolved_watchers.append(email_to_user[w].get("name") or email_to_user[w].get("accountId"))
            else:
                resolved_watchers.append(w)

        with st.spinner(f"正在为 {len(raw_issues)} 条 Issue 添加 Watchers: {', '.join(resolved_watchers)}"):
            watcher_success = []
            watcher_failed = []
            for issue in raw_issues:
                key = issue.get("key")
                if not key:
                    continue
                for watcher in resolved_watchers:
                    try:
                        client.add_watcher(key, watcher)
                        watcher_success.append({"key": key, "watcher": watcher})
                    except Exception as exc:
                        watcher_failed.append({
                            "key": key,
                            "watcher": watcher,
                            "error": str(exc),
                        })
        if watcher_success:
            st.success(f"添加 Watcher 成功：{len(watcher_success)} 次操作")
        if watcher_failed:
            st.markdown("#### 添加 Watcher 失败详情")
            st.dataframe(watcher_failed, use_container_width=True, hide_index=True)

    # 添加 Request Participants (Additional Viewer)
    if has_participants:
        participants = cmd["add_participants"]

        if not participants:
            st.error("没有有效的 accountId 可用于添加 Additional Viewer")
        else:
            with st.spinner(f"正在为 {len(raw_issues)} 条 Issue 添加 Additional Viewers"):
                participant_success = []
                participant_failed = []
                for issue in raw_issues:
                    key = issue.get("key")
                    if not key:
                        continue
                    for account_id in participants:
                        try:
                            client.add_request_participant(key, account_id)
                            participant_success.append({"key": key, "accountId": account_id})
                        except Exception as exc:
                            participant_failed.append({
                                "key": key,
                                "accountId": account_id,
                                "error": str(exc),
                            })
            if participant_success:
                st.success(f"添加 Additional Viewer 成功：{len(participant_success)} 次操作")
            if participant_failed:
                st.markdown("#### 添加 Additional Viewer 失败详情")
                st.dataframe(participant_failed, use_container_width=True, hide_index=True)

    # 状态转换 (Transition)
    if has_transition:
        transition_config = cmd["transition"]
        target_status = transition_config.get("to", "")
        transition_fields = transition_config.get("fields") or {}
        transition_comment = transition_config.get("comment")

        if not target_status:
            st.error("transition 配置中缺少 'to' 字段（目标状态）")
        else:
            transitioned = 0
            transition_failed = 0
            error_rows = []

            with st.spinner(f"正在将 {len(raw_issues)} 条 Issue 转换到状态 '{target_status}'..."):
                for issue in raw_issues:
                    key = issue.get("key")
                    if not key:
                        continue
                    try:
                        client.transition_issue(
                            issue_key=key,
                            target_status=target_status,
                            fields=transition_fields,
                            comment=transition_comment,
                        )
                        transitioned += 1
                    except Exception as exc:
                        transition_failed += 1
                        error_rows.append({
                            "key": key,
                            "error": str(exc),
                        })

            st.success(
                f"状态转换：共尝试 {transitioned + transition_failed} 条，成功 {transitioned} 条，失败 {transition_failed} 条。"
            )
            if error_rows:
                st.markdown("#### 状态转换失败详情")
                st.dataframe(error_rows, use_container_width=True, hide_index=True)

    return True, None


def main() -> None:
    st.set_page_config(page_title="JIRA 可视化看板", layout="wide")
    st.title("JIRA 数据可视化看板")

    st.markdown(
        "在左侧输入 JIRA 连接信息与过滤条件，然后点击 **拉取数据** 生成可视化报表。"
    )

    # 初始化会话状态
    if "pending_cmd" not in st.session_state:
        st.session_state["pending_cmd"] = None
    # 两步交互流程状态
    if "step1_complete" not in st.session_state:
        st.session_state["step1_complete"] = False
    if "step1_jql" not in st.session_state:
        st.session_state["step1_jql"] = ""
    if "step1_issues" not in st.session_state:
        st.session_state["step1_issues"] = []
    if "step1_raw_issues" not in st.session_state:
        st.session_state["step1_raw_issues"] = []
    if "step2_operation" not in st.session_state:
        st.session_state["step2_operation"] = None
    if "step2_confirmed" not in st.session_state:
        st.session_state["step2_confirmed"] = False
    # 用户选中的 issue keys（通过 checkbox 选择）
    if "selected_issue_keys" not in st.session_state:
        st.session_state["selected_issue_keys"] = set()
    # 存储 issue selector 的 DataFrame 状态，避免 re-render 时丢失选择
    if "issue_selector_df" not in st.session_state:
        st.session_state["issue_selector_df"] = None

    # 提前获取 step1_complete 状态，供侧边栏使用
    step1_complete = st.session_state.get("step1_complete", False)

    with st.sidebar:
        st.header("JIRA 配置")
        default_base_url = os.getenv(
            "JIRA_BASE_URL", "https://jira.tools.3stripes.net/"
        )
        default_email = os.getenv("JIRA_EMAIL", "peter.chen2@adidas.com")
        default_pat = os.getenv("JIRA_PAT", "")

        base_url = st.text_input("JIRA Base URL", value=default_base_url)
        email = st.text_input("JIRA 账号邮箱（仅用于展示，无需参与鉴权）", value=default_email)
        pat = st.text_input(
            "JIRA Personal Access Token（PAT）",
            type="password",
            value=default_pat,
        )

        st.caption("提示：为了安全起见，Token 不会被持久化，请在本地安全保存。")

        st.markdown("---")
        st.header("自然语言筛选与更新")

        # 第一步：搜索条件输入框（始终显示）
        step1_description = st.text_area(
            "第一步：用自然语言描述搜索条件",
            placeholder="例如：帮我搜索整个 2025 年 GINFOSEC 项目下，assign 给 Peter.chen2@adidas.com 的 ticket。",
            key="step1_description_input"
        )

        # 第二步：操作描述输入框（第一步完成后显示）
        step2_description = ""
        if step1_complete:
            step2_description = st.text_area(
                "第二步：描述要对选中 issue 执行的操作",
                placeholder="例如：把这些 ticket 都 assign 给 Peter.chen2@adidas.com，或者添加评论说'已完成审查'。",
                key="step2_description_input"
            )

        use_nl = st.checkbox("使用上面的自然语言描述生成 JQL / 更新指令", value=True)

        max_results = st.number_input(
            "最大结果数（用于 JQL 查询）", min_value=50, max_value=5000, step=50, value=1000
        )

        # 按钮逻辑：第一步完成后显示两个按钮，否则只显示搜索按钮
        if step1_complete:
            col_search, col_operate = st.columns(2)
            with col_search:
                st.button("重新搜索", key="step1_button")
            with col_operate:
                st.button("生成操作指令", key="step2_button")
        else:
            st.button("搜索", key="step1_button")

    # 允许用户只输入域名（例如 jira.tools.3stripes.net），这里自动补全协议
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url.strip("/")

    if not (base_url and pat):
        st.error("请完整填写 JIRA Base URL 和 PAT。")
        return

    # 保存 base_url 到 session_state 供后续使用
    st.session_state["jira_base_url"] = base_url

    # 附件下载选项（第一步完成后显示，在主逻辑中定义以便后续使用）
    download_attachments = False
    if step1_complete:
        with st.sidebar:
            download_attachments = st.checkbox(
                "下载所有附件",
                value=False,
                help="勾选后将在执行操作时下载选中 issue 的所有附件到本地文件夹"
            )

    # 处理第一步：搜索
    if st.session_state.get("step1_button") and not step1_complete:
        if not (use_nl and step1_description.strip()):
            st.error("请在文本框中输入搜索条件。")
            return
        try:
            cmd = interpret_nl_command(step1_description.strip(), default_project=None, step=1)
        except Exception as e:
            st.error(f"解析搜索条件失败：{e}")
            return

        jql = cmd.get("jql", "")
        jql = sanitize_jql(jql)

        # 执行搜索
        try:
            client = JiraClient(JiraConfig(base_url=base_url, pat=pat))
            with st.spinner("从 JIRA 拉取数据中，请稍候..."):
                raw_issues = client.search_issues(jql=jql, max_results=max_results)
        except Exception as e:
            st.error(f"拉取数据失败：{e}")
            return

        if not raw_issues:
            st.warning("未查询到任何 Issue，请调整过滤条件重试。")
            return

        # 保存到 session state
        st.session_state["step1_complete"] = True
        st.session_state["step1_jql"] = jql
        st.session_state["step1_issues"] = normalize_issues(raw_issues)
        st.session_state["step1_raw_issues"] = raw_issues
        st.session_state["selected_issue_keys"] = set()  # 重置选中状态
        st.session_state["issue_selector_df"] = None  # 重置表格状态
        st.session_state["issue_selector_needs_init"] = True  # 标记需要重新初始化
        st.rerun()

    # 处理第二步：操作
    if st.session_state.get("step2_button") and step1_complete:
        if not (use_nl and step2_description.strip()):
            st.error("请在文本框中输入操作描述。")
            return

        try:
            cmd = interpret_nl_command(step2_description.strip(), default_project=None, step=2)
        except Exception as e:
            st.error(f"解析操作指令失败：{e}")
            return

        # 保存到 session state
        st.session_state["step2_operation"] = cmd
        st.rerun()

    # 处理重新搜索（重置状态）
    if st.session_state.get("step1_button") and step1_complete:
        st.session_state["step1_complete"] = False
        st.session_state["step1_jql"] = ""
        st.session_state["step1_issues"] = []
        st.session_state["step1_raw_issues"] = []
        st.session_state["step2_operation"] = None
        st.session_state["step2_confirmed"] = False
        st.session_state["selected_issue_keys"] = set()
        st.session_state["issue_selector_df"] = None
        st.session_state["issue_selector_needs_init"] = True
        st.rerun()

    # 显示第一步的结果
    if step1_complete:
        jql = st.session_state["step1_jql"]
        df = st.session_state["step1_issues"]
        raw_issues = st.session_state["step1_raw_issues"]
        base_url = st.session_state.get("jira_base_url", base_url)  # 获取 JIRA base URL

        st.subheader("第一步：搜索结果")
        st.write(f"共找到 **{len(raw_issues)}** 条 issue")
        st.code(jql, language="sql")

        # 构建带选择列的 DataFrame
        display_columns = [
            "key",
            "summary",
            "status",
            "assignee",
            "reporter",
            "created",
            "resolutiondate",
            "cycle_time_days",
            "labels",
        ]

        st.markdown("### 搜索结果预览（勾选要操作的 ticket）")

        # 检查是否需要重新初始化表格（新搜索结果或全选/取消操作）
        init_key = st.session_state.get("issue_selector_init_key", 0)
        needs_init = st.session_state.get("issue_selector_needs_init", True)

        if needs_init or st.session_state.get("issue_selector_df") is None:
            # 初始化表格
            display_df = df[display_columns].copy()
            current_selected = st.session_state.get("selected_issue_keys", set())
            display_df.insert(0, "Select", display_df["key"].apply(lambda x: x in current_selected))
            st.session_state["issue_selector_df"] = display_df
            st.session_state["issue_selector_needs_init"] = False
            st.session_state["issue_selector_init_key"] = st.session_state.get("issue_selector_init_key", 0) + 1

        # 使用 data_editor 显示带 checkbox 的表格
        # 为 key 列添加超链接（显示为 GINFOSEC-xxxxx，点击跳转到 JIRA）
        edited_df = st.data_editor(
            st.session_state["issue_selector_df"],
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "选择",
                    default=False,
                    help="勾选要操作的 ticket",
                ),
                "key": st.column_config.LinkColumn(
                    "key",
                    help="点击链接打开 JIRA issue",
                    validate=r"^GINFOSEC-",  # 验证格式为 GINFOSEC-xxxxx
                    url=f"{base_url}browse/{{key}}",  # URL 模板，{key} 会被替换为列值
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="issue_selector",
            disabled=display_columns,  # 只允许编辑 Select 列
            num_rows="fixed",  # 固定行数，避免重新排序
        )

        # 保存表格状态到 session_state
        st.session_state["issue_selector_df"] = edited_df

        # 从 edited_df 中获取用户选择并保存到 session state
        new_selected = set(edited_df[edited_df["Select"]]["key"].tolist())
        old_selected = st.session_state.get("selected_issue_keys", set())
        if new_selected != old_selected:
            st.session_state["selected_issue_keys"] = new_selected

        # 显示选中统计
        selected_count = len(st.session_state["selected_issue_keys"])
        if selected_count > 0:
            st.success(f"已选中 **{selected_count}** 条 issue")
        else:
            st.info("请勾选要操作的 ticket，或点击'全选/取消全选'按钮")

        # 全选/取消全选按钮
        all_keys = set(df["key"].tolist())
        is_all_selected = st.session_state["selected_issue_keys"] == all_keys

        col1, col2 = st.columns(2)
        with col1:
            # 使用不同的 key 区分全选和取消全选
            if is_all_selected:
                if st.button("取消全选", key="deselect_all_btn"):
                    st.session_state["selected_issue_keys"] = set()
                    display_df = df[display_columns].copy()
                    display_df.insert(0, "Select", False)
                    display_df["key"] = display_df["key"].apply(lambda x: f"{base_url}browse/{x}")
                    st.session_state["issue_selector_df"] = display_df
                    st.session_state["issue_selector_needs_init"] = True
                    st.rerun()
            else:
                if st.button("全选", key="select_all_btn"):
                    st.session_state["selected_issue_keys"] = all_keys
                    display_df = df[display_columns].copy()
                    display_df.insert(0, "Select", True)
                    display_df["key"] = display_df["key"].apply(lambda x: f"{base_url}browse/{x}")
                    st.session_state["issue_selector_df"] = display_df
                    st.session_state["issue_selector_needs_init"] = True
                    st.rerun()

        # 显示第二步操作输入
        st.markdown("---")
        st.subheader("第二步：描述操作")

        step2_operation = st.session_state.get("step2_operation")

        if step2_operation:
            mode = step2_operation.get("mode", "update")
            fields_to_update = step2_operation.get("fields") or {}
            cmd = step2_operation

            # 获取用户选中的 issue
            selected_keys = st.session_state.get("selected_issue_keys", set())
            selected_raw_issues = [i for i in raw_issues if i.get("key") in selected_keys]

            if not selected_raw_issues:
                st.warning("没有选中的 issue，请在上表中勾选要操作的 ticket")
            else:
                st.write(f"模式：**{mode}**")
                st.info(f"将对选中的 **{len(selected_raw_issues)}** 条 issue 执行操作")

                # 显示操作预览
                st.markdown("### 操作预览")

                # 评论预览
                if fields_to_update.get("comment"):
                    comment_data = fields_to_update["comment"]
                    st.markdown("### 添加评论 (每 Issue)")
                    st.json({
                        "method": "POST",
                        "url": f"{base_url}rest/api/2/issue/xxx/comment",
                        "body": comment_data
                    })

                # 字段更新预览
                if fields_to_update:
                    fields_copy = {k: v for k, v in fields_to_update.items() if k != "comment"}
                    if fields_copy:
                        st.markdown("### 更新字段 (每 Issue)")
                        st.json({
                            "method": "PUT",
                            "url": f"{base_url}rest/api/2/issue/xxx",
                            "body": {"fields": fields_copy}
                        })

                # Watchers 预览
                if cmd.get("add_watchers"):
                    st.markdown("### 添加 Watchers (每 Issue)")
                    st.json({
                        "method": "POST",
                        "url": f"{base_url}rest/api/2/issue/xxx/watchers",
                        "body": {"name": "<username>"}
                    })

                # Participants 预览
                if cmd.get("add_participants"):
                    st.markdown("### 添加 Additional Viewer (每 Issue)")
                    st.json({
                        "method": "PUT",
                        "url": f"{base_url}rest/api/2/issue/xxx",
                        "body": {"fields": {"customfield_15000": [{"name": "<email>"}]}}
                    })
                    st.caption("Additional Viewer 是一个 custom field (customfield_15000)，支持多用户。")

                # Transition 预览
                if cmd.get("transition"):
                    trans = cmd["transition"]
                    st.markdown("### 状态转换 (每 Issue)")
                    st.json({
                        "method": "POST",
                        "url": f"{base_url}rest/api/2/issue/xxx/transitions",
                        "body": {
                            "transition": {"id": "<transition_id>"},
                            "fields": trans.get("fields", {}),
                            "update": {"comment": [{"add": {"body": trans.get("comment", "")}}]} if trans.get("comment") else {}
                        }
                    })

                # 附件下载预览
                if download_attachments:
                    st.markdown("### 下载附件")
                    st.info(f"将下载选中 issue 的附件到当前目录下的 `jira_attachments` 文件夹")

                st.warning(f"⚠️ 注意：上述更新操作将对已选中的 {len(selected_raw_issues)} 条 issue 执行。")

                # 确认执行按钮
                if st.button("确认执行操作"):
                    email_to_user = {}
                    client = JiraClient(JiraConfig(base_url=base_url, pat=pat))
                    success, error = execute_update_operations(client, selected_raw_issues, fields_to_update, cmd, email_to_user)
                    if success:
                        st.success("操作执行完成！")
                    if error:
                        st.error(error)

                    # 执行附件下载
                    if download_attachments:
                        from datetime import datetime
                        # 所有附件下载到统一的 jira_attachments 目录
                        output_dir = os.path.join(os.getcwd(), "jira_attachments")
                        os.makedirs(output_dir, exist_ok=True)

                        with st.spinner(f"正在下载附件到 {output_dir} ..."):
                            all_success = []
                            all_failed = []
                            for issue in selected_raw_issues:
                                key = issue.get("key")
                                if not key:
                                    continue
                                try:
                                    result = client.download_issue_attachments(key, output_dir, flatten=True)
                                    all_success.extend(result.get("success", []))
                                    all_failed.extend(result.get("failed", []))
                                except Exception as e:
                                    all_failed.append({
                                        "key": key,
                                        "error": str(e),
                                    })

                            if all_success:
                                st.success(f"附件下载成功：{len(all_success)} 个文件，保存到：{output_dir}")
                                # 显示下载的文件列表
                                st.markdown("#### 下载的文件列表")
                                st.json(all_success[:50])  # 只显示前 50 个
                                if len(all_success) > 50:
                                    st.caption(f"... 还有 {len(all_success) - 50} 个文件")
                            if all_failed:
                                st.markdown("#### 下载失败详情")
                                st.dataframe(all_failed, use_container_width=True, hide_index=True)

                    st.session_state["step2_confirmed"] = True
                    st.rerun()

                # 重新搜索按钮
                if st.button("重新搜索"):
                    st.session_state["step1_complete"] = False
                    st.session_state["step1_jql"] = ""
                    st.session_state["step1_issues"] = []
                    st.session_state["step1_raw_issues"] = []
                    st.session_state["step2_operation"] = None
                    st.session_state["step2_confirmed"] = False
                    st.session_state["selected_issue_keys"] = set()
                    st.rerun()
        else:
            # 等待用户输入第二步操作描述
            st.info("请在上方输入框中描述要对这些 issue 执行的操作，然后点击'执行操作'按钮。")

    else:
        # 第一步尚未完成，显示初始提示
        st.info("请在左侧填写配置后点击 **搜索**。")
        return


if __name__ == "__main__":
    main()
