import os
from typing import List, Optional

import plotly.express as px
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from baidu_llm import QwenClient, QwenConfig
from data_processing import normalize_issues, calculate_kpis
from jira_client import JiraClient, JiraConfig, TESTING_LINK_TYPES
from kpi_charts import render_kpi_dashboard


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


def enforce_project(jql: str, project: str = "GINFOSEC") -> str:
    """
    确保 JQL 中包含指定 project 条件。
    若 JQL 已包含 project = 或 project in，则不重复添加。
    """
    import re
    if re.search(r'\bproject\s*[=i]', jql, re.IGNORECASE):
        return jql
    return f'project = "{project}" AND {jql}'


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
            '  "mode": "query",       // 固定为 query\n'
            '  "jql": "...",          // 一条完整的 JQL，用于选中 issue\n'
            '  "resolve_links": true  // (可选) 若为 true，先用 jql 找到 ticket，再返回这些 ticket 通过 Testing discovered / Discovered while testing 关联的所有 tickets\n'
            "}\n"
            "无论用户是否指定 project，JQL 中**必须始终包含** project = \"GINFOSEC\" 作为第一个条件。\n"
            "时间范围请转换为 created 字段的 >= 和 <= 形式，使用具体日期（如 2026-01-01），不要使用 +、- 等相对日期符号。\n"
            "JQL 中的值如果包含特殊字符（如 +、空格等），必须用双引号包裹。\n"
            "当 reporter 或 assignee 使用邮箱或用户名（例如包含 @ 的值）时，JQL 中必须使用双引号包裹。\n"
            "\n=== resolve_links 使用规则（重要）===\n"
            "当用户的意图是：先通过 title/summary 等条件找到某些 tickets，再找这些 tickets 的关联 tickets（linked issues）时，\n"
            "必须同时输出 jql（用于找到中间 ticket）和 resolve_links: true。\n"
            "resolve_links 场景下的 jql 同样必须包含 project = \"GINFOSEC\"。\n"
            "例如：\n"
            '  用户说「查找和 title 含有 "xxx" 的 ticket 有关的所有 tickets」\n'
            '  → {"mode": "query", "jql": "project = \\"GINFOSEC\\" AND summary ~ \\"xxx\\"", "resolve_links": true}\n'
            '  用户说「找到和 GINFOSEC-123 linked 的所有 tickets」\n'
            '  → {"mode": "query", "jql": "project = \\"GINFOSEC\\" AND key = \\"GINFOSEC-123\\"", "resolve_links": true}\n'
            "注意：resolve_links 的场景下，jql 是用来找到「源 ticket」的，最终结果是源 ticket 的所有 Testing discovered / Discovered while testing 关联 tickets。\n"
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
            '  "link_to": "XXX-12345",           // (可选) 目标 issue key，用于将选中的 issue link 到目标 ticket\n'
            "}\n"
            "\n=== fields 字段的正确格式示例 ===\n"
            "1. 添加/更新评论 (comment): {\"comment\": {\"add\": [{\"body\": \"这里是评论内容\"}]}}\n"
            "2. 修改摘要 (summary): {\"summary\": \"新的摘要内容\"}\n"
            "3. 修改描述 (description): {\"description\": \"新的描述内容\"}\n"
            "4. 添加标签 (labels): 在原有基础上添加用 {\"labels\": [{\"add\": \"新标签\"}]}；覆盖所有标签用 {\"labels\": [\"label1\", \"label2\"]}\n"
            "   ⚠️ 注意：labels 必须是数组格式，添加/删除操作时每个操作是数组中的一个对象\n"
            "5. 修改优先级 (priority): {\"priority\": {\"name\": \"High\"}}\n"
            "6. 修改处理人 (assignee): {\"assignee\": {\"accountId\": \"5b10a2844c20165700ede94g\"}}\n"
            "7. 添加 Fix Version: {\"fixVersions\": [{\"add\": [{\"name\": \"v1.0\"}]}]}\n"
            "8. 添加 Version: {\"versions\": [{\"add\": [{\"name\": \"v2.0\"}]}]}\n"
            "9. 添加组件 (components): {\"components\": [{\"add\": [{\"name\": \"组件名\"}]}]}\n"
            "10. 自定义字段 (customfield_xxxxx): {\"customfield_10010\": \"值\"} 或 {\"customfield_10010\": {\"value\": \"选项值\"}}\n"
            "\n重要提示：\n"
            "- comment 字段使用嵌套结构：{\"comment\": {\"add\": [{\"body\": \"评论内容\"}]}}\n"
            "- 用户相关字段 (assignee, reporter) 使用 name，格式：{\"assignee\": {\"name\": \"邮箱或用户名\"}}, {\"reporter\": {\"name\": \"邮箱或用户名\"}}\n"
            "- 选项类字段 (priority, severity 等) 格式：{\"priority\": {\"name\": \"High\"}}\n"
            "- 数组类字段 (labels, versions, components 等) 使用 add/remove 操作或直接赋值数组\n"
            "- status 字段不能直接更新，需要通过工作流转换 (transition)，不要将其包含在 fields 中\n"
            "- created、updated、id、key 字段不能通过 API 直接更新，不要将其包含在 fields 中\n"
            "\n=== transition 状态转换示例 ===\n"
            "- 关闭 issue: {\"to\": \"Closed\"}\n"
            "- 解决 issue: {\"to\": \"Resolved\", \"fields\": {\"resolution\": {\"name\": \"Fixed\"}}}\n"
            "- 开始进行：{\"to\": \"In Progress\"}\n"
            "- 转换状态并添加评论：{\"to\": \"Done\", \"comment\": \"已完成修复\"}\n"
            "\n重要提示：\n"
            "- transition 的 \"to\" 字段必须是目标状态的名称（如 \"Closed\", \"Done\", \"In Progress\"）\n"
            "\n=== link_to 字段示例 ===\n"
            "- 链接到目标 ticket: {\"link_to\": \"GINFOSEC-94691\"}\n"
            "- 当用户说'把这些 ticket 都 link 到 xxx'时，使用此字段\n"
            "- Link Type 固定使用 \"Testing discovered\" 或 \"Discovered while testing\"\n"
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

    # 处理 labels 字段：如果是 add/remove 格式，需要移动到 update 对象中
    update_operations = {}
    if fields_to_update.get("labels"):
        labels_val = fields_to_update["labels"]
        # 检查是否是 add/remove 格式
        if isinstance(labels_val, list) and len(labels_val) > 0:
            # 格式 1: [{"add": "xxx"}] （数组）
            if isinstance(labels_val[0], dict) and ("add" in labels_val[0] or "remove" in labels_val[0]):
                update_operations["labels"] = labels_val
                fields_to_update = {k: v for k, v in fields_to_update.items() if k != "labels"}
        elif isinstance(labels_val, dict) and ("add" in labels_val or "remove" in labels_val):
            # 格式 2: {"add": "xxx"} （对象，转换为数组）
            update_operations["labels"] = [labels_val]
            fields_to_update = {k: v for k, v in fields_to_update.items() if k != "labels"}

    has_fields_update = bool(fields_to_update)
    has_update_operations = bool(update_operations)
    has_watchers = bool(cmd.get("add_watchers"))
    has_participants = bool(cmd.get("add_participants"))
    has_transition = bool(cmd.get("transition"))
    has_comments = bool(comments_to_add)
    has_link = bool(cmd.get("link_to"))

    if not has_fields_update and not has_update_operations and not has_watchers and not has_participants and not has_transition and not has_comments and not has_link:
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

    # 检查 reporter 中的邮箱
    if fields_to_update.get("reporter"):
        reporter_val = fields_to_update["reporter"]
        if isinstance(reporter_val, str) and "@" in reporter_val:
            emails_to_resolve.add(reporter_val)
        elif isinstance(reporter_val, dict):
            for key in ("accountId", "name"):
                v = reporter_val.get(key, "")
                if v and "@" in str(v):
                    emails_to_resolve.add(v)

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

    # 修正 reporter - 使用从 API 获取的 name 字段（完整邮箱格式）
    if fields_to_update.get("reporter"):
        reporter_val = fields_to_update["reporter"]
        resolved_reporter = None

        if isinstance(reporter_val, str) and "@" in reporter_val:
            resolved_reporter = email_to_user.get(reporter_val, {}).get("name") or reporter_val
        elif isinstance(reporter_val, dict):
            val_to_resolve = reporter_val.get("accountId") or reporter_val.get("name")
            if val_to_resolve and "@" in str(val_to_resolve):
                resolved_reporter = email_to_user.get(val_to_resolve, {}).get("name") or val_to_resolve
            else:
                resolved_reporter = val_to_resolve

        if resolved_reporter:
            fields_to_update["reporter"] = {"name": resolved_reporter}

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
    if has_fields_update or has_update_operations:
        updated = 0
        failed = 0
        error_rows = []
        with st.spinner("正在批量更新 Issue 字段，请稍候..."):
            for issue in raw_issues:
                key = issue.get("key")
                if not key:
                    continue
                try:
                    client.update_issue(key, fields=fields_to_update if has_fields_update else None, update=update_operations if has_update_operations else None)
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

    # 处理 Issue Link
    if has_link:
        target_key = cmd["link_to"]
        link_type = "Testing discovered"  # 固定的 link type

        with st.spinner(f"正在将 {len(raw_issues)} 条 Issue link 到 {target_key}..."):
            linked = 0
            link_failed = 0
            error_rows = []

            for issue in raw_issues:
                key = issue.get("key")
                if not key:
                    continue
                try:
                    client.link_issues_to_target(
                        source_issue_keys=[key],
                        target_issue_key=target_key,
                        link_type_name=link_type,
                    )
                    linked += 1
                except Exception as exc:
                    link_failed += 1
                    error_rows.append({
                        "key": key,
                        "error": str(exc),
                    })

            st.success(
                f"Issue Link：共尝试 {linked + link_failed} 条，成功 {linked} 条，失败 {link_failed} 条。"
            )
            if error_rows:
                st.markdown("#### Issue Link 失败详情")
                st.dataframe(error_rows, use_container_width=True, hide_index=True)

    return True, None


def create_appsec_service_pie_chart(df: pd.DataFrame):
    """
    按服务类型分类的饼图。悬停时显示数量、占比及该类别的 ticket 列表。
    分类规则（互斥，按优先级）：
      1. SAST:              reporter == Shervin.Aghdaei@adidas.com
                            AND assignee in (Jesse.Zhang / Du.Chen / Kiba.Yang /
                            John.Fu / Zone.Tian / David.Wei / Spencer.Shao /
                            Laura.Yuan / Jane.Lu / Newman.Xu)
      2. Pentest:           labels 含 ChaiTin_PenTests
      3. BugBounty:         labels 含 BugBounty
      4. Container Security:labels 含 GCA-Issues-Q1-Critical / ContainerSecurity / ContainerSecurityL1.3
      5. DAST:              labels 含 DAST
      6. Ad-hoc:            labels 为空且不属于 SAST
      7. Other:             有标签但不匹配以上任一类别
    """
    import plotly.graph_objects as go

    SAST_REPORTER = "shervin.aghdaei@adidas.com"
    SAST_ASSIGNEES = {
        "jesse.zhang@adidas.com", "du.chen@adidas.com", "kiba.yang@adidas.com",
        "john.fu@adidas.com", "zone.tian@adidas.com", "david.wei@adidas.com",
        "spencer.shao@adidas.com", "laura.yuan@adidas.com", "jane.lu@adidas.com",
        "newman.xu@adidas.com",
    }
    PENTEST_TAGS = {"ChaiTin_PenTests"}
    BUGBOUNTY_TAGS = {"BugBounty"}
    CONTAINER_TAGS = {"GCA-Issues-Q1-Critical", "ContainerSecurity", "ContainerSecurityL1.3"}
    DAST_TAGS = {"DAST"}

    CATEGORY_ORDER = ["SAST", "Pentest", "BugBounty", "Container Security", "DAST", "Ad-hoc", "Other"]
    tickets = {c: [] for c in CATEGORY_ORDER}

    for _, row in df.iterrows():
        raw_labels = row.get("labels") or []
        label_set = set(raw_labels) if isinstance(raw_labels, list) else set()
        reporter = str(row.get("reporter_name") or row.get("reporter") or "").lower()
        assignee = str(row.get("assignee_name") or row.get("assignee") or "").lower()
        key = str(row.get("key") or "")

        if reporter == SAST_REPORTER and assignee in SAST_ASSIGNEES:
            tickets["SAST"].append(key)
        elif label_set & PENTEST_TAGS:
            tickets["Pentest"].append(key)
        elif label_set & BUGBOUNTY_TAGS:
            tickets["BugBounty"].append(key)
        elif label_set & CONTAINER_TAGS:
            tickets["Container Security"].append(key)
        elif label_set & DAST_TAGS:
            tickets["DAST"].append(key)
        elif not label_set:
            tickets["Ad-hoc"].append(key)
        else:
            tickets["Other"].append(key)

    active_cats = [c for c in CATEGORY_ORDER if tickets[c]]
    values_list = [len(tickets[c]) for c in active_cats]

    MAX_DISPLAY = 15

    def _format_tickets(keys):
        lines = keys[:MAX_DISPLAY]
        text = "<br>".join(lines)
        if len(keys) > MAX_DISPLAY:
            text += f"<br>...以及另外 {len(keys) - MAX_DISPLAY} 条"
        return text

    customdata = [_format_tickets(tickets[c]) for c in active_cats]

    palette = ["#a78bfa", "#6366f1", "#38bdf8", "#4ade80", "#f59e0b", "#fb923c", "#94a3b8"]

    fig = go.Figure(go.Pie(
        labels=active_cats,
        values=values_list,
        hole=0.4,
        textinfo="label+value",
        textposition="auto",
        marker=dict(colors=palette[:len(active_cats)]),
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "数量：%{value}  占比：%{percent}<br>"
            "─────────────────<br>"
            "%{customdata}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="服务类型分布",
        height=450,
        margin=dict(l=40, r=160, t=60, b=40),
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    return fig


def create_appsec_status_chart(df: pd.DataFrame):
    """
    按修复状态分类的饼图（Open / Accepted / Closed / Reopen）。
    悬停时显示数量、占比及该状态下的 ticket 列表。
    """
    import plotly.graph_objects as go

    TARGET_STATUSES = ["Open", "Accepted", "Closed", "Reopen"]
    tickets = {s: [] for s in TARGET_STATUSES}
    tickets["Other"] = []

    for _, row in df.iterrows():
        val_lower = str(row.get("status") or "").lower()
        key = str(row.get("key") or "")
        matched = False
        for s in TARGET_STATUSES:
            if s.lower() in val_lower:
                tickets[s].append(key)
                matched = True
                break
        if not matched:
            tickets["Other"].append(key)

    active = [s for s in TARGET_STATUSES + ["Other"] if tickets[s]]
    values_list = [len(tickets[s]) for s in active]

    MAX_DISPLAY = 15

    def _format_tickets(keys):
        lines = keys[:MAX_DISPLAY]
        text = "<br>".join(lines)
        if len(keys) > MAX_DISPLAY:
            text += f"<br>...以及另外 {len(keys) - MAX_DISPLAY} 条"
        return text

    customdata = [_format_tickets(tickets[s]) for s in active]

    palette = ["#38bdf8", "#4ade80", "#94a3b8", "#f87171", "#fb923c"]

    fig = go.Figure(go.Pie(
        labels=active,
        values=values_list,
        hole=0.4,
        textinfo="label+value",
        textposition="auto",
        marker=dict(colors=palette[:len(active)]),
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "数量：%{value}  占比：%{percent}<br>"
            "─────────────────<br>"
            "%{customdata}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="修复状态分布（Open / Accepted / Closed / Reopen）",
        height=450,
        margin=dict(l=40, r=160, t=60, b=40),
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    return fig


def create_appsec_service_bar_chart(df: pd.DataFrame):
    """
    按 AppSec 服务类型的柱状图（堆叠：已修复 vs 未修复），柱顶显示修复率。
    分类规则（互斥，按优先级）：
      1. SAST:              reporter == Shervin.Aghdaei@adidas.com
                            AND assignee in (Jesse.Zhang / Du.Chen / Kiba.Yang /
                            John.Fu / Zone.Tian / David.Wei / Spencer.Shao /
                            Laura.Yuan / Jane.Lu / Newman.Xu)
      2. Pentest:           labels 含 ChaiTin_PenTests
      3. BugBounty:         labels 含 BugBounty
      4. Container Security:labels 含 GCA-Issues-Q1-Critical / ContainerSecurity / ContainerSecurityL1.3
      5. DAST:              labels 含 DAST
      6. Ad-hoc:            labels 为空且不属于 SAST
      7. Other:             有标签但不匹配以上任一类别
    """
    import plotly.graph_objects as go

    SAST_REPORTER = "shervin.aghdaei@adidas.com"
    SAST_ASSIGNEES = {
        "jesse.zhang@adidas.com", "du.chen@adidas.com", "kiba.yang@adidas.com",
        "john.fu@adidas.com", "zone.tian@adidas.com", "david.wei@adidas.com",
        "spencer.shao@adidas.com", "laura.yuan@adidas.com", "jane.lu@adidas.com",
        "newman.xu@adidas.com",
    }
    PENTEST_TAGS = {"ChaiTin_PenTests"}
    BUGBOUNTY_TAGS = {"BugBounty"}
    CONTAINER_TAGS = {"GCA-Issues-Q1-Critical", "ContainerSecurity", "ContainerSecurityL1.3"}
    DAST_TAGS = {"DAST"}

    CATEGORIES = ["Pentest", "BugBounty", "Container Security", "DAST", "SAST", "Ad-hoc", "Other"]
    total = {c: 0 for c in CATEGORIES}
    resolved = {c: 0 for c in CATEGORIES}

    def is_resolved(row):
        return pd.notna(row.get("resolutiondate")) and row.get("resolutiondate") is not None

    for _, row in df.iterrows():
        raw_labels = row.get("labels") or []
        label_set = set(raw_labels) if isinstance(raw_labels, list) else set()
        reporter = str(row.get("reporter_name") or row.get("reporter") or "").lower()
        assignee = str(row.get("assignee_name") or row.get("assignee") or "").lower()

        # 分类（优先级顺序）
        if reporter == SAST_REPORTER and assignee in SAST_ASSIGNEES:
            cat = "SAST"
        elif label_set & PENTEST_TAGS:
            cat = "Pentest"
        elif label_set & BUGBOUNTY_TAGS:
            cat = "BugBounty"
        elif label_set & CONTAINER_TAGS:
            cat = "Container Security"
        elif label_set & DAST_TAGS:
            cat = "DAST"
        elif not label_set:
            cat = "Ad-hoc"
        else:
            cat = "Other"

        total[cat] += 1
        if is_resolved(row):
            resolved[cat] += 1

    # 只保留有数据的类别，按预设顺序排列
    active_cats = [c for c in CATEGORIES if total[c] > 0]
    total_vals = [total[c] for c in active_cats]
    resolved_vals = [resolved[c] for c in active_cats]
    unresolved_vals = [total[c] - resolved[c] for c in active_cats]
    rate_labels = [
        f"{resolved[c] / total[c] * 100:.0f}%" if total[c] > 0 else "0%"
        for c in active_cats
    ]

    fig = go.Figure()

    # 已修复（绿色，底层）
    fig.add_trace(go.Bar(
        name="已修复",
        x=active_cats,
        y=resolved_vals,
        marker_color="#4ade80",
        hovertemplate="<b>%{x}</b><br>已修复：%{y}<extra></extra>",
    ))

    # 未修复（灰色，上层）
    fig.add_trace(go.Bar(
        name="未修复",
        x=active_cats,
        y=unresolved_vals,
        marker_color="#64748b",
        hovertemplate="<b>%{x}</b><br>未修复：%{y}<extra></extra>",
        text=rate_labels,
        textposition="outside",
        textfont=dict(size=13, color="#f8fafc"),
    ))

    fig.update_layout(
        title="各服务 Ticket 数量与修复率",
        xaxis_title="AppSec 服务",
        yaxis_title="Ticket 数量",
        barmode="stack",
        height=450,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def _render_overdue_metric(
    label: str,
    overdue_count: int,
    overdue_keys: list,
    jira_base_url: str,
) -> None:
    """
    渲染 Overdue KPI 卡片。
    有逾期 ticket 时，鼠标悬浮显示可点击的 ticket 链接列表。
    """
    import streamlit.components.v1 as _cv1

    if overdue_count == 0 or not overdue_keys:
        st.metric(label=label, value=overdue_count)
        return

    base = jira_base_url.rstrip("/")
    links_html = "".join(
        f'<a href="{base}/browse/{key}" target="_blank">{key}</a>'
        for key in overdue_keys
    )
    # 每行最多显示的 ticket 数，超出时显示滚动条
    tooltip_height = min(len(overdue_keys) * 26 + 16, 260)
    card_html = f"""
<style>
  .ov-wrap {{
    position: relative;
    font-family: sans-serif;
    display: inline-block;
    width: 100%;
  }}
  .ov-label {{
    font-size: 0.875rem;
    color: #888;
    margin-bottom: 4px;
  }}
  .ov-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #d9534f;
    cursor: default;
    user-select: none;
  }}
  .ov-hint {{
    font-size: 0.72rem;
    color: #aaa;
    margin-top: 2px;
  }}
  .ov-tooltip {{
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 9999;
    background: #1e1e2e;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px 10px;
    width: max-content;
    max-width: 320px;
    max-height: {tooltip_height}px;
    overflow-y: auto;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  }}
  .ov-tooltip a {{
    display: block;
    color: #4fc3f7;
    text-decoration: none;
    font-size: 0.85rem;
    padding: 2px 0;
    white-space: nowrap;
  }}
  .ov-tooltip a:hover {{
    text-decoration: underline;
    color: #81d4fa;
  }}
  .ov-wrap:hover .ov-tooltip {{
    display: block;
  }}
</style>
<div class="ov-wrap">
  <div class="ov-label">{label}</div>
  <div class="ov-value">{overdue_count}</div>
  <div class="ov-hint">🖱 悬浮查看 tickets</div>
  <div class="ov-tooltip">{links_html}</div>
</div>
"""
    _cv1.html(card_html, height=90)


def main() -> None:
    st.set_page_config(page_title="JIRA 可视化看板", layout="wide")

    # 翻译字典
    TRANSLATIONS = {
        "zh": {
            "title": "JIRA 数据可视化看板",
            "jira_config": "🔧 JIRA 配置",
            "jira_base_url": "JIRA Base URL",
            "jira_email": "JIRA 账号邮箱",
            "jira_pat": "JIRA PAT",
            "token_caption": "Token 不会被持久化，请本地保存",
            "max_results": "最大结果数",
            "max_results_help": "每次搜索最多返回的 issue 数量",
            "nl_filter": "自然语言筛选与更新",
            "search_desc": "用自然语言描述搜索条件",
            "search_placeholder": "例如：帮我搜索整个 2025 年 GINFOSEC 项目下，assign 给 Peter.chen2@adidas.com 的 ticket。",
            "operation_desc": "描述要对选中 issue 执行的操作",
            "operation_placeholder": "例如：把这些 ticket 都 assign 给 Peter.chen2@adidas.com，或者添加评论说'已完成审查'。",
            "preset_select": "或选择预生成筛选条件",
            "preset_placeholder": "请选择...",
            "search_button": "搜索",
            "regenerate_button": "重新搜索",
            "generate_cmd_button": "生成操作指令",
            "step1_result": "第一步：搜索结果",
            "step2_desc": "第二步：描述操作",
            "select_all": "全选",
            "deselect_all": "取消全选",
            "selected_count": "已选中 **{count}** 条 issue",
            "select_prompt": "请勾选要操作的 ticket，或点击'全选/取消全选'按钮",
            "search_prompt": "请在左侧填写配置后点击 **搜索**。",
            "execute_button": "确认执行操作",
            "execute_success": "操作执行完成！",
            "operation_preview": "操作预览",
            "no_issues_selected": "没有选中的 issue，请在上表中勾选要操作的 ticket",
            "operation_info": "将对选中的 **{count}** 条 issue 执行操作",
            "download_attachments": "下载所有附件",
            "download_attachments_help": "勾选后将在执行操作时下载选中 issue 的所有附件到本地文件夹",
            "kpi_button_show": "显示 KPI 图表",
            "kpi_button_hide": "隐藏 KPI 图表",
            "kpi_title": "KPI 指标看板",
            "kpi_total": "总 Issue 数",
            "kpi_resolved": "已解决数",
            "kpi_closed": "已关闭数",
            "kpi_resolution_rate": "解决率",
            "kpi_avg_cycle": "平均周期",
            "kpi_median_cycle": "中位数周期",
            "kpi_max_cycle": "最长周期",
            "kpi_overdue": "Overdue",
        },
        "en": {
            "title": "JIRA Data Visualization Dashboard",
            "jira_config": "🔧 JIRA Configuration",
            "jira_base_url": "JIRA Base URL",
            "jira_email": "JIRA Email",
            "jira_pat": "JIRA PAT",
            "token_caption": "Token will not be persisted, please save it locally",
            "max_results": "Max Results",
            "max_results_help": "Maximum number of issues to return per search",
            "nl_filter": "Natural Language Filter & Update",
            "search_desc": "Describe search conditions in natural language",
            "search_placeholder": "e.g.: Help me search all tickets in GINFOSEC project in 2025 assigned to Peter.chen2@adidas.com",
            "operation_desc": "Describe the operation to perform on selected issues",
            "operation_placeholder": "e.g.: Assign these tickets to Peter.chen2@adidas.com, or add a comment saying 'Review completed'",
            "preset_select": "Or select a preset filter",
            "preset_placeholder": "Please select...",
            "search_button": "Search",
            "regenerate_button": "Re-search",
            "generate_cmd_button": "Generate Command",
            "step1_result": "Step 1: Search Results",
            "step2_desc": "Step 2: Describe Operation",
            "select_all": "Select All",
            "deselect_all": "Deselect All",
            "selected_count": "Selected **{count}** issues",
            "select_prompt": "Please select tickets to operate on, or click 'Select All/Deselect All' button",
            "search_prompt": "Please fill in the configuration on the left and click **Search**.",
            "execute_button": "Confirm Execute",
            "execute_success": "Operation completed!",
            "operation_preview": "Operation Preview",
            "no_issues_selected": "No issues selected, please check the tickets in the table above",
            "operation_info": "Will execute operation on **{count}** selected issues",
            "download_attachments": "Download All Attachments",
            "download_attachments_help": "Check to download all attachments of selected issues to local folder when executing operations",
            "kpi_button_show": "Show KPI Charts",
            "kpi_button_hide": "Hide KPI Charts",
            "kpi_title": "KPI Dashboard",
            "kpi_total": "Total Issues",
            "kpi_resolved": "Resolved",
            "kpi_closed": "Closed",
            "kpi_resolution_rate": "Resolution Rate",
            "kpi_avg_cycle": "Avg Cycle Time",
            "kpi_median_cycle": "Median Cycle Time",
            "kpi_max_cycle": "Max Cycle Time",
            "kpi_overdue": "Overdue",
        },
    }

    # 获取当前语言
    lang = st.session_state.get("language", "zh")
    t = TRANSLATIONS.get(lang, TRANSLATIONS["zh"])

    st.title(t["title"])

    # 初始化会话状态
    if "pending_cmd" not in st.session_state:
        st.session_state["pending_cmd"] = None
    # 两步交互流程状态
    if "step1_complete" not in st.session_state:
        st.session_state["step1_complete"] = False
    if "step1_jql" not in st.session_state:
        st.session_state["step1_jql"] = ""
    if "step1_query_steps" not in st.session_state:
        st.session_state["step1_query_steps"] = None  # None=普通查询, list=多步查询
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
    # 语言设置
    if "language" not in st.session_state:
        st.session_state["language"] = "zh"  # zh = 中文，en = English
    # KPI 图表显示状态
    if "kpi_charts_visible" not in st.session_state:
        st.session_state["kpi_charts_visible"] = False
    if "kpi_snapshot_keys" not in st.session_state:
        st.session_state["kpi_snapshot_keys"] = None  # None 表示尚未触发计算

    # 提前获取 step1_complete 状态，供侧边栏使用
    step1_complete = st.session_state.get("step1_complete", False)

    with st.sidebar:
        # 语言切换按钮（JIRA 配置上方，左对齐）
        lang_label = "English" if lang == "zh" else "中文"
        if st.button(lang_label, key="lang_toggle", use_container_width=True):
            st.session_state["language"] = "en" if lang == "zh" else "zh"
            st.rerun()

        # 使用 expander 折叠 JIRA 配置（默认收起）
        with st.expander(t["jira_config"], expanded=False):
            default_base_url = os.getenv(
                "JIRA_BASE_URL", "https://jira.tools.3stripes.net/"
            )
            default_email = os.getenv("JIRA_EMAIL", "peter.chen2@adidas.com")
            default_pat = os.getenv("JIRA_PAT", "")

            base_url = st.text_input(t["jira_base_url"], value=default_base_url)
            email = st.text_input(t["jira_email"], value=default_email)
            pat = st.text_input(
                t["jira_pat"],
                type="password",
                value=default_pat,
            )
            st.caption(t["token_caption"])

            # 最大结果数下拉菜单
            max_results_options = [100, 500, 1000, 2000, 5000]
            max_results = st.selectbox(
                t["max_results"],
                options=max_results_options,
                index=max_results_options.index(1000),
                help=t["max_results_help"]
            )

        st.divider()

        st.header(t["nl_filter"])

        # 预生成的两级筛选条件模板
        PRESET_CATEGORIES = {
            "AppSec服务概览": {
                "AppSec所有service情况": 'project = "GINFOSEC" AND created >= "2026-01-01" AND created <= "2026-12-31" AND ((reporter in ("peter.chen2@adidas.com", "hanzi.liu@externals.adidas.com", "Leon.Wang@externals.adidas.com") AND summary !~ "Application penetration test") OR (reporter = "Shervin.Aghdaei@adidas.com" AND assignee in ("Jesse.Zhang@adidas.com", "Du.Chen@adidas.com", "Kiba.Yang@adidas.com", "kiba.Yang@adidas.com", "John.Fu@adidas.com", "Zone.Tian@adidas.com", "David.Wei@adidas.com", "Spencer.Shao@adidas.com", "Laura.Yuan@adidas.com", "Jane.Lu@adidas.com", "Newman.Xu@adidas.com"))) ORDER BY created DESC',
                "AppSec所有High和Critical tickets": 'project = "GINFOSEC" AND created >= "2026-01-01" AND created <= "2026-12-31" AND (reporter = "peter.chen2@adidas.com" OR reporter = "hanzi.liu@externals.adidas.com" OR reporter = "Leon.Wang@externals.adidas.com") AND summary !~ "Application penetration test" AND priority in ("High", "Critical") ORDER BY created DESC',
            },
            "Pentest": {
                "2026 年 Pentest 项目": 'project = "GINFOSEC" AND reporter = "peter.chen2@adidas.com" AND created >= "2026-01-01" AND created <= "2026-12-31" AND summary ~ "Application penetration test"',
                "2026 年 Pentest - Critical 和 High 漏洞": 'project = "GINFOSEC" AND type = "Defect" AND labels = "ChaiTin_PenTests" AND created >= "2026-01-01" AND created <= "2026-12-31"',
            },
            "BugBounty": {
                "2026 年 BugBounty 漏洞": 'project = "GINFOSEC" AND reporter = "peter.chen2@adidas.com" AND labels = "BugBounty" AND created >= "2026-01-01" AND created <= "2026-12-31"',
            },
            "Container Security": {
                "2026 年 Container Security": 'project = "GINFOSEC" AND created >= "2026-01-01" AND created <= "2026-12-31" AND (labels = "GCA-Issues-Q1-Critical" OR labels = "ContainerSecurityL1.3")',
            },
            "SAST": {
                "SAST": 'project = "GINFOSEC" AND created >= "2026-01-01" AND created <= "2026-12-31" AND reporter = "Shervin.Aghdaei@adidas.com" AND assignee in ("Jesse.Zhang@adidas.com", "Du.Chen@adidas.com", "Kiba.Yang@adidas.com", "kiba.Yang@adidas.com", "John.Fu@adidas.com", "Zone.Tian@adidas.com", "David.Wei@adidas.com", "Spencer.Shao@adidas.com", "Laura.Yuan@adidas.com", "Jane.Lu@adidas.com", "Newman.Xu@adidas.com") ORDER BY created DESC',
            },
        }

        # 第一步：搜索条件输入框（始终显示）
        step1_description = st.text_area(
            t["search_desc"],
            placeholder=t["search_placeholder"],
            key="step1_description_input",
            height=225,
        )

        # ── 两级下拉菜单 ──────────────────────────────────────────
        # 自定义 CSS：下拉菜单文字不省略
        st.markdown("""
        <style>
        /* 已选中项显示区域：不省略 */
        div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="singleValue"],
        div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="placeholder"],
        div[data-testid="stSidebar"] div[data-baseweb="select"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: break-word !important;
        }
        /* 下拉列表选项：不省略 */
        div[data-baseweb="popover"] li span,
        div[data-baseweb="popover"] [role="option"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: break-word !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 第一级：选择类别
        _selected_cat = st.selectbox(
            t["preset_select"],
            options=[""] + list(PRESET_CATEGORIES.keys()),
            index=0,
            format_func=lambda x: "📂  请选择类别..." if x == "" else f"📁  {x}",
            key="preset_cat_selector",
        )

        # 第二级：根据第一级动态显示子项（带视觉缩进）
        if _selected_cat:
            _sub_options = list(PRESET_CATEGORIES[_selected_cat].keys())
            _selected_sub = st.selectbox(
                "sub",
                options=[""] + _sub_options,
                index=0,
                format_func=lambda x: "请选择子项..." if x == "" else x,
                key="preset_sub_selector",
                label_visibility="collapsed",
            )
            if _selected_sub:
                step1_description = PRESET_CATEGORIES[_selected_cat][_selected_sub]

        # 分割线：分隔第一步搜索和第二步操作
        if step1_complete:
            st.divider()

        # 第二步：操作描述输入框（始终显示，但"第二步：描述操作"标题在点击按钮后才显示）
        step2_description = ""
        if step1_complete:
            step2_description = st.text_area(
                t["operation_desc"],
                placeholder=t["operation_placeholder"],
                key="step2_description_input",
            )

        # 按钮逻辑：第一步完成后显示两个按钮，否则只显示搜索按钮
        if step1_complete:
            col_search, col_operate = st.columns(2)
            with col_search:
                st.button(t["regenerate_button"], key="step1_button")
            with col_operate:
                st.button(t["generate_cmd_button"], key="step2_button")

            # 分割线（在按钮下方）
            st.divider()

            # KPI 图表按钮
            kpi_button_label = "隐藏 KPI 图表" if st.session_state.get("kpi_charts_visible", False) else "显示 KPI 图表"
            if st.button(kpi_button_label, key="kpi_toggle_button", use_container_width=True):
                will_show = not st.session_state.get("kpi_charts_visible", False)
                st.session_state["kpi_charts_visible"] = will_show
                if will_show:
                    # 点击「显示」时快照当前选中的 keys，后续 checkbox 变化不影响 KPI
                    st.session_state["kpi_snapshot_keys"] = set(
                        st.session_state.get("selected_issue_keys", set())
                    )
                st.rerun()
        else:
            st.button(t["search_button"], key="step1_button")

    # 允许用户只输入域名（例如 jira.tools.3stripes.net），这里自动补全协议和末尾斜杠
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url.strip("/")
    if base_url and not base_url.endswith("/"):
        base_url += "/"

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
                t["download_attachments"],
                value=False,
                help=t["download_attachments_help"]
            )

    # 处理第一步：搜索
    if st.session_state.get("step1_button") and not step1_complete:
        if not step1_description.strip():
            st.error("请在文本框中输入搜索条件。")
            return
        try:
            cmd = interpret_nl_command(step1_description.strip(), default_project="GINFOSEC", step=1)
        except Exception as e:
            st.error(f"解析搜索条件失败：{e}")
            return

        jql = cmd.get("jql", "")
        jql = sanitize_jql(jql)
        jql = enforce_project(jql)  # 兜底：确保始终在 GINFOSEC project 下查询
        resolve_links = cmd.get("resolve_links", False)

        # 执行搜索
        query_steps = None  # 多步查询时记录步骤详情
        try:
            client = JiraClient(JiraConfig(base_url=base_url, pat=pat))
            if resolve_links:
                # 多步处理：先找到源 tickets，再找它们的 linked tickets
                with st.spinner("第一步：正在查找源 tickets..."):
                    source_issues = client.search_issues(jql=jql, max_results=max_results)

                if not source_issues:
                    st.warning("第一步未找到任何 ticket，请调整条件重试。")
                    return

                source_keys = [i["key"] for i in source_issues]

                # 收集所有 linked keys（去重）
                all_linked_keys: set = set()
                errors = []
                debug_types: dict = {}
                with st.spinner("第二步：正在获取关联 tickets..."):
                    for key in source_keys:
                        try:
                            linked, seen_types = client.get_linked_issue_keys(key, link_type_name=TESTING_LINK_TYPES)
                            all_linked_keys.update(linked)
                            debug_types[key] = seen_types
                        except Exception as e:
                            errors.append(f"{key}: {e}")

                if errors:
                    st.warning(f"部分 ticket 获取 links 失败：{'; '.join(errors)}")

                if not all_linked_keys:
                    st.warning("未找到任何通过 Testing discovered / Discovered while testing 关联的 tickets。")
                    if debug_types:
                        with st.expander("🔍 调试：源 ticket 实际包含的 link types"):
                            for k, types in debug_types.items():
                                if types:
                                    st.write(f"**{k}**：{', '.join(types)}")
                                else:
                                    st.write(f"**{k}**：（无任何 issue links）")
                    return

                # 批量查询 linked tickets
                keys_jql = "key in (" + ", ".join(f'"{k}"' for k in sorted(all_linked_keys)) + ")"
                with st.spinner(f"第三步：正在拉取 {len(all_linked_keys)} 个关联 tickets..."):
                    raw_issues = client.search_issues(jql=keys_jql, max_results=len(all_linked_keys) + 10)

                # 记录多步骤详情（用于 UI 展示）
                query_steps = [
                    {"label": "第一步：查找源 tickets", "jql": jql, "count": len(source_keys), "keys": source_keys},
                    {"label": "第二步：获取 Testing discovered / Discovered while testing 关联 keys", "keys": sorted(all_linked_keys), "count": len(all_linked_keys)},
                    {"label": "第三步：拉取关联 tickets", "jql": keys_jql, "count": len(raw_issues)},
                ]
                jql = keys_jql  # step1_jql 存最终 JQL
            else:
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
        st.session_state["step1_query_steps"] = query_steps
        st.session_state["step1_issues"] = normalize_issues(raw_issues)
        st.session_state["step1_raw_issues"] = raw_issues
        st.session_state["selected_issue_keys"] = set()  # 重置选中状态
        st.session_state["issue_selector_df"] = None  # 重置表格状态
        st.session_state["issue_selector_needs_init"] = True  # 标记需要重新初始化
        st.session_state["step2_result"] = None  # 清除之前的执行结果
        # 保存当前激活的预设条件（用于条件渲染专属图表）
        st.session_state["current_preset_cat"] = st.session_state.get("preset_cat_selector", "")
        st.session_state["current_preset_sub"] = st.session_state.get("preset_sub_selector", "")
        st.rerun()

    # 处理第二步：操作
    if st.session_state.get("step2_button") and step1_complete:
        # 从 session_state 获取输入框的值
        step2_description_val = st.session_state.get("step2_description_input", "")
        if not step2_description_val.strip():
            st.error("请在文本框中输入操作描述。")
            return

        try:
            cmd = interpret_nl_command(step2_description_val.strip(), default_project=None, step=2)
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
        st.session_state["step1_query_steps"] = None
        st.session_state["step1_issues"] = []
        st.session_state["step1_raw_issues"] = []
        st.session_state["step2_operation"] = None
        st.session_state["step2_confirmed"] = False
        st.session_state["selected_issue_keys"] = set()
        st.session_state["issue_selector_df"] = None
        st.session_state["issue_selector_needs_init"] = True
        st.session_state["kpi_charts_visible"] = False
        st.session_state["kpi_snapshot_keys"] = None
        st.session_state.pop("preset_cat_selector", None)
        st.session_state.pop("preset_sub_selector", None)
        st.rerun()

    # 显示第一步的结果
    if step1_complete:
        jql = st.session_state["step1_jql"]
        query_steps = st.session_state.get("step1_query_steps")
        df = st.session_state["step1_issues"]
        raw_issues = st.session_state["step1_raw_issues"]
        base_url = st.session_state.get("jira_base_url", base_url)  # 获取 JIRA base URL

        st.subheader(t["step1_result"])
        st.write(f"共找到 **{len(raw_issues)}** 条 issue")

        if query_steps:
            # 多步查询：展示每一步详情
            with st.expander("🔍 查询步骤详情", expanded=True):
                for i, step in enumerate(query_steps, 1):
                    st.markdown(f"**{step['label']}**（共 {step['count']} 条）")
                    if step.get("jql"):
                        st.code(step["jql"], language="sql")
                    if step.get("keys") and i < len(query_steps):
                        # 中间步骤显示 keys，最后一步不重复
                        st.caption("关联 ticket keys：" + ", ".join(step["keys"][:20]) + ("..." if len(step["keys"]) > 20 else ""))
        else:
            # 普通查询：直接显示 JQL
            st.code(jql, language="sql")

        # 构建带选择列的 DataFrame
        display_columns = [
            "summary",
            "status",
            "assignee",
            "reporter",
            "created",
            "duedate",
            "resolutiondate",
            "cycle_time_days",
            "labels",
        ]

        st.markdown(t["select_prompt"])

        # 检查是否需要重新初始化表格（新搜索结果或全选/取消操作）
        init_key = st.session_state.get("issue_selector_init_key", 0)
        needs_init = st.session_state.get("issue_selector_needs_init", True)

        if needs_init or st.session_state.get("issue_selector_df") is None:
            # 初始化表格
            display_df = df[display_columns].copy()
            current_selected = st.session_state.get("selected_issue_keys", set())
            # 从原始 df 中获取 key 列来创建 Select 列
            display_df.insert(0, "Select", df["key"].apply(lambda x: x in current_selected))
            # 保留 key 列用于内部引用（从 df 中获取）
            display_df.insert(1, "key", df["key"])
            # 创建 key_url 列存储完整 URL（用于 LinkColumn），插入到 key 列之后
            display_df.insert(2, "key_url", df["key"].apply(lambda x: f"{base_url}browse/{x}"))
            st.session_state["issue_selector_df"] = display_df
            st.session_state["issue_selector_needs_init"] = False
            st.session_state["issue_selector_init_key"] = st.session_state.get("issue_selector_init_key", 0) + 1

        # 使用 data_editor 显示带 checkbox 的表格
        # key_url 列使用 LinkColumn，显示为短的 key 格式 (GINFOSEC-xxxxx)
        edited_df = st.data_editor(
            st.session_state["issue_selector_df"],
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "选择",
                    default=False,
                    help="勾选要操作的 ticket",
                ),
                "key": None,  # 隐藏 key 列（只显示 key_url）
                "key_url": st.column_config.LinkColumn(
                    "key",
                    help="点击链接打开 JIRA issue",
                    # display_text 使用正则表达式提取 key 部分（GINFOSEC-xxxxx）
                    display_text=r"https?://[^/]+/browse/([A-Z]+-\d+)",
                    validate=r"^https?://.+",  # 确保是有效的 URL
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="issue_selector",
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
            st.success(t["selected_count"].format(count=selected_count))
        else:
            st.info(t["select_prompt"])

        # 全选/取消全选按钮 - 使用回调函数处理
        all_keys = set(df["key"].tolist())
        is_all_selected = st.session_state["selected_issue_keys"] == all_keys

        def toggle_select_all():
            if st.session_state["selected_issue_keys"] == all_keys:
                # 当前已全选，取消全选
                st.session_state["selected_issue_keys"] = set()
            else:
                # 全选
                st.session_state["selected_issue_keys"] = all_keys
            # 重置表格初始化标志
            st.session_state["issue_selector_needs_init"] = True

        # 根据状态显示不同按钮文本，但使用相同的 key
        button_text = t["deselect_all"] if is_all_selected else t["select_all"]
        if st.button(button_text, key=f"toggle_select_{st.session_state.get('toggle_counter', 0)}"):
            st.session_state["toggle_counter"] = st.session_state.get("toggle_counter", 0) + 1
            toggle_select_all()
            st.rerun()

        # 从 edited_df 中获取用户选择并保存到 session state
        new_selected = set(edited_df[edited_df["Select"]]["key"].tolist())
        old_selected = st.session_state.get("selected_issue_keys", set())
        if new_selected != old_selected:
            st.session_state["selected_issue_keys"] = new_selected

        # ========== KPI 图表区域（在表格下方，第二步上方）==========
        kpi_charts_visible = st.session_state.get("kpi_charts_visible", False)

        if kpi_charts_visible:
            st.markdown("---")
            st.subheader(t["kpi_title"])

            # 使用快照 keys（点击按钮时固定），不随 checkbox 实时变化
            snapshot_keys = st.session_state.get("kpi_snapshot_keys")
            if snapshot_keys:
                selected_df = df[df["key"].isin(snapshot_keys)].copy()
            else:
                selected_df = df.copy()

            # 提示用户当前 KPI 基于的数据范围
            st.caption(f"📊 基于点击「显示 KPI 图表」时选中的 {len(selected_df)} 条 issue（重新点击按钮可刷新）")

            # 计算 KPI 指标
            kpi_result = calculate_kpis(selected_df)

            # 显示 KPI 指标卡片（使用 st.metric）
            kpi_cols = st.columns(4)
            kpi_cols[0].metric(label=t["kpi_total"], value=kpi_result.total_count)
            kpi_cols[1].metric(label=t["kpi_resolved"], value=kpi_result.resolved_count)
            kpi_cols[2].metric(label=t["kpi_closed"], value=kpi_result.closed_count)
            kpi_cols[3].metric(label=t["kpi_resolution_rate"], value=f"{kpi_result.resolution_rate:.1f}%")

            cycle_cols = st.columns(4)
            cycle_cols[0].metric(label=t["kpi_avg_cycle"], value=f"{kpi_result.avg_cycle_days:.1f}天" if kpi_result.avg_cycle_days else "N/A")
            cycle_cols[1].metric(label=t["kpi_median_cycle"], value=f"{kpi_result.median_cycle_days:.1f}天" if kpi_result.median_cycle_days else "N/A")
            cycle_cols[2].metric(label=t["kpi_max_cycle"], value=f"{kpi_result.max_cycle_days:.1f}天" if kpi_result.max_cycle_days else "N/A")
            with cycle_cols[3]:
                _render_overdue_metric(
                    label=t["kpi_overdue"],
                    overdue_count=kpi_result.overdue_count,
                    overdue_keys=kpi_result.overdue_keys,
                    jira_base_url=st.session_state.get("jira_base_url", ""),
                )

            st.markdown("---")

            # ── AppSec 服务类型分布（仅在对应预设下显示，作为第一个图表）──
            if (
                st.session_state.get("current_preset_cat") == "AppSec服务概览"
                and st.session_state.get("current_preset_sub") == "AppSec所有service情况"
            ):
                st.subheader("AppSec 服务类型分布")
                col1, col2 = st.columns(2)
                with col1:
                    appsec_pie = create_appsec_service_pie_chart(selected_df)
                    st.plotly_chart(appsec_pie, use_container_width=True)
                with col2:
                    status_pie = create_appsec_status_chart(selected_df)
                    st.plotly_chart(status_pie, use_container_width=True)
                service_bar = create_appsec_service_bar_chart(selected_df)
                st.plotly_chart(service_bar, use_container_width=True)
                st.markdown("---")

            # ── 调用 Flask 图表服务渲染 ECharts ──────────────────────────
            import requests as _req
            import streamlit.components.v1 as _cv1

            CHART_SERVER = "http://127.0.0.1:5050"

            # 将 DataFrame 序列化为 JSON（日期转字符串）
            _records = selected_df.copy()
            for _col in ["created", "updated", "resolutiondate", "duedate", "created_date", "resolved_date"]:
                if _col in _records.columns:
                    _records[_col] = _records[_col].astype(str)
            # 用 pandas to_json 序列化（自动处理 NaN/NaT → null），再包装成 requests 可用的格式
            import json as _json
            _json_str = _records.to_json(orient="records", force_ascii=False)
            _payload_bytes = f'{{"records": {_json_str}}}'.encode("utf-8")
            try:
                _resp = _req.post(
                    f"{CHART_SERVER}/charts",
                    data=_payload_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                _resp.raise_for_status()
                _cv1.html(_resp.text, height=1000, scrolling=True)
            except Exception as _e:
                st.error(f"图表服务未启动或请求失败：{_e}\n\n请先运行：`python chart_server.py`")

            if snapshot_keys and len(snapshot_keys) < len(df):
                st.caption(f"注：以上 KPI 基于选中的 {len(snapshot_keys)} 条 issue（共 {len(df)} 条）")
            else:
                st.caption(f"注：以上 KPI 基于全部 {len(df)} 条 issue")

        # ========== KPI 图表区域结束 ==========

        # 显示第二步操作界面（当 KPI 图表显示时隐藏，且只有在生成操作指令后才显示）
        if not kpi_charts_visible and st.session_state.get("step2_operation"):
            st.markdown("---")
            st.subheader(t["step2_desc"])

            step2_operation = st.session_state.get("step2_operation")
            step2_result = st.session_state.get("step2_result")

            # 如果已经执行过操作，显示结果
            if step2_result and step2_result.get("operations_executed"):
                st.markdown("### 执行结果")
                if step2_result.get("success"):
                    st.success(t["execute_success"])
                if step2_result.get("error"):
                    st.error(step2_result["error"])

                st.info("如需执行新的操作，请重新选择操作描述或重新搜索。")

            if step2_operation:
                mode = step2_operation.get("mode", "update")
                fields_to_update = step2_operation.get("fields") or {}
                cmd = step2_operation

                # 获取用户选中的 issue
                selected_keys = st.session_state.get("selected_issue_keys", set())
                selected_raw_issues = [i for i in raw_issues if i.get("key") in selected_keys]

                if not selected_raw_issues:
                    st.warning(t["no_issues_selected"])
                else:
                    st.write(f"模式：**{mode}**")
                    st.info(t["operation_info"].format(count=len(selected_raw_issues)))

                    # 显示操作预览
                    st.markdown(t["operation_preview"])

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

                # Issue Link 预览
                if cmd.get("link_to"):
                    target_key = cmd["link_to"]
                    st.markdown("### Link 到目标 Issue (每 Issue)")
                    st.json({
                        "method": "POST",
                        "url": f"{base_url}rest/api/2/issueLink",
                        "body": {
                            "type": {"name": "Testing discovered"},
                            "inwardIssue": {"key": "<source_issue>"},
                            "outwardIssue": {"key": target_key}
                        }
                    })
                    st.info(f"选中的 issue 将使用 Link Type **Testing discovered** link 到 **{target_key}**")

                # 附件下载预览
                if download_attachments:
                    st.markdown("### 下载附件")
                    st.info(f"将下载选中 issue 的附件到当前目录下的 `jira_attachments` 文件夹")

                st.warning(f"⚠️ 注意：上述更新操作将对已选中的 {len(selected_raw_issues)} 条 issue 执行。")

                # 确认执行按钮
                if st.button(t["execute_button"]):
                    email_to_user = {}
                    client = JiraClient(JiraConfig(base_url=base_url, pat=pat))
                    success, error = execute_update_operations(client, selected_raw_issues, fields_to_update, cmd, email_to_user)

                    # 保存执行结果到 session_state
                    result = {
                        "success": success,
                        "error": error,
                        "operations_executed": True,
                    }
                    st.session_state["step2_result"] = result

                    if success:
                        st.success(t["execute_success"])
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
                    # 移除 st.rerun()，让结果保留在页面上

                # 重新搜索按钮
                if st.button(t["regenerate_button"]):
                    st.session_state["step1_complete"] = False
                    st.session_state["step1_jql"] = ""
                    st.session_state["step1_query_steps"] = None
                    st.session_state["step1_issues"] = []
                    st.session_state["step1_raw_issues"] = []
                    st.session_state["step2_operation"] = None
                    st.session_state["step2_confirmed"] = False
                    st.session_state["step2_result"] = None
                    st.session_state["selected_issue_keys"] = set()
                    st.session_state["kpi_charts_visible"] = False
                    st.session_state["kpi_snapshot_keys"] = None
                    st.rerun()
            else:
                # 等待用户输入第二步操作描述
                st.info("请在上方输入框中描述要对这些 issue 执行的操作，然后点击'执行操作'按钮。")

    else:
        # 第一步尚未完成，显示初始提示
        st.info(t["search_prompt"])
        return


if __name__ == "__main__":
    main()
