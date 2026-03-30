import dataclasses
from typing import List, Optional, Dict, Any

import requests
from requests import HTTPError


@dataclasses.dataclass
class JiraConfig:
    base_url: str
    pat: str


class JiraClient:
    def __init__(self, config: JiraConfig) -> None:
        if not config.base_url.endswith("/"):
            config.base_url += "/"
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.pat}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def get_user_account_id(self, email: str, project_key: str = None) -> str:
        """
        Get user identifier by email address.
        返回可用于设置 assignee 的用户标识符。

        对于 JIRA Server/DC: 返回 name（通常是邮箱）
        对于 JIRA Cloud: 返回 accountId
        """
        user_info = self.get_user_info(email, project_key)
        # 优先返回 name（Server/DC），其次 accountId（Cloud）
        return user_info.get("name") or user_info.get("accountId") or user_info.get("key")

    def get_user_info(self, email: str, project_key: str = None) -> dict:
        """
        Get full user info by email address.
        返回包含 name、accountId、key 等字段的完整用户信息。
        """
        import sys

        # 使用 /rest/api/2/user/search?username= (JIRA Server/DC 兼容)
        print(f"DEBUG: 调用 /rest/api/2/user/search?username={email}", file=sys.stderr)
        resp = self._session.get(
            f"{self.config.base_url}rest/api/2/user/search",
            params={"username": email},
            timeout=30,
        )
        print(f"DEBUG: 响应状态码：{resp.status_code}", file=sys.stderr)

        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"查询用户 '{email}' 失败：{resp.status_code} {resp.text}"
            ) from exc

        users = resp.json()
        if not users:
            raise RuntimeError(f"未找到邮箱为 '{email}' 的用户")

        # 精确匹配邮箱
        for user in users:
            user_email = user.get("emailAddress", "").lower()
            if user_email == email.lower():
                print(f"DEBUG: 找到用户（精确匹配）: {user}", file=sys.stderr)
                return user

        # 返回第一个结果
        first = users[0]
        print(f"DEBUG: 找到用户（模糊匹配）: {first}", file=sys.stderr)
        return first

    def get_account_ids(self, emails: List[str]) -> Dict[str, str]:
        """
        Get accountIds for multiple email addresses.
        Returns a dict mapping email -> accountId (or error message).
        """
        result = {}
        for email in emails:
            try:
                result[email] = self.get_user_account_id(email)
            except Exception as e:
                result[email] = f"ERROR: {str(e)}"
        return result

    def search_issues(
        self,
        jql: str,
        max_results: int = 1000,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch issues from JIRA using JQL. Paginates until all results (or max_results) are fetched.
        """
        if fields is None:
            fields = [
                "summary",
                "status",
                "created",
                "updated",
                "resolutiondate",
                "assignee",
                "reporter",
                "labels",
            ]

        issues: List[Dict[str, Any]] = []
        start_at = 0

        while start_at < max_results:
            remaining = max_results - start_at
            page_size = min(remaining, 100)
            # 使用 POST /rest/api/2/search，将 JQL 放在请求体中
            resp = self._session.post(
                f"{self.config.base_url}rest/api/2/search",
                json={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": page_size,
                    "fields": fields,
                },
                timeout=30,
            )
            try:
                resp.raise_for_status()
            except HTTPError as exc:
                text = resp.text
                if len(text) > 800:
                    text = text[:800] + "...(truncated)"
                raise RuntimeError(
                    f"Jira API error {resp.status_code}: {text}"
                ) from exc

            try:
                data = resp.json()
            except ValueError as exc:
                # JIRA 有时会在 200 时返回 HTML 错误页或登录页面，这里把正文前几百字抛出去方便排查
                text = resp.text
                if len(text) > 800:
                    text = text[:800] + "...(truncated)"
                raise RuntimeError(
                    f"Jira API 返回的不是 JSON（可能是 HTML 错误页）：{text}"
                ) from exc
            batch = data.get("issues", [])
            issues.extend(batch)

            if len(batch) < page_size:
                break
            start_at += page_size

        return issues

    def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> None:
        """
        Update a single issue using JIRA REST API.
        调用方需要提供符合 /rest/api/2/issue PUT schema 的 fields。
        """
        resp = self._session.put(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}",
            json={"fields": fields},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"更新 Issue {issue_key} 失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_watcher(self, issue_key: str, user_identifier: str) -> None:
        """
        Add a watcher to an issue.
        Watchers receive notifications when the issue is updated.
        user_identifier 可以是 username、邮箱或 accountId。
        """
        # 如果是邮箱格式，尝试使用 accountId
        if "@" in user_identifier:
            # JIRA 支持使用 accountId 添加 watcher
            resp = self._session.post(
                f"{self.config.base_url}rest/api/2/issue/{issue_key}/watchers",
                json={"accountId": user_identifier},
                timeout=30,
            )
        else:
            # 使用 username
            resp = self._session.post(
                f"{self.config.base_url}rest/api/2/issue/{issue_key}/watchers",
                json={"name": user_identifier},
                timeout=30,
            )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"添加 Watcher '{user_identifier}' 到 Issue {issue_key} 失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_watchers(self, issue_key: str, usernames: List[str]) -> Dict[str, Any]:
        """
        Add multiple watchers to an issue.
        Returns a dict with 'success' and 'failed' lists.
        """
        success = []
        failed = []
        for username in usernames:
            try:
                self.add_watcher(issue_key, username)
                success.append(username)
            except Exception as e:
                failed.append({"username": username, "error": str(e)})
        return {"success": success, "failed": failed}

    def add_additional_viewer(self, issue_key: str, user_identifier: str) -> None:
        """
        Add an Additional Viewer to an issue via custom field (customfield_15000).
        Additional Viewer is a multi-user picker field.
        user_identifier should be the user's name (email) or key.

        对于 JIRA Server/Data Center，使用 {'name': 'email'} 格式
        对于 JIRA Cloud，使用 {'accountId': '...'} 格式

        使用 'add' 操作在现有值基础上添加，不会覆盖已有观众。
        """
        # 使用 add 操作，在现有值基础上添加
        resp = self._session.put(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}",
            json={
                "update": {
                    "customfield_15000": [{"add": {"name": user_identifier}}]
                }
            },
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"添加 Additional Viewer '{user_identifier}' 到 Issue {issue_key} 失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_additional_viewers(
        self, issue_key: str, user_identifiers: List[str]
    ) -> Dict[str, Any]:
        """
        Add multiple Additional Viewers to an issue.
        Uses the 'add' operation to append users without overwriting existing ones.
        Returns a dict with 'success' and 'failed' lists.
        """
        success = []
        failed = []
        for user_id in user_identifiers:
            try:
                resp = self._session.put(
                    f"{self.config.base_url}rest/api/2/issue/{issue_key}",
                    json={
                        "update": {
                            "customfield_15000": [{"add": {"name": user_id}}]
                        }
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                success.append(user_id)
            except Exception as e:
                failed.append({"user_id": user_id, "error": str(e)})
        return {"success": success, "failed": failed}

    def set_additional_viewers(
        self, issue_key: str, user_identifiers: List[str]
    ) -> None:
        """
        Set Additional Viewers on an issue (replaces all existing viewers).
        user_identifiers should be a list of user names (emails) or keys.
        """
        viewers = [{"name": uid} for uid in user_identifiers]
        resp = self._session.put(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}",
            json={"fields": {"customfield_15000": viewers}},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"设置 Additional Viewers 在 Issue {issue_key} 上失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_request_participant(
        self, issue_key: str, user_identifier: str, try_fallback: bool = True
    ) -> None:
        """
        Add a request participant (Additional Viewer) to a JSM issue.
        Only available for JIRA Service Management issues.
        user_identifier can be accountId (recommended) or username.

        如果 JSM API 失败且 try_fallback=True，会尝试使用标准的 comment/assignee 更新方式
        """
        # 主 API: JSM Service Desk API
        resp = self._session.post(
            f"{self.config.base_url}rest/servicedeskapi/issue/{issue_key}/request/participants",
            json={"accountIds": [user_identifier]},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
            # 检查返回内容是否是 HTML（JIRA Server 返回 HTML 表示错误）
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                # HTML 响应表示 JSM API 不可用，触发降级
                if try_fallback:
                    self.add_additional_viewer(issue_key, user_identifier)
                    return
                raise RuntimeError(
                    f"JSM API 返回了 HTML 而不是 JSON（可能是错误页面）：{resp.text[:200]}"
                )
            # JSON 响应表示成功
            return
        except HTTPError as exc:
            # 检查是否是 JSM 特有的错误（不是 JSM issue 或没有权限）
            error_text = resp.text.lower()
            if "servicedesk" in error_text or "not found" in error_text or "404" in str(resp.status_code):
                if try_fallback:
                    # 降级：使用 Additional Viewer custom field（适用于普通 JIRA）
                    self.add_additional_viewer(issue_key, user_identifier)
                    return
            raise RuntimeError(
                f"添加 Additional Viewer '{user_identifier}' 到 Issue {issue_key} 失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_request_participants(
        self, issue_key: str, user_identifiers: List[str]
    ) -> Dict[str, Any]:
        """
        Add multiple request participants (Additional Viewers) to a JSM issue.
        Returns a dict with 'success' and 'failed' lists.
        """
        success = []
        failed = []
        for user_id in user_identifiers:
            try:
                self.add_request_participant(issue_key, user_id)
                success.append(user_id)
            except Exception as e:
                failed.append({"user_id": user_id, "error": str(e)})
        return {"success": success, "failed": failed}

    def get_available_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """
        Get available transitions for an issue.
        Returns a list of transitions, each containing 'id', 'name', 'to' (target status), etc.
        """
        resp = self._session.get(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}/transitions",
            params={"expand": "transitions.fields"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"获取 Issue {issue_key} 的可用转换失败：{resp.status_code} {resp.text}"
            ) from exc

        data = resp.json()
        return data.get("transitions", [])

    def do_transition(
        self,
        issue_key: str,
        transition_id: str,
        fields: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Execute a transition on an issue (e.g., change status).

        Args:
            issue_key: Issue key (e.g., 'PROJ-123')
            transition_id: The transition ID from get_available_transitions()
            fields: Optional fields to update during transition (e.g., resolution)
            comment: Optional comment to add during transition
        """
        payload: Dict[str, Any] = {"transition": {"id": transition_id}}

        if fields:
            payload["fields"] = fields
        if comment:
            payload["update"] = {
                "comment": [{"add": {"body": comment}}]
            }

        resp = self._session.post(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}/transitions",
            json=payload,
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"执行转换 {transition_id} 在 Issue {issue_key} 上失败：{resp.status_code} {resp.text}"
            ) from exc

    def transition_issue(
        self,
        issue_key: str,
        target_status: str,
        fields: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Transition an issue to a target status by name.
        Automatically finds the matching transition ID.

        Returns True if successful, False if no matching transition found.
        """
        transitions = self.get_available_transitions(issue_key)

        for t in transitions:
            to_status = (t.get("to") or {}).get("name", "")
            transition_name = t.get("name", "")

            # 匹配目标状态名或转换名
            if (to_status and to_status.lower() == target_status.lower()) or \
               (transition_name and transition_name.lower() == target_status.lower()):
                self.do_transition(issue_key, t["id"], fields, comment)
                return True

        available_statuses = [
            (t.get("to") or {}).get("name", "") for t in transitions
        ]
        raise RuntimeError(
            f"找不到目标状态 '{target_status}'。Issue {issue_key} 的可用状态：{available_statuses}"
        )

    def add_comment(self, issue_key: str, body: str) -> None:
        """
        Add a comment to an issue using /rest/api/2/issue/{key}/comment API.
        """
        resp = self._session.post(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}/comment",
            json={"body": body},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"添加评论到 Issue {issue_key} 失败：{resp.status_code} {resp.text}"
            ) from exc

    def add_comments(self, issue_key: str, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple comments to an issue.
        comments 格式：[{"body": "评论 1"}, {"body": "评论 2"}]
        Returns a dict with 'success' and 'failed' lists.
        """
        success = []
        failed = []
        for c in comments:
            try:
                self.add_comment(issue_key, c["body"])
                success.append(c)
            except Exception as e:
                failed.append({"comment": c, "error": str(e)})
        return {"success": success, "failed": failed}

    def get_issue_attachments(self, issue_key: str) -> List[Dict[str, Any]]:
        """
        Get all attachments for an issue.
        Returns a list of attachment objects with 'id', 'filename', 'size', 'mimeType', 'content' (URL), etc.
        """
        resp = self._session.get(
            f"{self.config.base_url}rest/api/2/issue/{issue_key}",
            params={"fields": "attachment"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"获取 Issue {issue_key} 的附件失败：{resp.status_code} {resp.text}"
            ) from exc

        data = resp.json()
        fields = data.get("fields", {})
        return fields.get("attachment", [])

    def download_attachment(
        self, attachment_url: str, save_path: str
    ) -> str:
        """
        Download an attachment from JIRA and save it to the specified path.
        Returns the save path.

        Args:
            attachment_url: The 'content' URL from the attachment object
            save_path: Local file path to save the attachment
        """
        resp = self._session.get(attachment_url, timeout=60)
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"下载附件 {attachment_url} 失败：{resp.status_code} {resp.text}"
            ) from exc

        # Ensure parent directory exists
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(resp.content)

        return save_path

    def download_issue_attachments(
        self,
        issue_key: str,
        output_dir: str,
        flatten: bool = True,
    ) -> Dict[str, Any]:
        """
        Download all attachments from an issue.

        Args:
            issue_key: Issue key (e.g., 'PROJ-123')
            output_dir: Directory to save attachments
            flatten: If True, save all files in output_dir directly.
                     If False, create subfolder per issue.

        Returns:
            Dict with 'success' and 'failed' lists
        """
        import os

        attachments = self.get_issue_attachments(issue_key)
        success = []
        failed = []

        if not attachments:
            return {"success": [], "failed": [], "message": "No attachments found"}

        # Create output directory
        if not flatten:
            issue_dir = os.path.join(output_dir, issue_key.replace(":", "_"))
            os.makedirs(issue_dir, exist_ok=True)
        else:
            issue_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        for att in attachments:
            filename = att.get("filename", "unknown")
            content_url = att.get("content", "")

            if not content_url:
                failed.append({
                    "key": issue_key,
                    "filename": filename,
                    "error": "No content URL",
                })
                continue

            # Build save path
            save_path = os.path.join(issue_dir, f"{issue_key}_{filename}")

            try:
                self.download_attachment(content_url, save_path)
                success.append({
                    "key": issue_key,
                    "filename": filename,
                    "size": att.get("size", 0),
                    "path": save_path,
                })
            except Exception as e:
                failed.append({
                    "key": issue_key,
                    "filename": filename,
                    "error": str(e),
                })

        return {"success": success, "failed": failed}
