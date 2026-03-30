from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import requests


@dataclass
class QwenConfig:
    """
    配置 Qwen / DashScope 调用参数。
    """

    api_key: str
    model: str = "qwen-plus"
    # 兼容模式 chat/completions 端点；如需使用国际站可改为 dashscope-intl 域名
    base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


class QwenClient:
    """
    使用阿里云 DashScope 的 Qwen 模型，OpenAI 兼容模式：
    POST /compatible-mode/v1/chat/completions
    """

    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat_raw(
        self, messages: List[Dict[str, str]], temperature: float = 0
    ) -> Dict[str, Any]:
        resp = self._session.post(
            self.config.base_url,
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        data = self.chat_raw(messages=messages, temperature=temperature)
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Qwen 返回结果解析失败：{data}") from exc
        return str(content).strip()

