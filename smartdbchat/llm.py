import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from smartdbchat.config import LLMConfig
from smartdbchat.models import StreamCallback


def clean_sql_response(response_text: str) -> str:
    """清理大模型返回的 SQL 文本，去除代码块标记与多余空行。"""
    patterns = [
        r"```sql\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"`(.*?)`",
    ]

    cleaned_text = response_text.strip()
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if matches:
            cleaned_text = matches[0]
            break

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    return "\n".join(lines)


class LLMClient:
    """OpenAI 兼容大语言模型客户端封装。"""

    def __init__(self, config: LLMConfig, client: Optional[OpenAI] = None):
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key or "ollama",
            base_url=config.base_url,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """同步调用对话补全接口，返回原始文本响应。"""
        kwargs: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def generate_sql(self, messages: List[Dict[str, str]]) -> str:
        """同步生成并清洗 SQL 语句。"""
        raw_sql = self.chat_completion(messages)
        return clean_sql_response(raw_sql)

    def stream_analysis(
        self,
        messages: List[Dict[str, str]],
        on_chunk: Optional[StreamCallback] = None,
    ) -> str:
        """流式生成自然语言分析与图表配置，返回完整文本。"""
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
        )
        response_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta is not None:
                response_text += delta
                if on_chunk is not None:
                    on_chunk(response_text)
        return response_text
