from types import SimpleNamespace
from unittest.mock import MagicMock

from smartdbchat.config import LLMConfig
from smartdbchat.llm import LLMClient, clean_sql_response


def test_clean_sql_response_strips_markdown_fences():
    raw = """```sql
    SELECT name, avg_score
    FROM students;
    ```"""
    assert clean_sql_response(raw) == "SELECT name, avg_score\nFROM students;"


def test_clean_sql_response_handles_plain_and_inline():
    assert clean_sql_response("`SELECT 1`") == "SELECT 1"
    assert clean_sql_response("  SELECT 1\n\n  FROM dual  ") == "SELECT 1\nFROM dual"


_LLM_ENV_VARS = (
    "SMARTDB_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
    "SMARTDB_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL",
    "SMARTDB_MODEL", "LLM_MODEL",
)


def test_llm_config_from_env_defaults(monkeypatch):
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    cfg = LLMConfig.from_env()
    assert cfg == LLMConfig()


def test_llm_config_from_env_priority(monkeypatch):
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("SMARTDB_MODEL", "qwen-plus")
    monkeypatch.setenv("LLM_MODEL", "ignored")
    cfg = LLMConfig.from_env()
    assert cfg.api_key == "sk-llm"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model == "qwen-plus"


def test_llm_client_chat_completion_and_generate_sql():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  ```sql\nSELECT 1;\n```  "))]
    )
    client = LLMClient(LLMConfig(model="gpt-test", api_key="test"), client=mock_openai)

    raw = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="override-model",
        temperature=0.2,
    )
    assert raw == "```sql\nSELECT 1;\n```"
    mock_openai.chat.completions.create.assert_called_once_with(
        model="override-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
    )

    mock_openai.chat.completions.create.reset_mock()
    sql = client.generate_sql([{"role": "user", "content": "select 1"}])
    assert sql == "SELECT 1;"
    mock_openai.chat.completions.create.assert_called_once_with(
        model="gpt-test",
        messages=[{"role": "user", "content": "select 1"}],
    )
