"""Тесты src/collect.py: сеть подменена, настоящих запросов нет."""

import pytest
import requests

from src import collect


class FakeResponse:
    """Фиктивный ответ сервера с кодом и JSON."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def patch_network(monkeypatch, responses):
    """Подменяет requests.get и time.sleep; возвращает списки вызовов."""
    get_calls = []
    sleep_calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        get_calls.append({"url": url, "headers": headers})
        return responses[len(get_calls) - 1]

    monkeypatch.setattr(collect.requests, "get", fake_get)
    monkeypatch.setattr(collect.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    return get_calls, sleep_calls


def test_get_json_returns_payload(monkeypatch):
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, {"a": 1})])
    result = collect.get_json("http://x", "key123")
    assert result == {"a": 1}
    assert get_calls[0]["headers"] == {"X-API-Key": "key123"}


def test_get_json_retries_after_429(monkeypatch):
    responses = [FakeResponse(429), FakeResponse(200, {"ok": True})]
    get_calls, sleep_calls = patch_network(monkeypatch, responses)
    result = collect.get_json("http://x", "key")
    assert result == {"ok": True}
    assert len(sleep_calls) == 1
    assert len(get_calls) == 2


def test_get_json_gives_up_after_max_retries(monkeypatch):
    responses = [FakeResponse(429)] * 3
    get_calls, _ = patch_network(monkeypatch, responses)
    with pytest.raises(RuntimeError):
        collect.get_json("http://x", "key", max_retries=3)
    assert len(get_calls) == 3


def test_get_json_raises_on_other_errors(monkeypatch):
    patch_network(monkeypatch, [FakeResponse(401)])
    with pytest.raises(requests.HTTPError):
        collect.get_json("http://x", "key")


def test_get_api_key_missing(monkeypatch):
    # Подмена load_dotenv, иначе тест прочитает настоящий .env
    monkeypatch.setattr(collect, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        collect.get_api_key()
