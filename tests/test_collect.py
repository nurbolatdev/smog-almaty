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


def make_sensor(sensor_id, name, units="µg/m³"):
    """Фиктивный сенсор в той же форме, что в ответе OpenAQ."""
    return {"id": sensor_id, "parameter": {"name": name, "units": units}}


def make_location_response(sensors):
    """Фиктивный ответ /locations/8876 с урезанными полями."""
    location = {
        "id": 8876,
        "name": "Almaty",
        "coordinates": {"latitude": 43.25, "longitude": 76.93},
        "sensors": sensors,
    }
    return {"results": [location]}


def test_find_pm25_sensor_id_picks_pm25():
    sensors = [make_sensor(1, "pm10"), make_sensor(2, "pm25"), make_sensor(3, "temperature", "c")]
    assert collect.find_pm25_sensor_id(sensors) == 2


def test_find_pm25_sensor_id_none_found():
    with pytest.raises(ValueError):
        collect.find_pm25_sensor_id([make_sensor(1, "pm10")])


def test_find_pm25_sensor_id_two_found():
    with pytest.raises(ValueError):
        collect.find_pm25_sensor_id([make_sensor(1, "pm25"), make_sensor(2, "pm25")])


def test_parse_location():
    data = make_location_response([make_sensor(1, "pm10"), make_sensor(2, "pm25")])
    result = collect.parse_location(data)
    assert result == {
        "location_id": 8876,
        "name": "Almaty",
        "latitude": 43.25,
        "longitude": 76.93,
        "sensor_id": 2,
    }


def test_parse_sensor_period():
    data = {
        "results": [
            {
                "datetimeFirst": {"utc": "2020-04-09T17:00:00Z", "local": "x"},
                "datetimeLast": {"utc": "2025-11-14T11:00:00Z", "local": "y"},
            }
        ]
    }
    result = collect.parse_sensor_period(data)
    assert result["datetime_first_utc"] == "2020-04-09T17:00:00Z"
    assert result["datetime_last_utc"] == "2025-11-14T11:00:00Z"
