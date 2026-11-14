"""Тесты src/collect.py: сеть подменена, настоящих запросов нет."""

import pandas as pd
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
        get_calls.append({"url": url, "headers": headers, "params": params})
        return responses[len(get_calls) - 1]

    monkeypatch.setattr(collect.requests, "get", fake_get)
    monkeypatch.setattr(collect.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    return get_calls, sleep_calls


def test_get_json_returns_payload(monkeypatch):
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, {"a": 1})])
    result = collect.get_json("http://x", "key123")
    assert result == {"a": 1}
    assert get_calls[0]["headers"] == {"X-API-Key": "key123"}


def test_get_json_without_key_sends_no_key_header(monkeypatch):
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, {"a": 1})])
    result = collect.get_json("http://x")
    assert result == {"a": 1}
    assert get_calls[0]["headers"] == {}


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


def test_parse_location_rejects_wrong_units():
    data = make_location_response([make_sensor(2, "pm25", units="ppm")])
    with pytest.raises(AssertionError):
        collect.parse_location(data)


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


def make_sensor_response():
    """Фиктивный ответ /sensors/2 с периодом данных."""
    sensor = {
        "datetimeFirst": {"utc": "2020-04-09T17:00:00Z"},
        "datetimeLast": {"utc": "2025-11-14T11:00:00Z"},
    }
    return {"results": [sensor]}


def patch_station_requests(monkeypatch, location_response):
    """Подменяет ключ и requests.get: ответ выбирается по URL; возвращает список URL."""
    urls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        urls.append(url)
        if "/locations/" in url:
            return FakeResponse(200, location_response)
        return FakeResponse(200, make_sensor_response())

    monkeypatch.setattr(collect, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(collect.requests, "get", fake_get)
    return urls


def test_get_station_info_combines_both_requests(monkeypatch):
    location_response = make_location_response([make_sensor(1, "pm10"), make_sensor(2, "pm25")])
    urls = patch_station_requests(monkeypatch, location_response)
    result = collect.get_station_info()
    assert result == {
        "location_id": 8876,
        "name": "Almaty",
        "latitude": 43.25,
        "longitude": 76.93,
        "sensor_id": 2,
        "datetime_first_utc": "2020-04-09T17:00:00Z",
        "datetime_last_utc": "2025-11-14T11:00:00Z",
    }
    assert urls[1].endswith("/sensors/2")


def test_get_station_info_rejects_coordinates_outside_almaty(monkeypatch):
    location_response = make_location_response([make_sensor(2, "pm25")])
    # Координаты Астаны
    location_response["results"][0]["coordinates"] = {"latitude": 51.17, "longitude": 71.43}
    patch_station_requests(monkeypatch, location_response)
    with pytest.raises(AssertionError):
        collect.get_station_info()


def test_check_station_info_rejects_non_integer_sensor_id():
    info = {
        "latitude": 43.25,
        "longitude": 76.93,
        "sensor_id": "2",
        "datetime_first_utc": "2020-04-09T17:00:00Z",
        "datetime_last_utc": "2025-11-14T11:00:00Z",
    }
    with pytest.raises(AssertionError):
        collect.check_station_info(info)


def test_make_year_windows_are_adjacent():
    windows = collect.make_year_windows("2020-04-09T16:00:00Z", "2022-03-01T11:00:00Z")
    assert windows == [
        ("2020-04-09T16:00:00Z", "2021-01-01T00:00:00Z"),
        ("2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        ("2022-01-01T00:00:00Z", "2022-03-01T11:00:00Z"),
    ]


def test_make_year_windows_inside_one_year():
    windows = collect.make_year_windows("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z")
    assert windows == [("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z")]


def test_fetch_sensor_measurements_stops_on_empty_page(monkeypatch):
    responses = [
        FakeResponse(200, {"results": [{"value": 1}, {"value": 2}]}),
        FakeResponse(200, {"results": [{"value": 3}]}),
        FakeResponse(200, {"results": []}),
    ]
    get_calls, sleep_calls = patch_network(monkeypatch, responses)
    results = collect.fetch_sensor_measurements(
        25903, "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "key"
    )
    assert len(results) == 3
    assert len(get_calls) == 3
    assert [call["params"]["page"] for call in get_calls] == [1, 2, 3]
    assert len(sleep_calls) == 2


def test_fetch_sensor_measurements_sends_period_and_limit(monkeypatch):
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, {"results": []})])
    collect.fetch_sensor_measurements(25903, "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "key")
    assert get_calls[0]["url"].endswith("/sensors/25903/measurements")
    params = get_calls[0]["params"]
    assert params["datetime_from"] == "2025-01-01T00:00:00Z"
    assert params["datetime_to"] == "2025-02-01T00:00:00Z"
    assert params["limit"] == 1000
    assert params["page"] == 1


def test_fetch_sensor_measurements_raises_when_pages_do_not_end(monkeypatch):
    responses = [FakeResponse(200, {"results": [{"value": 1}]})] * 3
    get_calls, _ = patch_network(monkeypatch, responses)
    with pytest.raises(RuntimeError):
        collect.fetch_sensor_measurements(
            25903, "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "key", max_pages=3
        )
    assert len(get_calls) == 3


def test_fetch_period_measurements_requests_each_year(monkeypatch):
    # Окно 2024: одна запись и пустая страница; окно 2025: сразу пустая страница
    responses = [
        FakeResponse(200, {"results": [{"value": 1}]}),
        FakeResponse(200, {"results": []}),
        FakeResponse(200, {"results": []}),
    ]
    get_calls, _ = patch_network(monkeypatch, responses)
    results = collect.fetch_period_measurements(
        25903, "2024-06-01T00:00:00Z", "2025-03-01T00:00:00Z", "key"
    )
    assert results == [{"value": 1}]
    assert get_calls[0]["params"]["datetime_to"] == "2025-01-01T00:00:00Z"
    assert get_calls[2]["params"]["datetime_from"] == "2025-01-01T00:00:00Z"


def make_measurement(datetime_from, datetime_to, value, name="pm25", units="µg/m³"):
    """Запись измерения в форме ответа API."""
    return {
        "value": value,
        "flagInfo": {"hasFlags": False},
        "parameter": {"name": name, "units": units},
        "period": {
            "datetimeFrom": {"utc": datetime_from},
            "datetimeTo": {"utc": datetime_to},
        },
    }


def make_table(times, values=None):
    """Таблица в формате сырого файла: метка конца часа и значения."""
    if values is None:
        values = [10.0] * len(times)
    return pd.DataFrame(
        {
            "datetime_from_utc": times,
            "datetime_to_utc": times,
            "pm25": values,
            "has_flags": [False] * len(times),
        }
    )


def test_measurements_to_dataframe_keeps_raw_columns():
    results = [
        make_measurement("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10.0),
        make_measurement("2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", 12.5),
    ]
    df = collect.measurements_to_dataframe(results)
    assert list(df.columns) == ["datetime_from_utc", "datetime_to_utc", "pm25", "has_flags"]
    assert list(df["pm25"]) == [10.0, 12.5]
    assert df["datetime_to_utc"][1] == "2024-01-01T02:00:00Z"


def test_measurements_to_dataframe_rejects_other_parameter():
    results = [make_measurement("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10.0, name="pm10")]
    with pytest.raises(AssertionError):
        collect.measurements_to_dataframe(results)


def test_measurements_to_dataframe_rejects_wrong_units():
    results = [make_measurement("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10.0, units="ppm")]
    with pytest.raises(AssertionError):
        collect.measurements_to_dataframe(results)


def test_check_pm25_export_accepts_valid_table():
    df = make_table(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"])
    collect.check_pm25_export(df, "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def test_check_pm25_export_rejects_time_outside_period():
    df = make_table(["2024-01-01T01:00:00Z", "2024-01-01T05:00:00Z"])
    with pytest.raises(AssertionError):
        collect.check_pm25_export(df, "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def test_check_pm25_export_rejects_bad_datetime():
    df = make_table(["2024-01-01T01:00:00Z", "not a date"])
    with pytest.raises(AssertionError):
        collect.check_pm25_export(df, "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def test_check_pm25_export_rejects_more_rows_than_hours():
    # Период 2 часа (from, to] вмещает 2 строки, пришло 3
    times = ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T02:00:00Z"]
    df = make_table(times)
    with pytest.raises(AssertionError):
        collect.check_pm25_export(df, "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def test_download_pm25_combines_steps(monkeypatch):
    monkeypatch.setattr(collect, "get_api_key", lambda: "test-key")
    records = [
        make_measurement("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10.0),
        make_measurement("2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", 12.5),
    ]
    responses = [
        FakeResponse(200, {"results": records}),
        FakeResponse(200, {"results": []}),
    ]
    patch_network(monkeypatch, responses)
    df = collect.download_pm25(25903, "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")
    assert df.shape == (2, 4)
    assert list(df.columns) == ["datetime_from_utc", "datetime_to_utc", "pm25", "has_flags"]


def make_day_times(day="2024-01-01"):
    """24 часовые метки одного дня в формате Open-Meteo."""
    times = pd.date_range(day, periods=24, freq="h")
    return times.strftime(collect.METEO_TIME_FORMAT).tolist()


def make_meteo_response(
    hourly, hourly_units, utc_offset_seconds=0, latitude=43.25, longitude=76.93
):
    """Ответ Open-Meteo в форме реального."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "utc_offset_seconds": utc_offset_seconds,
        "hourly_units": hourly_units,
        "hourly": hourly,
    }


def make_weather_response(**kwargs):
    """Ответ с погодой за один день: 6 переменных, верные единицы."""
    hourly = {"time": make_day_times()}
    for variable in collect.WEATHER_UNITS:
        hourly[variable] = [1.0] * 24
    units = dict(collect.WEATHER_UNITS)
    units.update(kwargs.pop("units", {}))
    return make_meteo_response(hourly, units, **kwargs)


def make_cams_response(values, **kwargs):
    """Ответ CAMS: время и pm2_5."""
    hourly = {"time": make_day_times()[: len(values)], "pm2_5": values}
    return make_meteo_response(hourly, dict(collect.CAMS_UNITS), **kwargs)


def make_hourly_table(times):
    """Таблица для check_hourly_export: метки и одна числовая колонка."""
    return pd.DataFrame({"time_utc": times, "cams_pm25": [1.0] * len(times)})


def test_hourly_to_dataframe_renames_and_keeps_values():
    values = [10.5, None, 12.0]
    data = make_cams_response(values)
    df = collect.hourly_to_dataframe(data, collect.CAMS_UNITS, collect.CAMS_COLUMNS)
    assert list(df.columns) == ["time_utc", "cams_pm25"]
    assert df["cams_pm25"].iloc[0] == 10.5
    assert pd.isna(df["cams_pm25"].iloc[1])
    assert df["cams_pm25"].iloc[2] == 12.0


def test_hourly_to_dataframe_rejects_wrong_units():
    data = make_weather_response(units={"wind_speed_10m": "km/h"})
    with pytest.raises(AssertionError):
        collect.hourly_to_dataframe(data, collect.WEATHER_UNITS, collect.WEATHER_COLUMNS)


def test_hourly_to_dataframe_rejects_local_time():
    data = make_cams_response([1.0], utc_offset_seconds=18000)
    with pytest.raises(AssertionError):
        collect.hourly_to_dataframe(data, collect.CAMS_UNITS, collect.CAMS_COLUMNS)


def test_hourly_to_dataframe_rejects_far_coordinates():
    data = make_cams_response([1.0], latitude=45.0)
    with pytest.raises(AssertionError):
        collect.hourly_to_dataframe(data, collect.CAMS_UNITS, collect.CAMS_COLUMNS)


def test_hourly_to_dataframe_rejects_far_longitude():
    data = make_cams_response([1.0], longitude=79.0)
    with pytest.raises(AssertionError):
        collect.hourly_to_dataframe(data, collect.CAMS_UNITS, collect.CAMS_COLUMNS)


def test_check_hourly_export_accepts_full_day():
    df = make_hourly_table(make_day_times())
    collect.check_hourly_export(df, "2024-01-01", "2024-01-01")


def test_check_hourly_export_rejects_bad_time():
    times = make_day_times()
    times[5] = "not a date"
    with pytest.raises(AssertionError):
        collect.check_hourly_export(make_hourly_table(times), "2024-01-01", "2024-01-01")


def test_check_hourly_export_rejects_missing_hour():
    times = make_day_times()
    del times[5]
    with pytest.raises(AssertionError):
        collect.check_hourly_export(make_hourly_table(times), "2024-01-01", "2024-01-01")


def test_check_hourly_export_rejects_duplicate_hour():
    times = make_day_times()
    times[5] = times[4]
    with pytest.raises(AssertionError):
        collect.check_hourly_export(make_hourly_table(times), "2024-01-01", "2024-01-01")


def fail_if_key_requested():
    """Подмена get_api_key: ключ Open-Meteo не нужен, вызов — ошибка."""
    raise AssertionError("ключ OpenAQ не должен запрашиваться")


def test_download_weather_sends_coordinates_and_units(monkeypatch):
    monkeypatch.setattr(collect, "get_api_key", fail_if_key_requested)
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, make_weather_response())])
    df = collect.download_weather("2024-01-01", "2024-01-01")
    call = get_calls[0]
    assert call["url"] == collect.WEATHER_URL
    assert call["headers"] == {}
    assert call["params"]["latitude"] == collect.LATITUDE
    assert call["params"]["longitude"] == collect.LONGITUDE
    assert call["params"]["timezone"] == "UTC"
    assert call["params"]["wind_speed_unit"] == "ms"
    for variable in collect.WEATHER_UNITS:
        assert variable in call["params"]["hourly"]
    assert df.shape == (24, 7)
    assert list(df.columns) == list(collect.WEATHER_COLUMNS.values())


def test_download_cams_sends_domain_and_pm25(monkeypatch):
    monkeypatch.setattr(collect, "get_api_key", fail_if_key_requested)
    response = make_cams_response([5.0] * 24)
    get_calls, _ = patch_network(monkeypatch, [FakeResponse(200, response)])
    df = collect.download_cams("2024-01-01", "2024-01-01")
    call = get_calls[0]
    assert call["url"] == collect.AIR_QUALITY_URL
    assert call["headers"] == {}
    assert call["params"]["hourly"] == "pm2_5"
    assert call["params"]["domains"] == "cams_global"
    assert call["params"]["latitude"] == collect.LATITUDE
    assert call["params"]["longitude"] == collect.LONGITUDE
    assert call["params"]["timezone"] == "UTC"
    assert df.shape == (24, 2)
    assert list(df.columns) == ["time_utc", "cams_pm25"]


def test_save_raw_csv_writes_file(tmp_path):
    df = make_table(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], [10.0, 12.5])
    path = str(tmp_path / "raw.csv")
    collect.save_raw_csv(df, path)
    saved = pd.read_csv(path)
    assert list(saved.columns) == list(df.columns)
    assert list(saved["pm25"]) == [10.0, 12.5]


def test_save_raw_csv_refuses_to_overwrite(tmp_path):
    path = str(tmp_path / "raw.csv")
    collect.save_raw_csv(make_table(["2024-01-01T01:00:00Z"], [10.0]), path)
    with open(path) as file:
        before = file.read()
    with pytest.raises(FileExistsError):
        collect.save_raw_csv(make_table(["2024-01-01T02:00:00Z"], [99.0]), path)
    with open(path) as file:
        assert file.read() == before
