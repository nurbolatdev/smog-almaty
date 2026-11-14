"""Сбор данных из API OpenAQ v3 и Open-Meteo: метаданные станции, PM2.5, погода и CAMS."""

import os
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 8876

# Сенсор и период из задачи 1.1 (python -m src.collect, 2026-11-08).
# datetime_from на час раньше первой метки: окно API (from, to], первая строка to=17:00
SENSOR_ID = 25903
PERIOD_FROM_UTC = "2020-04-09T16:00:00Z"
PERIOD_TO_UTC = "2025-11-14T11:00:00Z"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
PAGE_LIMIT = 1000
# Лимит OpenAQ — 60 запросов в минуту: пауза 1 с после каждой непустой страницы;
# всего 47 запросов, это меньше лимита 60 в минуту
PAUSE_SECONDS = 1
# Колонки сырого файла: путь в JSON -> имя колонки (значения не меняем)
RAW_COLUMNS = {
    "period.datetimeFrom.utc": "datetime_from_utc",
    "period.datetimeTo.utc": "datetime_to_utc",
    "value": "pm25",
    "flagInfo.hasFlags": "has_flags",
}


def get_api_key() -> str:
    """Читает ключ OpenAQ из .env; если его нет — понятная ошибка."""
    load_dotenv()
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY is not set, see .env.example")
    return api_key


def get_json(
    url: str,
    api_key: str | None = None,
    params: dict | None = None,
    max_retries: int = 3,
    wait_seconds: int = 60,
) -> dict:
    """GET-запрос к API (OpenAQ — с ключом, Open-Meteo — без); при 429 ждёт и повторяет."""
    headers = {}
    if api_key is not None:
        # Ключ нужен только OpenAQ; Open-Meteo работает без ключа
        headers["X-API-Key"] = api_key
    for _attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            # Слишком много запросов: ждём, пока сбросится минутное окно
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Too many requests (429): gave up after {max_retries} attempts")


def find_pm25_sensor_id(sensors: list[dict]) -> int:
    """Возвращает id единственного сенсора PM2.5 из списка сенсоров локации."""
    pm25_sensors = [s for s in sensors if s["parameter"]["name"] == "pm25"]
    if len(pm25_sensors) != 1:
        raise ValueError(f"Expected exactly 1 pm25 sensor, found {len(pm25_sensors)}")
    return pm25_sensors[0]["id"]


def parse_location(data: dict) -> dict:
    """Достаёт из ответа /locations имя, координаты и id сенсора PM2.5."""
    location = data["results"][0]
    sensor_id = find_pm25_sensor_id(location["sensors"])
    # Находим сенсор PM2.5 и один раз проверяем единицы: анализ ведётся в µg/m³
    pm25_sensor = [s for s in location["sensors"] if s["id"] == sensor_id][0]
    assert pm25_sensor["parameter"]["units"] == "µg/m³", "PM2.5 не в µg/m³"
    return {
        "location_id": location["id"],
        "name": location["name"],
        "latitude": location["coordinates"]["latitude"],
        "longitude": location["coordinates"]["longitude"],
        "sensor_id": sensor_id,
    }


def parse_sensor_period(data: dict) -> dict:
    """Достаёт из ответа /sensors первую и последнюю метки времени (UTC)."""
    sensor = data["results"][0]
    return {
        "datetime_first_utc": sensor["datetimeFirst"]["utc"],
        "datetime_last_utc": sensor["datetimeLast"]["utc"],
    }


def check_station_info(info: dict) -> None:
    """Проверяет, что станция в Алматы, сенсор задан и период не перепутан."""
    assert 43.0 < info["latitude"] < 43.5, "станция не в Алматы: широта вне рамки"
    assert 76.6 < info["longitude"] < 77.2, "станция не в Алматы: долгота вне рамки"
    assert isinstance(info["sensor_id"], int), "sensor_id должен быть целым числом"
    assert info["sensor_id"] > 0, "sensor_id должен быть положительным"
    first = datetime.fromisoformat(info["datetime_first_utc"])
    last = datetime.fromisoformat(info["datetime_last_utc"])
    assert first < last, "начало периода должно быть раньше конца"


def get_station_info(location_id: int = LOCATION_ID) -> dict:
    """Возвращает координаты станции, id сенсора PM2.5 и период его данных (UTC)."""
    api_key = get_api_key()
    # Запрос 1: локация -> координаты и id сенсора PM2.5
    location_data = get_json(f"{BASE_URL}/locations/{location_id}", api_key)
    info = parse_location(location_data)
    # Запрос 2: сенсор -> период его данных
    sensor_data = get_json(f"{BASE_URL}/sensors/{info['sensor_id']}", api_key)
    info.update(parse_sensor_period(sensor_data))
    check_station_info(info)
    return info


def make_year_windows(datetime_from: str, datetime_to: str) -> list[tuple[str, str]]:
    """Режет период на окна по календарным годам; конец одного окна = начало следующего."""
    start = datetime.strptime(datetime_from, TIME_FORMAT)
    end = datetime.strptime(datetime_to, TIME_FORMAT)
    windows = []
    window_from = start
    while window_from < end:
        new_year = datetime(window_from.year + 1, 1, 1)
        window_to = min(new_year, end)
        windows.append((window_from.strftime(TIME_FORMAT), window_to.strftime(TIME_FORMAT)))
        window_from = window_to
    return windows


def fetch_sensor_measurements(
    sensor_id: int,
    datetime_from: str,
    datetime_to: str,
    api_key: str,
    max_pages: int = 100,
) -> list[dict]:
    """Скачивает все страницы измерений сенсора за одно окно, пока не придёт пустая."""
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
    results = []
    for page in range(1, max_pages + 1):
        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": PAGE_LIMIT,
            "page": page,
        }
        data = get_json(url, api_key, params=params)
        if len(data["results"]) == 0:
            return results
        results.extend(data["results"])
        print(f"page {page}: {len(data['results'])} rows")
        # Пауза после каждой непустой страницы из-за лимита запросов
        time.sleep(PAUSE_SECONDS)
    raise RuntimeError(f"No empty page after {max_pages} pages: pagination did not end")


def fetch_period_measurements(
    sensor_id: int, datetime_from: str, datetime_to: str, api_key: str
) -> list[dict]:
    """Скачивает измерения за весь период по годовым окнам (глубокие страницы дают 408)."""
    results = []
    for window_from, window_to in make_year_windows(datetime_from, datetime_to):
        print(f"window: {window_from} .. {window_to}")
        results.extend(fetch_sensor_measurements(sensor_id, window_from, window_to, api_key))
    return results


def measurements_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Превращает список записей API в таблицу с нужными колонками, значения не меняет."""
    assert len(results) > 0, "API не вернул ни одной записи"
    df = pd.json_normalize(results)
    assert (df["parameter.name"] == "pm25").all(), "в данных не только pm25"
    assert (df["parameter.units"] == "µg/m³").all(), "PM2.5 не в µg/m³"
    return df[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)


def check_pm25_export(df: pd.DataFrame, datetime_from: str, datetime_to: str) -> None:
    """Проверяет выгрузку: колонки, даты, период (from, to] по datetime_to_utc, число строк."""
    assert len(df) > 0, "таблица пустая"
    assert list(df.columns) == list(RAW_COLUMNS.values()), "колонки не те, что ожидали"
    times = pd.to_datetime(df["datetime_to_utc"], utc=True, format=TIME_FORMAT, errors="coerce")
    assert times.isna().sum() == 0, "есть даты, которые не удалось разобрать"
    start = pd.Timestamp(datetime_from)
    end = pd.Timestamp(datetime_to)
    # Окно API (from, to]: начало не включается, конец включается
    assert times.min() > start, "есть метки не позже начала периода"
    assert times.max() <= end, "есть метки позже конца периода"
    assert pd.api.types.is_numeric_dtype(df["pm25"]), "pm25 не число"
    max_rows = (end - start) / pd.Timedelta(hours=1)
    assert len(df) <= max_rows, "строк больше, чем часов: повтор страниц"


def download_pm25(
    sensor_id: int = SENSOR_ID,
    datetime_from: str = PERIOD_FROM_UTC,
    datetime_to: str = PERIOD_TO_UTC,
) -> pd.DataFrame:
    """Выгружает PM2.5 за период по годам и возвращает проверенную таблицу, на диск не пишет."""
    api_key = get_api_key()
    results = fetch_period_measurements(sensor_id, datetime_from, datetime_to, api_key)
    df = measurements_to_dataframe(results)
    check_pm25_export(df, datetime_from, datetime_to)
    return df


def save_raw_csv(df: pd.DataFrame, path: str) -> None:
    """Сохраняет сырую таблицу в CSV; существующий файл не перезаписывает."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists: raw data is never overwritten")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    station_info = get_station_info()
    for key, value in station_info.items():
        print(f"{key}: {value}")
