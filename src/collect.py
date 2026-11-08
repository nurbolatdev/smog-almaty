"""Сбор данных из API OpenAQ v3: метаданные станции и запросы с повтором."""

import os
import time

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 8876


def get_api_key() -> str:
    """Читает ключ OpenAQ из .env; если его нет — понятная ошибка."""
    load_dotenv()
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY is not set, see .env.example")
    return api_key


def get_json(
    url: str,
    api_key: str,
    params: dict | None = None,
    max_retries: int = 3,
    wait_seconds: int = 60,
) -> dict:
    """Делает GET-запрос к OpenAQ; при коде 429 ждёт и повторяет."""
    for _attempt in range(max_retries):
        response = requests.get(
            url, headers={"X-API-Key": api_key}, params=params, timeout=30
        )
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
    # Единицы проверяем один раз: анализ дальше ведётся в µg/m³
    for sensor in location["sensors"]:
        if sensor["id"] == sensor_id:
            assert sensor["parameter"]["units"] == "µg/m³", "PM2.5 не в µg/m³"
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
