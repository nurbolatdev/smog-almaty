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
