# Как мы работаем

Короткие правила команды, чтобы история проекта оставалась понятной.

## Ветки
- `main` — стабильная версия, напрямую в неё не коммитим.
- Одна задача = одна ветка `stage/<N>-<name>`, например `stage/1-collect`.
- Одна ветка = один Pull Request в `main`. Описание PR — по шаблону, на русском.
- PR сливаем merge commit'ом после проверки вторым участником.

## Коммиты
Формат [Conventional Commits](https://www.conventionalcommits.org/ru/), на английском:

```
feat(collect): download PM2.5 from OpenAQ with pagination
fix(clean): drop negative PM2.5 values
docs(readme): add data sources
```

Типы: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.

## Перед коммитом
```bash
bash scripts/check.sh   # должно закончиться GATE PASS
```

## Правила кода
- Python 3.11+, стиль — `ruff`.
- Комментарии и markdown в ноутбуках — на русском; имена переменных и колонок — английский `snake_case` (`pm25`, `temp_c`, `wind_speed`).
- Каждый шаг ноутбука начинается markdown-ячейкой: что делаем и зачем.
- Сырые данные в `data/raw/` не меняем; очистка — функциями в `src/clean.py` с проверками.
- `merge` — всегда с `validate=` и проверкой числа строк.
- Модели — разбиение по времени, сравнение с наивным прогнозом.
- Графики — заголовок-вывод, оси с единицами (µg/m³, °C, м/с), сохранение через `savefig`.
- Секреты — только в `.env` (не коммитится); пример — `.env.example`.
