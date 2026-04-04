# transport-analytics-platform

Локальный проект для обработки дорожного видео и исторических данных.

## Что внутри
- PostgreSQL со схемами `raw`, `core`, `dm`, `ops`
- batch-пайплайн для CSV
- video pipeline на `YOLO Ultralytics`
- live-режим с обработкой коротких сегментов
- представления для BI

## Структура
- `src/` - код
- `scripts/` - запуск
- `sql/init/` - SQL
- `data/` - входные файлы
- `notebooks/` - ноутбук
- `ARCHITECTURE.md` - краткая схема
- `assets/readme/` - примеры детекции

## Примеры

Кадр с детекцией:

![demo road](assets/readme/demo-road.jpg)

Кадр из live-просмотра:

![demo live](assets/readme/demo-live.jpg)

GIF с live-детекцией:

![demo stream](assets/readme/demo-stream.gif)

## Запуск
```bash
cp .env.example .env
bash scripts/setup_env.sh
bash scripts/start_postgres.sh
bash scripts/init_db.sh
bash scripts/run_batch.sh
bash scripts/run_streaming.sh
```

Сквозной запуск:
```bash
bash scripts/run_all.sh
```

Остановка БД:
```bash
bash scripts/stop_postgres.sh
```

## Потоковый режим
```bash
bash scripts/run_streaming.sh --source "<video_or_stream_source>"
```

Live-просмотр:
```bash
bash scripts/run_live_view.sh --source "<video_or_stream_source>"
```

В окне и в консоли показываются:
- `frame_counts` - объекты на текущем кадре
- `total_tracks` - накопительный счётчик уникальных объектов с начала запуска

## Что пишется в БД
Таблицы:
- `ops.pipeline_runs`
- `ops.data_quality_checks`
- `raw.tracking_source_rows`
- `raw.traffic_30min_source`
- `core.detection_events`
- `core.tracked_objects`
- `dm.streaming_metrics`
- `dm.batch_metrics`

Представления:
- `dm.v_datalens_live_traffic`
- `dm.v_datalens_historical_traffic`
- `dm.v_datalens_vehicle_mix`
- `dm.v_datalens_stream_vs_history`
- `dm.v_datalens_traffic_all`

## Проверка
```bash
psql -h /tmp/traffic_pg_socket -p 55432 -U transport_user -d transport_analytics -c "SELECT COUNT(*) FROM core.detection_events;"
psql -h /tmp/traffic_pg_socket -p 55432 -U transport_user -d transport_analytics -c "SELECT COUNT(*) FROM dm.streaming_metrics;"
psql -h /tmp/traffic_pg_socket -p 55432 -U transport_user -d transport_analytics -c "SELECT COUNT(*) FROM dm.batch_metrics;"
```

## Ограничения
- скорость по видео считается как proxy
- качество детекции зависит от камеры
- стабильность live зависит от доступности потока и `ffmpeg/yt-dlp`
- исторический CSV не является готовой train-разметкой для YOLO
