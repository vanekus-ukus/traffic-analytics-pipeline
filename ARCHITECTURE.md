# ARCHITECTURE

## Состав
- `src/traffic_analytics/` - код пайплайнов и утилит
- `scripts/` - shell-обёртки запуска
- `sql/init/` - схемы, таблицы и представления
- `data/` - входные файлы
- `notebooks/report.ipynb` - короткий отчётный ноутбук

## Контуры

### Batch
- читает CSV из `data/`
- грузит источники в `raw`
- считает агрегаты в `dm.batch_metrics`
- обновляет BI-представления

### Streaming
- принимает локальный файл, media URL или page URL
- запускает YOLO Ultralytics
- пишет события в `core.detection_events`
- считает оконные метрики в `dm.streaming_metrics`

### Live
- получает короткие сегменты потока
- обрабатывает их по очереди
- пытается сшивать треки между соседними сегментами
- сохраняет события и метрики в те же таблицы

## База

### Схемы
- `ops`
- `raw`
- `core`
- `dm`

### Основные таблицы
- `ops.pipeline_runs`
- `ops.data_quality_checks`
- `raw.tracking_source_rows`
- `raw.traffic_30min_source`
- `core.detection_events`
- `core.tracked_objects`
- `dm.batch_metrics`
- `dm.streaming_metrics`

### Представления
- `dm.v_datalens_live_traffic`
- `dm.v_datalens_historical_traffic`
- `dm.v_datalens_vehicle_mix`
- `dm.v_datalens_stream_vs_history`
- `dm.v_datalens_traffic_all`

## Ограничения
- скорость по видео считается как proxy
- качество зависит от сцены и угла камеры
- для live-источников устойчивость зависит от доступности потока и `ffmpeg`
- исторический CSV не является готовым train dataset для YOLO
