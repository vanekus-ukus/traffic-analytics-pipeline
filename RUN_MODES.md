# Run Modes

Все команды предполагают запуск из корня репозитория.

## Базовый запуск
```bash
cp .env.example .env
bash scripts/setup_env.sh
bash scripts/start_postgres.sh
bash scripts/init_db.sh
```

Перед batch и fallback-режимами нужно указать свои пути во `.env`.

## Batch
```bash
bash scripts/run_batch.sh
```

## Streaming
```bash
bash scripts/run_streaming.sh --source "<video_or_stream_source>"
```

## Segmented live stream
```bash
bash scripts/run_live_stream.sh --source "<video_or_stream_source>"
```

## Live viewer
```bash
bash scripts/run_live_view.sh --source "<video_or_stream_source>"
```

`<video_or_stream_source>` может быть:
- локальным видеофайлом
- прямым media URL
- HTML-страницей с поддерживаемым player config

В окне и в консоли показываются:
- `frame_counts`
- `active_tracks`
- `lifetime_tracks`
- `new_last_10s`
- `top_speed`

## Пример под плотный поток
```bash
bash scripts/run_live_view.sh \
  --source "<video_or_stream_source>" \
  --scene-preset fast_road \
  --track-stale-seconds 6 \
  --stitch-gap-seconds 4 \
  --stitch-distance-px 160 \
  --stitch-min-iou 0.04 \
  --reid-memory-seconds 25 \
  --reid-distance-px 220 \
  --overlap-iou 0.60 \
  --person-min-track-hits 3 \
  --person-min-duration-seconds 0.35 \
  --static-duration-seconds 8 \
  --static-movement-px 24 \
  --static-min-hits 14 \
  --speed-smoothing-alpha 0.35 \
  --speed-kmh-factor 0.18 \
  --speed-window-seconds 2.0 \
  --speed-max-jump-ratio 2.4 \
  --speed-reset-gap-seconds 0.6 \
  --speed-max-kmh-estimated 140 \
  --renderer ffplay \
  --log-level INFO
```

## Viewer tuning
Основные параметры:
- `--target-fps` - частота кадров для обработки
- `--imgsz` - размер входа модели
- `--confidence` - порог детекции

Счётчик и треки:
- `--stitch-gap-seconds` - время жизни трека при пропадании
- `--stitch-distance-px` - расстояние сшивки
- `--stitch-min-iou` - минимальное перекрытие для плотного потока
- `--reid-memory-seconds` - память о недавно пропавшем объекте
- `--reid-distance-px` - расстояние повторного привязывания
- `--min-track-hits` - порог подтверждения для транспорта
- `--edge-min-track-hits` - отдельный порог у края кадра
- `--person-min-track-hits` - порог подтверждения для `person`
- `--person-min-duration-seconds` - минимальная длительность для `person`

Подавление ложных объектов:
- `--overlap-iou` - подавление перекрывающихся bbox
- `--static-duration-seconds` - длительность для статичного ложного трека
- `--static-movement-px` - допустимый сдвиг статичного ложного трека
- `--static-min-hits` - минимальное число наблюдений для статичного ложного трека
- `--suppression-padding-px` - размер зоны подавления

Скорость:
- `--speed-smoothing-alpha` - сглаживание скорости
- `--speed-kmh-factor` - коэффициент перевода proxy speed в `km/h est`
- `--speed-window-seconds` - окно истории скорости
- `--speed-max-jump-ratio` - ограничение скачков скорости
- `--speed-reset-gap-seconds` - сброс speed history после длинного разрыва
- `--speed-max-kmh-estimated` - верхний предел оценочной скорости

## Viewer renderer
Если OpenCV GUI недоступен:
```bash
bash scripts/run_live_view.sh --source "<video_or_stream_source>" --renderer ffplay
```

## Типичные проблемы

### `bus` и `truck` путаются
Обычно это происходит, когда крупный транспорт сначала виден далеко или частично, а трек слишком рано теряет историю.

Что помогает:
- увеличить `--imgsz`
- увеличить `--track-stale-seconds`
- увеличить `--stitch-gap-seconds`
- увеличить `--reid-memory-seconds`
- не ставить слишком высокий `--confidence`

Пример:
```bash
bash scripts/run_live_view.sh \
  --source "<video_or_stream_source>" \
  --scene-preset fast_road \
  --imgsz 896 \
  --confidence 0.18 \
  --track-stale-seconds 7 \
  --stitch-gap-seconds 5 \
  --stitch-distance-px 180 \
  --stitch-min-iou 0.04 \
  --reid-memory-seconds 30 \
  --reid-distance-px 240 \
  --renderer ffplay \
  --log-level INFO
```

### Метки моргают, часть машин не считается
Обычно это нехватка вычислений или слишком жёсткая сшивка треков.

Что помогает:
- уменьшить `--target-fps`
- уменьшить `--imgsz`
- увеличить `--track-stale-seconds`
- увеличить `--reid-memory-seconds`
- уменьшить `--stitch-distance-px`, если разные машины слипаются в один трек

Пример:
```bash
bash scripts/run_live_view.sh \
  --source "<video_or_stream_source>" \
  --target-fps 8 \
  --imgsz 640 \
  --track-stale-seconds 6 \
  --stitch-gap-seconds 4 \
  --stitch-distance-px 140 \
  --reid-memory-seconds 20 \
  --reid-distance-px 180 \
  --renderer ffplay \
  --log-level INFO
```

### Ложные `person` или `car` на статичных объектах
Обычно это повторяющиеся короткие ложные треки на фоне или конструкции рядом с дорогой.

Что помогает:
- увеличить `--person-min-track-hits`
- увеличить `--person-min-duration-seconds`
- увеличить `--static-duration-seconds`
- увеличить `--static-min-hits`

Пример:
```bash
bash scripts/run_live_view.sh \
  --source "<video_or_stream_source>" \
  --person-min-track-hits 4 \
  --person-min-duration-seconds 0.50 \
  --static-duration-seconds 10 \
  --static-min-hits 18 \
  --renderer ffplay \
  --log-level INFO
```

### Скорость иногда скачет слишком сильно
Обычно это происходит после потери трека и повторного появления объекта.

Что помогает:
- уменьшить `--speed-max-jump-ratio`
- уменьшить `--speed-reset-gap-seconds`
- уменьшить `--speed-max-kmh-estimated`
- увеличить `--speed-window-seconds`

Пример:
```bash
bash scripts/run_live_view.sh \
  --source "<video_or_stream_source>" \
  --speed-window-seconds 2.5 \
  --speed-max-jump-ratio 1.8 \
  --speed-reset-gap-seconds 0.4 \
  --speed-max-kmh-estimated 110 \
  --renderer ffplay \
  --log-level INFO
```

## Полный запуск
```bash
bash scripts/run_all.sh
```

## Остановка PostgreSQL
```bash
bash scripts/stop_postgres.sh
```
