Запуск:
1. cp .env.example .env
2. bash scripts/setup_env.sh
3. bash scripts/start_postgres.sh
4. bash scripts/init_db.sh
5. bash scripts/run_batch.sh
6. bash scripts/run_streaming.sh

Live:
bash scripts/run_live_stream.sh --source "<video_or_stream_source>"

Просмотр:
bash scripts/run_live_view.sh --source "<video_or_stream_source>"
