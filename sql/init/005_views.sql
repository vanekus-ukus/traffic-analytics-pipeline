CREATE OR REPLACE VIEW dm.v_datalens_traffic_all AS
SELECT
    'streaming'::TEXT AS metric_source,
    metric_id,
    run_id,
    camera_id,
    window_start,
    window_end,
    window_granularity,
    total_count,
    car_count,
    truck_count,
    bus_count,
    motorcycle_count,
    bicycle_count,
    person_count,
    avg_speed_proxy,
    avg_speed_kmh,
    movement_score_avg,
    occupancy_proxy,
    congestion_proxy,
    heavy_vehicle_share,
    NULL::TEXT AS weather_type,
    NULL::INTEGER AS weather_code,
    NULL::NUMERIC AS temperature,
    NULL::NUMERIC AS precipitation
FROM dm.streaming_metrics
UNION ALL
SELECT
    'batch'::TEXT AS metric_source,
    metric_id,
    run_id,
    camera_id,
    window_start,
    window_end,
    window_granularity,
    total_count,
    car_count,
    truck_count,
    bus_count,
    motorcycle_count,
    bicycle_count,
    person_count,
    avg_speed_proxy,
    avg_speed_kmh,
    movement_score_avg,
    occupancy_proxy,
    congestion_proxy,
    heavy_vehicle_share,
    weather_type,
    weather_code,
    temperature,
    precipitation
FROM dm.batch_metrics;

CREATE OR REPLACE VIEW dm.v_datalens_vehicle_mix AS
SELECT
    metric_source,
    camera_id,
    window_start,
    window_end,
    window_granularity,
    class_name AS vehicle_class,
    class_count AS vehicle_count,
    CASE
        WHEN total_count = 0 THEN NULL
        ELSE ROUND(class_count::NUMERIC / total_count, 6)
    END AS vehicle_share
FROM (
    SELECT
        metric_source,
        camera_id,
        window_start,
        window_end,
        window_granularity,
        total_count,
        values_table.class_name,
        values_table.class_count
    FROM dm.v_datalens_traffic_all
    CROSS JOIN LATERAL (
        VALUES
            ('car', car_count),
            ('truck', truck_count),
            ('bus', bus_count),
            ('motorcycle', motorcycle_count),
            ('bicycle', bicycle_count),
            ('person', person_count)
    ) AS values_table(class_name, class_count)
) s;

CREATE OR REPLACE VIEW dm.v_datalens_live_traffic AS
SELECT *
FROM dm.streaming_metrics
ORDER BY window_start DESC, camera_id;

CREATE OR REPLACE VIEW dm.v_datalens_historical_traffic AS
SELECT *
FROM dm.batch_metrics
ORDER BY window_start DESC, camera_id;

CREATE OR REPLACE VIEW dm.v_datalens_stream_vs_history AS
WITH stream_slots AS (
    SELECT
        metric_id,
        camera_id,
        window_start,
        window_end,
        total_count,
        COALESCE(avg_speed_proxy, avg_speed_kmh) AS current_speed,
        ((EXTRACT(HOUR FROM window_start)::INT * 60 + EXTRACT(MINUTE FROM window_start)::INT) / 30)::INT AS slot_30m
    FROM dm.streaming_metrics
),
history_slots AS (
    SELECT
        camera_id,
        ((EXTRACT(HOUR FROM window_start)::INT * 60 + EXTRACT(MINUTE FROM window_start)::INT) / 30)::INT AS slot_30m,
        AVG(total_count)::NUMERIC AS historical_total_count,
        AVG(COALESCE(avg_speed_kmh, avg_speed_proxy))::NUMERIC AS historical_avg_speed
    FROM dm.batch_metrics
    GROUP BY camera_id, ((EXTRACT(HOUR FROM window_start)::INT * 60 + EXTRACT(MINUTE FROM window_start)::INT) / 30)::INT
)
SELECT
    s.metric_id,
    s.camera_id,
    s.window_start,
    s.window_end,
    s.total_count AS current_total_count,
    h.historical_total_count,
    s.current_speed AS current_avg_speed,
    h.historical_avg_speed,
    s.total_count - COALESCE(h.historical_total_count, 0) AS delta_intensity_abs,
    CASE
        WHEN COALESCE(h.historical_total_count, 0) = 0 THEN NULL
        ELSE ROUND((s.total_count - h.historical_total_count) / h.historical_total_count, 6)
    END AS delta_intensity_pct,
    s.current_speed - COALESCE(h.historical_avg_speed, 0) AS delta_speed_abs,
    CASE
        WHEN COALESCE(h.historical_avg_speed, 0) = 0 THEN NULL
        ELSE ROUND((s.current_speed - h.historical_avg_speed) / h.historical_avg_speed, 6)
    END AS delta_speed_pct
FROM stream_slots s
LEFT JOIN history_slots h
    ON h.slot_30m = s.slot_30m;

