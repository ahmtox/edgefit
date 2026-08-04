"""Corpus schema.

Every table stores the lossless canonical JSON of the record in ``payload``,
plus denormalised columns for the hot query paths. The JSON is the source of
truth — it survives schema evolution — while the columns are what the atlas and
the future cost model actually scan. Columns are derived at insert time and are
never the authority.

There is no ``UPDATE`` or ``DELETE`` anywhere in this system by design
(PROJECT.md §14.3).
"""

from __future__ import annotations

DDL = """
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id          VARCHAR PRIMARY KEY,
    schema_version     INTEGER NOT NULL,
    model_ref          VARCHAR NOT NULL,
    task               VARCHAR NOT NULL,
    runtime_kind       VARCHAR NOT NULL,
    intended_provider  VARCHAR NOT NULL,
    providers          VARCHAR NOT NULL,
    weight_dtype       VARCHAR,
    weight_granularity VARCHAR,
    activation_quant   VARCHAR,
    quant_algorithm    VARCHAR,
    num_threads        INTEGER,
    label              VARCHAR,
    payload            VARCHAR NOT NULL,
    first_seen_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_fingerprints (
    fingerprint_id     VARCHAR PRIMARY KEY,
    n_nodes            INTEGER NOT NULL,
    n_parameters       BIGINT NOT NULL,
    attention_variant  VARCHAR NOT NULL,
    norm_type          VARCHAR NOT NULL,
    payload            VARCHAR NOT NULL,
    first_seen_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id        VARCHAR PRIMARY KEY,
    schema_version        INTEGER NOT NULL,
    harness_version       VARCHAR NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL,

    recipe_id             VARCHAR NOT NULL,
    model_ref             VARCHAR NOT NULL,
    graph_fingerprint_id  VARCHAR,

    device_id             VARCHAR NOT NULL,
    sku_id                VARCHAR NOT NULL,
    device_model          VARCHAR NOT NULL,
    soc                   VARCHAR NOT NULL,
    os_name               VARCHAR NOT NULL,
    os_version            VARCHAR NOT NULL,
    os_build              VARCHAR NOT NULL,

    measurement_source    VARCHAR NOT NULL,
    source_detail         VARCHAR,
    reported_latency_ms   DOUBLE,
    stress_profile        VARCHAR NOT NULL,
    outcome               VARCHAR NOT NULL,
    failure_reason        VARCHAR,
    run_count             INTEGER NOT NULL,
    warmup_count          INTEGER NOT NULL,

    latency_p50_ms        DOUBLE,
    latency_p95_ms        DOUBLE,
    latency_cv            DOUBLE,
    ttft_p50_ms           DOUBLE,
    decode_tok_s_p50      DOUBLE,
    sustained_tok_s_5min  DOUBLE,
    peak_rss_bytes        BIGINT,
    artifact_bytes        BIGINT,
    lowering_ms           DOUBLE,
    accuracy              DOUBLE,
    accuracy_delta_vs_fp16 DOUBLE,
    output_cosine_vs_reference DOUBLE,
    power_mw              DOUBLE,

    fallback_node_pct     DOUBLE,
    fallback_flops_pct    DOUBLE,
    fallback_time_pct     DOUBLE,
    as_run_node_pct       DOUBLE,
    as_run_time_pct       DOUBLE,
    as_run_partitions     INTEGER,

    power_source          VARCHAR NOT NULL,
    low_power_mode        BOOLEAN,
    thermal_state         VARCHAR NOT NULL,
    load_avg_1m           DOUBLE,
    calibration_ratio     DOUBLE,

    payload               VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS measurements_by_config ON measurements (recipe_id);
CREATE INDEX IF NOT EXISTS measurements_by_device ON measurements (device_id);
CREATE INDEX IF NOT EXISTS measurements_by_model  ON measurements (model_ref);
"""
