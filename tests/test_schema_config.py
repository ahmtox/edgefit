"""Config record invariants (PROJECT.md §6.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edgefit.schema import (
    ActivationQuant,
    CalibrationConfig,
    ConfigRecord,
    CoreMLComputeUnits,
    Dtype,
    Granularity,
    ModelRef,
    OrtProvider,
    OrtRuntimeConfig,
    QuantizationConfig,
    TaskType,
)

MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"


def test_round_trips_through_json(cpu_config: ConfigRecord) -> None:
    restored = ConfigRecord.model_validate_json(cpu_config.model_dump_json())
    assert restored == cpu_config
    assert restored.config_id == cpu_config.config_id


def test_is_frozen(cpu_config: ConfigRecord) -> None:
    with pytest.raises(ValidationError):
        cpu_config.label = "mutated"  # type: ignore[misc]


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ConfigRecord(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(),
            quantisation=None,  # British spelling typo must not silently vanish
        )


class TestConfigId:
    def test_is_stable_across_field_order(self) -> None:
        """Same semantics, different construction order -> same id."""
        first = ConfigRecord(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(providers=(OrtProvider.CPU,)),
            quantization=QuantizationConfig(weight_dtype=Dtype.INT8),
        )
        second = ConfigRecord(
            quantization=QuantizationConfig(weight_dtype=Dtype.INT8),
            runtime=OrtRuntimeConfig(providers=(OrtProvider.CPU,)),
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        )
        assert first.config_id == second.config_id

    def test_ignores_label(self, cpu_config: ConfigRecord) -> None:
        """A human tag carries no semantics and must not fork the id."""
        tagged = cpu_config.model_copy(update={"label": "baseline"})
        assert tagged.config_id == cpu_config.config_id

    def test_changes_with_semantics(
        self, cpu_config: ConfigRecord, coreml_config: ConfigRecord
    ) -> None:
        assert cpu_config.config_id != coreml_config.config_id

    def test_changes_with_schema_version(self, cpu_config: ConfigRecord) -> None:
        """A migration must produce new ids rather than silently reusing old ones."""
        bumped = cpu_config.model_copy(update={"schema_version": 2})
        assert bumped.config_id != cpu_config.config_id


class TestIntendedProvider:
    def test_is_the_first_non_cpu_provider(self, coreml_config: ConfigRecord) -> None:
        assert coreml_config.intended_provider == OrtProvider.COREML

    def test_falls_back_to_cpu(self, cpu_config: ConfigRecord) -> None:
        assert cpu_config.intended_provider == OrtProvider.CPU


class TestQuantizationCoherence:
    def test_blockwise_requires_block_size(self) -> None:
        with pytest.raises(ValidationError, match="block_size"):
            QuantizationConfig(weight_dtype=Dtype.INT4, weight_granularity=Granularity.BLOCKWISE)

    def test_block_size_rejected_when_not_blockwise(self) -> None:
        with pytest.raises(ValidationError, match="only meaningful"):
            QuantizationConfig(
                weight_dtype=Dtype.INT8,
                weight_granularity=Granularity.PER_CHANNEL,
                block_size=64,
            )

    def test_activation_dtype_requires_activation_quant(self) -> None:
        with pytest.raises(ValidationError):
            QuantizationConfig(weight_dtype=Dtype.INT8, activation_dtype=Dtype.INT8)

    def test_accepts_a_realistic_int4_blockwise_config(self) -> None:
        quant = QuantizationConfig(
            weight_dtype=Dtype.INT4,
            weight_granularity=Granularity.BLOCKWISE,
            block_size=64,
            activation_quant=ActivationQuant.DYNAMIC,
            activation_dtype=Dtype.INT8,
            kv_cache_dtype=Dtype.FP16,
        )
        assert quant.block_size == 64


class TestRuntimeCoherence:
    def test_rejects_duplicate_providers(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            OrtRuntimeConfig(providers=(OrtProvider.CPU, OrtProvider.CPU))

    def test_rejects_coreml_flags_without_coreml_provider(self) -> None:
        """A vendor flag that silently does nothing is how configs start lying."""
        with pytest.raises(ValidationError, match="CoreML EP flags"):
            OrtRuntimeConfig(
                providers=(OrtProvider.CPU,),
                coreml_compute_units=CoreMLComputeUnits.CPU_AND_NE,
            )


def test_static_activation_quant_requires_calibration() -> None:
    with pytest.raises(ValidationError, match="calibration"):
        ConfigRecord(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(),
            quantization=QuantizationConfig(
                weight_dtype=Dtype.INT8,
                activation_quant=ActivationQuant.STATIC,
                activation_dtype=Dtype.INT8,
            ),
        )


def test_static_activation_quant_accepted_with_calibration() -> None:
    config = ConfigRecord(
        model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        runtime=OrtRuntimeConfig(),
        quantization=QuantizationConfig(
            weight_dtype=Dtype.INT8,
            activation_quant=ActivationQuant.STATIC,
            activation_dtype=Dtype.INT8,
        ),
        calibration=CalibrationConfig(dataset_ref="generic:wikitext-2", sample_count=128),
    )
    assert config.calibration is not None
