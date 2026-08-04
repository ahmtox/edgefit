"""Recipe invariants and composition (PROJECT.md §6.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edgefit.cli.recipes import available_recipes, load_recipe
from edgefit.schema import (
    RECIPE_SCHEMA_VERSION,
    ActivationQuant,
    CalibrationConfig,
    CoreMLComputeUnits,
    Dtype,
    ExecutionConfig,
    Granularity,
    ModelRef,
    OrtProvider,
    OrtRuntimeConfig,
    QuantizationConfig,
    Recipe,
    TaskType,
)

MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"


def test_round_trips_through_json(cpu_recipe: Recipe) -> None:
    restored = Recipe.model_validate_json(cpu_recipe.model_dump_json())
    assert restored == cpu_recipe
    assert restored.recipe_id == cpu_recipe.recipe_id


def test_is_frozen(cpu_recipe: Recipe) -> None:
    with pytest.raises(ValidationError):
        cpu_recipe.label = "mutated"  # type: ignore[misc]


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Recipe(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(),
            quantisation=None,  # British spelling typo must not silently vanish
        )


class TestConfigId:
    def test_is_stable_across_field_order(self) -> None:
        """Same semantics, different construction order -> same id."""
        first = Recipe(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(providers=(OrtProvider.CPU,)),
            quantization=QuantizationConfig(weight_dtype=Dtype.INT8),
        )
        second = Recipe(
            quantization=QuantizationConfig(weight_dtype=Dtype.INT8),
            runtime=OrtRuntimeConfig(providers=(OrtProvider.CPU,)),
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        )
        assert first.recipe_id == second.recipe_id

    def test_ignores_label(self, cpu_recipe: Recipe) -> None:
        """A human tag carries no semantics and must not fork the id."""
        tagged = cpu_recipe.model_copy(update={"label": "baseline"})
        assert tagged.recipe_id == cpu_recipe.recipe_id

    def test_changes_with_semantics(
        self, cpu_recipe: Recipe, coreml_recipe: Recipe
    ) -> None:
        assert cpu_recipe.recipe_id != coreml_recipe.recipe_id

    def test_changes_with_schema_version(self, cpu_recipe: Recipe) -> None:
        """A migration must produce new ids rather than silently reusing old ones."""
        bumped = cpu_recipe.model_copy(update={"schema_version": RECIPE_SCHEMA_VERSION + 1})
        assert bumped.recipe_id != cpu_recipe.recipe_id


class TestIntendedProvider:
    def test_is_the_first_non_cpu_provider(self, coreml_recipe: Recipe) -> None:
        assert coreml_recipe.intended_provider == OrtProvider.COREML

    def test_falls_back_to_cpu(self, cpu_recipe: Recipe) -> None:
        assert cpu_recipe.intended_provider == OrtProvider.CPU


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
        Recipe(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(),
            quantization=QuantizationConfig(
                weight_dtype=Dtype.INT8,
                activation_quant=ActivationQuant.STATIC,
                activation_dtype=Dtype.INT8,
            ),
        )


def test_static_activation_quant_accepted_with_calibration() -> None:
    recipe = Recipe(
        model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        runtime=OrtRuntimeConfig(),
        quantization=QuantizationConfig(
            weight_dtype=Dtype.INT8,
            activation_quant=ActivationQuant.STATIC,
            activation_dtype=Dtype.INT8,
        ),
        calibration=CalibrationConfig(dataset_ref="generic:wikitext-2", sample_count=128),
    )
    assert recipe.calibration is not None


class TestComposition:
    """PROJECT.md §6.1: recipes compose and inherit from expert-vetted defaults.

    This is the difference between a recipe object and a dict, and it is what makes
    a known-good baseline usable as search warm-start material instead of something
    to copy-paste.
    """

    def test_derive_merges_a_section(self, cpu_recipe: Recipe) -> None:
        derived = cpu_recipe.derive(execution={"num_threads": 4})
        assert derived.execution.num_threads == 4
        assert cpu_recipe.execution.num_threads is None, "the base must not be mutated"

    def test_derive_preserves_untouched_sections(self, coreml_recipe: Recipe) -> None:
        """One field change must not silently reset its neighbours."""
        derived = coreml_recipe.derive(execution={"num_threads": 2})
        assert derived.runtime.providers == coreml_recipe.runtime.providers

    def test_derive_merges_within_a_section_rather_than_replacing_it(self) -> None:
        base = Recipe(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=OrtRuntimeConfig(
                providers=(OrtProvider.COREML, OrtProvider.CPU),
                coreml_model_format="NeuralNetwork",
                coreml_compute_units=CoreMLComputeUnits.ALL,
            ),
        )
        derived = base.derive(runtime={"coreml_compute_units": "CPUAndNeuralEngine"})
        assert derived.runtime.coreml_compute_units is CoreMLComputeUnits.CPU_AND_NE
        assert derived.runtime.coreml_model_format == "NeuralNetwork"

    def test_derive_accepts_a_model_instance_to_replace_a_section(
        self, cpu_recipe: Recipe
    ) -> None:
        derived = cpu_recipe.derive(execution=ExecutionConfig(num_threads=8, batch_size=4))
        assert (derived.execution.num_threads, derived.execution.batch_size) == (8, 4)

    def test_derive_changes_the_recipe_id(self, cpu_recipe: Recipe) -> None:
        assert cpu_recipe.derive(execution={"num_threads": 4}).recipe_id != cpu_recipe.recipe_id

    def test_derive_rejects_an_unknown_section(self, cpu_recipe: Recipe) -> None:
        with pytest.raises(ValueError, match="unknown recipe section"):
            cpu_recipe.derive(quantisation={"weight_dtype": "int8"})

    def test_derive_still_validates(self, cpu_recipe: Recipe) -> None:
        """Composition must not be a way around the coherence rules."""
        with pytest.raises(ValidationError):
            cpu_recipe.derive(execution={"num_threads": 0})

    def test_label_is_settable_and_still_outside_the_id(self, cpu_recipe: Recipe) -> None:
        derived = cpu_recipe.derive(label="tagged")
        assert derived.label == "tagged"
        assert derived.recipe_id == cpu_recipe.recipe_id


class TestRecipeLibrary:
    """`recipes/` is the expert-default library of PROJECT.md §6.1."""

    def _write(self, tmp_path, name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body)
        return path

    def test_extends_inherits_the_base(self, tmp_path) -> None:
        self._write(
            tmp_path,
            "base.yaml",
            "label: base\nruntime:\n  kind: onnxruntime\n"
            "  providers: [CoreMLExecutionProvider, CPUExecutionProvider]\n"
            "  coreml_model_format: NeuralNetwork\n",
        )
        child = self._write(
            tmp_path,
            "child.yaml",
            "extends: base.yaml\nlabel: child\nruntime:\n"
            "  coreml_compute_units: CPUAndNeuralEngine\n",
        )
        recipe = load_recipe(child, MINILM)
        assert recipe.label == "child"
        assert recipe.runtime.coreml_compute_units is CoreMLComputeUnits.CPU_AND_NE
        # inherited, not reset by the partial override
        assert recipe.runtime.coreml_model_format == "NeuralNetwork"
        assert OrtProvider.COREML in recipe.runtime.providers

    def test_detects_a_cycle_instead_of_overflowing(self, tmp_path) -> None:
        self._write(tmp_path, "a.yaml", "extends: b.yaml\n")
        self._write(tmp_path, "b.yaml", "extends: a.yaml\n")
        with pytest.raises(ValueError, match="circular recipe inheritance"):
            load_recipe(tmp_path / "a.yaml", MINILM)

    def test_shipped_library_loads_and_is_coherent(self) -> None:
        """Every preset in the repo must actually be a valid recipe."""
        presets = available_recipes()
        assert presets, "recipe library is empty"
        for path in presets:
            recipe = load_recipe(path, MINILM)
            assert recipe.label, f"{path} has no label"
            assert recipe.runtime.providers


class TestLowering:
    """A recipe must fully determine its artifact (the sweep's caching invariant)."""

    def test_lowering_participates_in_the_id(self, cpu_recipe: Recipe) -> None:
        dynamic = cpu_recipe.derive(lowering={"static_shapes": False})
        assert dynamic.recipe_id != cpu_recipe.recipe_id

    def test_opset_participates_in_the_id(self, cpu_recipe: Recipe) -> None:
        assert cpu_recipe.derive(lowering={"opset": 18}).recipe_id != cpu_recipe.recipe_id

    def test_defaults_to_static_shapes(self, cpu_recipe: Recipe) -> None:
        """Dynamic shape is the biggest reason a delegate declines a subgraph."""
        assert cpu_recipe.lowering.static_shapes is True
