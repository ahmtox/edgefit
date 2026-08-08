"""Inferring a ModelSpec from a HuggingFace config.

The rule under test is **infer or refuse, never approximate**. A model measured through
the wrong input harness does not fail — it returns a plausible number for a workload
nobody asked about, which is hard rule #1's failure mode in a place where the result
looks entirely reasonable. So the refusals are as load-bearing as the successes, and
they are tested for naming the specific obstacle rather than saying "unsupported".
"""

from __future__ import annotations

import pytest

from edgefit.models.infer import (
    DEFAULT_IMAGE,
    DEFAULT_SEQUENCE,
    UninferableModelError,
    infer_spec,
)
from edgefit.models.registry import REGISTRY, UnknownModelError, resolve
from edgefit.schema.common import TaskType


def spec(architectures, **config):
    return infer_spec("hf:org/model", {"architectures": architectures, **config})


class TestHeads:
    def test_causal_lm_is_a_decoder(self) -> None:
        s = spec(["LlamaForCausalLM"], model_type="llama", max_position_embeddings=131072)
        assert s.task is TaskType.GENERATE
        assert s.exporter == "decoder"
        assert s.hf_class == "AutoModelForCausalLM"
        assert s.output_attr == "logits"
        # No sequence axis is pinned: a KV-cached decoder requires a dynamic one.
        assert s.static_shape == {"batch": 1}

    def test_sequence_classifier_emits_logits(self) -> None:
        s = spec(["DistilBertForSequenceClassification"], model_type="distilbert")
        assert (s.task, s.exporter, s.output_attr) == (TaskType.CLASSIFY, "text", "logits")
        assert s.hf_class == "AutoModelForSequenceClassification"

    def test_image_classifier_is_vision(self) -> None:
        s = spec(["ViTForImageClassification"], model_type="vit", image_size=384)
        assert (s.task, s.exporter) == (TaskType.VISION, "vision")
        assert s.static_shape["height"] == 384

    def test_bare_encoder_is_an_embedding(self) -> None:
        s = spec(["BertModel"], model_type="bert")
        assert (s.task, s.exporter, s.output_attr) == (
            TaskType.EMBED, "text", "last_hidden_state",
        )
        assert s.hf_class == "AutoModel"

    def test_masked_lm_is_measured_as_an_encoder(self) -> None:
        """The head is irrelevant to what we time; the encoder stack is the workload."""
        assert spec(["RobertaForMaskedLM"], model_type="roberta").exporter == "text"

    def test_a_bare_vision_encoder_is_detected_by_its_image_size(self) -> None:
        """`ViTModel` has no head, so only the presence of an image tells us modality."""
        s = spec(["ViTModel"], model_type="vit", image_size=224)
        assert (s.task, s.exporter) == (TaskType.VISION, "vision")

    def test_the_suffix_is_matched_not_the_whole_name(self) -> None:
        """Bert and DistilBert classifiers are the same measurement problem."""
        a = spec(["BertForSequenceClassification"], model_type="bert")
        b = spec(["XLMRobertaForSequenceClassification"], model_type="xlm-roberta")
        assert a.exporter == b.exporter and a.output_attr == b.output_attr


class TestShapes:
    def test_sequence_is_fixed_for_comparability(self) -> None:
        """Not the model's own maximum: that would measure sequence length, not models."""
        s = spec(["BertModel"], model_type="bert", max_position_embeddings=512)
        assert s.static_shape["sequence"] == DEFAULT_SEQUENCE

    def test_a_shorter_model_limit_is_respected(self) -> None:
        """Only shortened when the model genuinely cannot take the default."""
        s = spec(["BertModel"], model_type="bert", max_position_embeddings=64)
        assert s.static_shape["sequence"] == 64

    def test_image_size_comes_from_the_config(self) -> None:
        s = spec(["ViTForImageClassification"], model_type="vit", image_size=518)
        assert (s.static_shape["height"], s.static_shape["width"]) == (518, 518)

    def test_image_size_falls_back_when_absent(self) -> None:
        s = spec(["ViTForImageClassification"], model_type="vit")
        assert s.static_shape["height"] == DEFAULT_IMAGE

    def test_channel_count_is_read_not_assumed(self) -> None:
        s = spec(["ViTForImageClassification"], model_type="vit", num_channels=1)
        assert s.static_shape["channels"] == 1


class TestRefusals:
    """Each refusal names the field that defeated it, so nobody has to read our source."""

    def test_seq2seq_is_refused_because_it_has_two_stacks(self) -> None:
        with pytest.raises(UninferableModelError, match="is_encoder_decoder"):
            spec(["T5ForConditionalGeneration"], model_type="t5", is_encoder_decoder=True)

    def test_multimodal_is_refused_because_there_is_no_single_graph(self) -> None:
        with pytest.raises(UninferableModelError, match="vision and a text tower"):
            spec(["CLIPModel"], model_type="clip", vision_config={"image_size": 224},
                 text_config={"vocab_size": 49408})

    def test_a_config_without_architectures_is_refused(self) -> None:
        with pytest.raises(UninferableModelError, match="no `architectures`"):
            infer_spec("hf:org/model", {"model_type": "mystery"})

    def test_an_empty_config_is_refused(self) -> None:
        with pytest.raises(UninferableModelError, match="no readable config"):
            infer_spec("hf:org/model", {})

    def test_an_unknown_head_is_refused_and_lists_what_we_know(self) -> None:
        with pytest.raises(UninferableModelError) as caught:
            spec(["WhisperForAudioClassification"], model_type="whisper")
        message = str(caught.value)
        assert "WhisperForAudioClassification" in message
        assert "ForCausalLM" in message, "the refusal should say what we do handle"

    def test_refusals_are_not_silent_approximations(self) -> None:
        """The whole point: a wrong harness returns a plausible number, not an error."""
        for config in (
            {"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True},
            {"architectures": ["SomethingExotic"]},
            {},
        ):
            with pytest.raises(UninferableModelError):
                infer_spec("hf:org/model", config)


class TestResolve:
    def test_the_registry_wins_over_inference(self) -> None:
        """Hand-written specs are corrections; inference must not override them.

        bart-base is the case that proves it: inference refuses it outright as
        encoder-decoder, while the registry pins its encoder.
        """
        spec = resolve("hf:facebook/bart-base")
        assert spec.submodule == "encoder"
        assert spec is REGISTRY["hf:facebook/bart-base"]

    def test_clip_keeps_its_pinned_vision_tower(self) -> None:
        assert resolve("hf:openai/clip-vit-base-patch32").hf_class == "CLIPVisionModel"

    def test_inference_can_be_switched_off_for_replay_paths(self) -> None:
        """Replaying a cached artifact must not reach for the network."""
        with pytest.raises(UnknownModelError, match="inference is off"):
            resolve("hf:org/never-seen", infer=False)

    def test_a_non_hf_ref_cannot_be_inferred(self) -> None:
        with pytest.raises(UnknownModelError, match="only 'hf:"):
            resolve("file:/tmp/model.onnx")


class TestHierarchicalConfigs:
    """Per-stage architectures, which crashed the exporter before they were handled.

    Swin describes itself as `num_heads=[3,6,12,24]` and `depths=[2,2,6,2]` — one entry
    per stage. `int()` on that raises, so a whole model was unexportable, surfacing as a
    TypeError mid-export rather than as a bad number. The crash was the better failure,
    but still a failure.
    """

    def test_a_per_stage_config_does_not_crash(self) -> None:
        from edgefit.backends.export_onnx import architecture_from_config

        class Cfg:
            num_attention_heads = [3, 6, 12, 24]
            num_hidden_layers = [2, 2, 6, 2]

        assert architecture_from_config(Cfg()) == {"stages": 4}

    def test_no_scalar_head_count_is_invented(self) -> None:
        """There is no honest scalar for four stages with different head counts.

        A max or a sum would put a number in the fingerprint that no part of the model
        has — and the fingerprint is what a cost model indexes on. Same rule as the
        attention variant: reported exactly, or not reported.
        """
        from edgefit.backends.export_onnx import architecture_from_config

        class Cfg:
            num_attention_heads = [3, 6, 12, 24]
            num_hidden_layers = [2, 2, 6, 2]

        arch = architecture_from_config(Cfg())
        assert "n_heads" not in arch
        assert "kv_heads" not in arch
        assert "layers" not in arch

    def test_scalar_configs_are_unaffected(self) -> None:
        from edgefit.backends.export_onnx import architecture_from_config

        class Cfg:
            num_attention_heads = 12
            num_hidden_layers = 12

        assert architecture_from_config(Cfg()) == {"n_heads": 12, "kv_heads": 12, "layers": 12}
