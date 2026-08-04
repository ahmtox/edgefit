"""Where did each node actually execute? (PROJECT.md §2.2)

> "A delegate reports a successful partition, but a subset of ops it couldn't
> claim quietly execute on CPU. Nothing errors. Nothing warns. A model that
> should run at 20ms runs at 400ms and the team has no idea why."

ONNX Runtime does not expose partition decisions through its Python API, and it
refuses to serialise a graph containing compiled nodes. What it *does* expose is
a profiling trace in which every executed kernel carries its provider. Nodes the
accelerator claimed collapse into fused ``<EP>_..._<n>`` entries; everything else
appears under its original graph node name.

That gives exact attribution — verified empirically: with graph optimisation
disabled, 100% of CPU-side kernel events map back to original ONNX node names.

Three ratios are produced because they disagree, and the disagreement is the
point. On all-MiniLM-L6-v2 with the CoreML EP, node share reports ~46% fallback
while FLOP share reports ~99%: every MatMul in the model landed on CPU while the
accelerator claimed a scattering of cheap reshapes. A team reading only the node
number would conclude the delegate was working.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from onnx import ModelProto

from edgefit.backends.analysis.flops import FlopsTable
from edgefit.schema.measurement import FallbackReport

_KERNEL_SUFFIX = "_kernel_time"

# Fused partitions are named "<EP>_<hash>_<EP-short>_<hash>_<i>_<j>". We only
# need to recognise them, not decode them.
_FUSED_NODE = re.compile(r"^[A-Za-z]+ExecutionProvider_")

# Structural ops that vanish during graph construction (folded to initializers
# or elided). Counting them in the denominator would understate fallback.
_NON_EXECUTING_OPS = frozenset({"Constant", "Identity"})


@dataclass(frozen=True)
class KernelEvent:
    node_name: str
    op_type: str | None
    provider: str
    duration_us: float


def parse_profile(profile_path: Path | str) -> list[KernelEvent]:
    """Read ORT's profiling JSON into per-kernel execution events."""
    events = json.loads(Path(profile_path).read_text())
    parsed: list[KernelEvent] = []
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = event.get("name", "")
        if not name.endswith(_KERNEL_SUFFIX):
            continue
        args = event.get("args", {}) or {}
        provider = args.get("provider")
        if not provider:
            continue
        parsed.append(
            KernelEvent(
                node_name=name[: -len(_KERNEL_SUFFIX)],
                op_type=args.get("op_name"),
                provider=provider,
                duration_us=float(event.get("dur", 0.0)),
            )
        )
    return parsed


def _executing_nodes(model: ModelProto) -> dict[str, str]:
    """Original graph nodes that actually run, as name -> op_type."""
    return {
        node.name: node.op_type
        for node in model.graph.node
        if node.name and node.op_type not in _NON_EXECUTING_OPS
    }


def build_fallback_report(
    model: ModelProto,
    events: list[KernelEvent],
    intended_provider: str,
    flops: FlopsTable | None = None,
    runs: int = 1,
) -> FallbackReport:
    """Attribute the graph to providers and compute the three fallback ratios.

    ``runs`` is the number of inference calls captured in the profile; per-node
    events repeat once per run and are deduplicated by node name.
    """
    graph_nodes = _executing_nodes(model)
    nodes_total = len(graph_nodes)

    # Deduplicate across runs; a node executes on the same provider every time.
    provider_by_node: dict[str, str] = {}
    time_by_provider: Counter[str] = Counter()
    fused_partitions: set[str] = set()

    for event in events:
        time_by_provider[event.provider] += event.duration_us
        if _FUSED_NODE.match(event.node_name):
            fused_partitions.add(event.node_name)
        else:
            provider_by_node[event.node_name] = event.provider

    # Nodes seen executing off the intended provider are the fallback set.
    # Nodes never seen individually were absorbed into a fused partition, which
    # means the intended provider claimed them.
    fell_back = {
        name
        for name, provider in provider_by_node.items()
        if provider != intended_provider and name in graph_nodes
    }
    nodes_on_intended = nodes_total - len(fell_back)

    node_pct = 100.0 * len(fell_back) / nodes_total if nodes_total else 0.0

    # --- FLOP share: the honest one, when shapes resolved ---
    flops_total = flops_on_intended = None
    flops_pct = None
    if flops is not None and flops.is_complete and flops.total > 0:
        flops_total = flops.total
        fallback_flops = flops.subtotal(fell_back)
        flops_on_intended = flops_total - fallback_flops
        flops_pct = 100.0 * fallback_flops / flops_total

    # --- Time share: measured, but includes dispatch overhead ---
    total_us = sum(time_by_provider.values())
    time_pct = None
    intended_us = None
    if total_us > 0:
        intended_us = time_by_provider.get(intended_provider, 0.0)
        time_pct = 100.0 * (total_us - intended_us) / total_us
        intended_us /= max(runs, 1)
        total_us /= max(runs, 1)

    unclaimed = Counter(graph_nodes[name] for name in fell_back)

    return FallbackReport(
        intended_provider=intended_provider,
        nodes_total=nodes_total,
        nodes_on_intended=max(nodes_on_intended, 0),
        fallback_node_pct=round(node_pct, 4),
        flops_total=flops_total,
        flops_on_intended=flops_on_intended,
        fallback_flops_pct=round(flops_pct, 4) if flops_pct is not None else None,
        time_total_us=round(total_us, 3) if total_us else None,
        time_on_intended_us=round(intended_us, 3) if intended_us is not None else None,
        fallback_time_pct=round(time_pct, 4) if time_pct is not None else None,
        nodes_per_provider=dict(Counter(provider_by_node.values())) | (
            {intended_provider + " (fused partitions)": len(fused_partitions)}
            if fused_partitions
            else {}
        ),
        unclaimed_op_types=dict(unclaimed.most_common()),
    )


def build_as_run_report(
    events: list[KernelEvent],
    intended_provider: str,
    graph_optimization: str,
    runs: int = 1,
) -> FallbackReport:
    """Attribute the graph *as ORT actually executed it*.

    Necessary because graph optimisation rewrites the graph before partitioning, and
    it does so substantially: on ViT-base the CPU node count fell from 244 to 86 and
    CoreML's measured time share rose from 53.7% to 81.5% between level `disabled`
    and level `all`. A fallback figure taken from the unoptimized graph therefore
    does not describe the run that was timed.

    Fusion destroys the original node names, so this report deliberately carries no
    FLOP attribution — the denominator would be a guess. Its honest contents are
    time share, partition count, and the executed-node split.

    Note what time share is *not*: an efficiency measure. It says where the time
    went, not whether sending that work to the accelerator was a good idea. MiniLM
    spends 70.5% of its time on CoreML and is still 2x slower than plain CPU.
    """
    fused: set[str] = set()
    executed_by_provider: Counter[str] = Counter()
    time_by_provider: Counter[str] = Counter()

    for event in events:
        time_by_provider[event.provider] += event.duration_us
        if _FUSED_NODE.match(event.node_name):
            fused.add(event.node_name)
        else:
            executed_by_provider[event.provider] += 1

    # Each fused partition is one executed node from ORT's point of view.
    per_run = max(runs, 1)
    nodes_off_intended = sum(
        count for provider, count in executed_by_provider.items() if provider != intended_provider
    ) // per_run
    nodes_on_intended = len(fused) + (executed_by_provider.get(intended_provider, 0) // per_run)
    nodes_total = nodes_on_intended + nodes_off_intended

    total_us = sum(time_by_provider.values())
    intended_us = time_by_provider.get(intended_provider, 0.0)

    return FallbackReport(
        intended_provider=intended_provider,
        nodes_total=nodes_total,
        nodes_on_intended=nodes_on_intended,
        fallback_node_pct=(
            round(100.0 * nodes_off_intended / nodes_total, 4) if nodes_total else 0.0
        ),
        time_total_us=round(total_us / per_run, 3) if total_us else None,
        time_on_intended_us=round(intended_us / per_run, 3) if total_us else None,
        fallback_time_pct=(
            round(100.0 * (total_us - intended_us) / total_us, 4) if total_us else None
        ),
        nodes_per_provider={
            provider: count // per_run for provider, count in executed_by_provider.items()
        }
        | ({f"{intended_provider} (fused partitions)": len(fused)} if fused else {}),
        partition_count=len(fused) or None,
        node_basis="as_executed",
        analysis_graph_optimization=graph_optimization,
    )
