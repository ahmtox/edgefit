"""Runtime backends. One per layer-1 runtime we lower to."""

from edgefit.backends.base import Backend, DeviceRun, StaticAnalysis
from edgefit.backends.ort import OrtBackend
from edgefit.schema.common import RuntimeKind

_BACKENDS: dict[RuntimeKind, type] = {
    RuntimeKind.ONNXRUNTIME: OrtBackend,
}


class UnsupportedBackendError(Exception):
    """No backend is implemented for this runtime yet."""


def get_backend(kind: RuntimeKind) -> Backend:
    """Instantiate the backend for a runtime kind."""
    try:
        return _BACKENDS[kind]()  # type: ignore[return-value]
    except KeyError as exc:
        available = ", ".join(sorted(str(k) for k in _BACKENDS))
        raise UnsupportedBackendError(
            f"no backend for runtime {kind!r}. Available: {available}"
        ) from exc


__all__ = [
    "Backend",
    "DeviceRun",
    "OrtBackend",
    "StaticAnalysis",
    "UnsupportedBackendError",
    "get_backend",
]
