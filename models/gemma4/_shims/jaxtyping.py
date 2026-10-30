"""Stand-in for jaxtyping (only `_array_types.AbstractArray` is probed)."""
import types


class AbstractArray:
    pass


_array_types = types.SimpleNamespace(AbstractArray=AbstractArray)
