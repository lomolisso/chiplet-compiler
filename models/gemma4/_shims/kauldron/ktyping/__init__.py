"""Stand-in for kauldron.ktyping: shape-spec carrying array annotation types."""
from kauldron.ktyping import array_type_meta  # noqa: F401
from kauldron.ktyping.array_type_meta import ArrayTypeMeta


def _make(name):
    return ArrayTypeMeta(name, (), {"shape_spec": None})


Array = _make("Array")
Float = _make("Float")
Int = _make("Int")
Bool = _make("Bool")
UInt8 = _make("UInt8")
PyTree = _make("PyTree")
PRNGKey = _make("PRNGKey")


def typechecked(fn=None, **_):
    """No-op (runtime shape checking is not part of the computation)."""
    if fn is None:
        return lambda f: f
    return fn


def __getattr__(name):
    if name[:1].isupper():
        return _make(name)
    raise AttributeError(name)
