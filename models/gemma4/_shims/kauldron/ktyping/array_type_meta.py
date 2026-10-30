"""Stand-in for kauldron.ktyping.array_type_meta."""


class ArrayTypeMeta(type):
    shape_spec = None

    def __getitem__(cls, spec):
        return ArrayTypeMeta(f"{cls.__name__}[{spec!r}]", (), {"shape_spec": spec})


def is_array_type(ann) -> bool:
    return isinstance(ann, ArrayTypeMeta) and isinstance(ann.shape_spec, str)
