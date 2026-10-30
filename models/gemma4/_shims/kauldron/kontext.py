"""Stand-in for kauldron.kontext (config key plumbing only)."""
Key = str


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


class Path(tuple):
    @classmethod
    def from_jax_path(cls, path):
        return cls(str(p) for p in path)
