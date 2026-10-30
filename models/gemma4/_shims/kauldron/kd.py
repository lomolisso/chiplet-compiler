"""Stand-in for the `kauldron.kd` namespace (only names used by gemma4)."""
import types

import flax.linen as _nn

from kauldron import ktyping as typing  # noqa: F401


class Identity(_nn.Module):
    """Same as kauldron's kd.nn.Identity: returns its input (an activation tap)."""

    @_nn.compact
    def __call__(self, x):
        return x


nn = types.SimpleNamespace(Identity=Identity)
sharding = types.SimpleNamespace(ShardingTree=object,
                                 with_sharding_constraint=lambda x, sharding=None: x)
