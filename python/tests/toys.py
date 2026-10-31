"""Toy ports (Flax, Haiku, Keras) with the `CONFIGS` / `build` interface of `models/<key>/model.py`.

Everything is abstract: parameters come from `jax.eval_shape`, nothing is executed.
"""

from __future__ import annotations

import dataclasses
import types

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, Sym, as_spec


@dataclasses.dataclass(frozen=True)
class Cfg:
    batch: int = 2
    seq: int = 8
    width: int = 16


# --- Flax -------------------------------------------------------------------------------------------


class Block(nn.Module):
    cfg: Cfg

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(self.cfg.width, name="up")(x))
        s = jnp.einsum("bsf,btf->bst", h, h)          # function-level op inside the parent module
        return nn.Dense(self.cfg.width, name="down")(h) + s.mean(-1, keepdims=True)


class Net(nn.Module):
    cfg: Cfg
    branch_on_cfg: bool = False      # toy defect: structure depends on an input-size config field
    shape_arith: bool = False        # toy defect: Python arithmetic on an input dim

    @nn.compact
    def __call__(self, x):
        y = Block(self.cfg, name="b0")(nn.Dense(self.cfg.width, name="embed")(x))
        y = Block(self.cfg, name="b1")(y)
        if self.branch_on_cfg and self.cfg.seq > 8:
            y = y * 2.0
        if self.shape_arith:
            y = y[:, : int(x.shape[1]) // 2]
        return y


def flax_port(branch_on_cfg=False, shape_arith=False, symbolic=True):
    def build(cfg: Cfg) -> ModelSpec:
        net = Net(cfg, branch_on_cfg, shape_arith)
        x = jax.ShapeDtypeStruct((cfg.batch, cfg.seq, 4), jnp.float32)
        params = jax.eval_shape(lambda: net.init(jax.random.key(0), jnp.zeros((1, 2, 4))))

        def forward(p, x):
            return net.apply(p, x)

        def forward_b1(p, x):
            return net.apply(p, x)

        return ModelSpec(
            name="toy_flax",
            variant="toy",
            phases=[
                Phase("forward", forward, (x,), shapes=("B, S, _",) if symbolic else None),
                Phase("forward_b1", forward_b1, (jax.ShapeDtypeStruct((1, cfg.seq, 4), jnp.float32),),
                      shapes=("B, S, _",) if symbolic else None, size_variant_of="forward"),
            ],
            param_specs=as_spec(params),
            symbols={"B": Sym("batch", 3), "S": Sym("seq", 12)} if symbolic else {},
            constraints=("S >= 2",) if symbolic else (),
        )

    return types.SimpleNamespace(CONFIGS={"reduced": Cfg()}, build=build)


class CachedAttn(nn.Module):
    """Writes its keys into a mutable `cache` collection (created on the first call)."""
    cfg: Cfg

    @nn.compact
    def __call__(self, x):
        k = nn.Dense(self.cfg.width, name="k")(x)
        ck = self.variable("cache", "keys", jnp.zeros, k.shape, k.dtype)
        ck.value = ck.value + k
        return jnp.einsum("bsf,btf->bst", k, ck.value)


class CacheNet(nn.Module):
    cfg: Cfg

    @nn.compact
    def __call__(self, x):
        return CachedAttn(self.cfg, name="attn")(x)


def cache_port():
    def build(cfg: Cfg) -> ModelSpec:
        net = CacheNet(cfg)
        x = jax.ShapeDtypeStruct((cfg.batch, cfg.seq, 4), jnp.float32)
        params = jax.eval_shape(lambda: net.init(jax.random.key(0), jnp.zeros((1, 2, 4)))["params"])

        def step(p, x):
            return net.apply({"params": p}, x, mutable=["cache"])

        return ModelSpec(
            name="toy_cache", variant="toy",
            phases=[Phase("forward", step, (x,), shapes=("B, S, _",))],
            param_specs=as_spec(params),
            symbols={"B": Sym("batch", 3), "S": Sym("seq", 12)},
        )

    return types.SimpleNamespace(CONFIGS={"reduced": Cfg()}, build=build)


# --- Haiku ------------------------------------------------------------------------------------------


class HkBlock(hk.Module):
    def __init__(self, width, name=None):
        super().__init__(name=name)
        self.width = width

    def __call__(self, x):
        h = jax.nn.relu(hk.Linear(self.width, name="up")(x))
        w = hk.get_parameter("mix", (self.width, self.width), init=jnp.zeros)
        h = jnp.einsum("bsf,fg->bsg", h, w)
        return hk.LayerNorm(-1, True, True, name="norm")(hk.Linear(self.width, name="down")(h))


class HkNet(hk.Module):
    def __init__(self, width, name=None):
        super().__init__(name=name)
        self.width = width

    def __call__(self, x):
        y = HkBlock(self.width, name="b1")(HkBlock(self.width, name="b0")(x))
        if GRAPH:                            # reads an array from a global context, not its arguments
            y = HkScale(name="scale")(y)
        return y


class HkScale(hk.Module):
    def __call__(self, y):
        return y * GRAPH[0]


def _double(next_getter, value, context):
    return next_getter(value * 2.0)          # a port-side custom getter (e.g. a dequantization)


GRAPH = []                                   # a port-side global context (as graphcast's static graph)


def haiku_port(single_root=True, getter=False):
    def build(cfg: Cfg) -> ModelSpec:
        def body(x):
            if single_root:
                return HkNet(cfg.width, name="net")(x)
            return HkBlock(cfg.width, name="b1")(HkBlock(cfg.width, name="b0")(x))

        def net(x):
            if getter:
                with hk.custom_getter(_double):
                    return body(x)
            return body(x)

        f = hk.without_apply_rng(hk.transform(net))
        x = jax.ShapeDtypeStruct((cfg.batch, cfg.seq, 4), jnp.float32)
        g = jax.ShapeDtypeStruct((cfg.width,), jnp.float32)
        params = jax.eval_shape(lambda: f.init(jax.random.key(0), jnp.zeros((1, 2, 4))))

        def forward(p, x, g):
            GRAPH.append(g)
            try:
                return f.apply(p, x)
            finally:
                GRAPH.clear()

        return ModelSpec(
            name="toy_haiku",
            variant="toy",
            phases=[Phase("forward", forward, (x, g), shapes=("B, S, _", None))],
            param_specs=as_spec(params),
            symbols={"B": Sym("batch", 3), "S": Sym("seq", 12)},
        )

    return types.SimpleNamespace(CONFIGS={"reduced": Cfg()}, build=build)


# --- Keras ------------------------------------------------------------------------------------------


def keras_port():
    import keras

    class Cross(keras.layers.Layer):
        def __init__(self, d, **kw):
            super().__init__(**kw)
            self.d = d
            self.down = keras.layers.Dense(4)
            self.up = keras.layers.Dense(d)

        def call(self, x):
            return x * self.up(self.down(x)) + x

        def get_config(self):
            return {**super().get_config(), "d": self.d}

    def make(cfg):
        inp = keras.Input((cfg.width,))
        h = keras.layers.Dense(cfg.width, activation="relu")(inp)
        h = Cross(cfg.width)(h)
        return keras.Model(inp, keras.layers.Dense(1)(h))

    def build(cfg: Cfg) -> ModelSpec:
        with keras.StatelessScope(initialize_variables=False):
            m = make(cfg)
            params = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in m.trainable_variables]

        def forward(vals, x):
            with keras.StatelessScope(initialize_variables=False):
                model = make(cfg)
                out, _ = model.stateless_call(vals, [], x)
            return out

        x = jax.ShapeDtypeStruct((cfg.batch, cfg.width), jnp.float32)
        return ModelSpec(
            name="toy_keras",
            variant="toy",
            phases=[Phase("forward", forward, (x,), shapes=("B, _",))],
            param_specs=params,
            symbols={"B": Sym("batch", 3)},
        )

    return types.SimpleNamespace(CONFIGS={"reduced": Cfg()}, build=build)
