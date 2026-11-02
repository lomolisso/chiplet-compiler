"""cc_export on toy Flax / Haiku / Keras ports: composites, nesting, symbols, invariance, self-checks."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
import toys

from cc_export import composite, exporter, symbols


def _export(tmp_path, port, **kw):
    return exporter.export_model("toy", "reduced", tmp_path, port=port, **kw)


def _graph(tmp_path, phase="forward"):
    return json.loads((tmp_path / phase / "graph.json").read_text())


def _all_checks_ok(summary):
    for name, rec in summary["phases"].items():
        assert "error" not in rec, (name, rec.get("error"))
        bad = {k: v for k, v in rec["checks"].items() if not v["ok"]}
        assert not bad, (name, bad)
    assert all(c["ok"] for c in summary["checks"].values()), summary["checks"]


def test_flax_nested_composites(tmp_path):
    s = _export(tmp_path, toys.flax_port())
    _all_checks_ok(s)
    g = _graph(tmp_path)
    by_path = {c["path"]: c for c in g["composites"]}
    # every submodule is a component; the phase root (Net) is not
    assert {"embed", "b0", "b0/up", "b0/down", "b1", "b1/up", "b1/down"} <= set(by_path)
    assert not any(c["name"] == "cc.flax.Net" for c in g["composites"])
    # nesting: Dense inside Block, Block at the top
    assert by_path["b0/up"]["parent"] == by_path["b0"]["instance"]
    assert by_path["b0"]["parent"] is None
    assert by_path["b0/up"]["leaf"] and not by_path["b0"]["leaf"]
    # the einsum of Block is inside Block's composite (not anon)
    assert g["coverage"]["anon_by_op"].get("stablehlo.dot_general", 0) == 0
    # outside composites: only the entry's shape assertions (integer size arithmetic)
    assert set(g["coverage"]["anon_by_op"]) <= {
        "stablehlo.constant", "stablehlo.get_dimension_size", "stablehlo.compare", "stablehlo.and",
        "stablehlo.custom_call", "stablehlo.convert", "stablehlo.reshape", "stablehlo.concatenate",
        "stablehlo.add", "stablehlo.subtract", "stablehlo.multiply", "stablehlo.select",
    }, g["coverage"]["anon_by_op"]


def test_flax_attributes_are_static_config(tmp_path):
    _export(tmp_path, toys.flax_port())
    g = _graph(tmp_path)
    funcs = {f["name"]: f for f in g["module"]["funcs"]}
    attrs = None
    for f in funcs.values():
        for op in f["body"]["blocks"][0]["ops"]:
            if op["name"] == "stablehlo.composite" and op["attributes"]["name"]["value"] == "cc.flax.Dense":
                attrs = op["attributes"]["composite_attributes"]["value"]
    assert attrs["features"] == {"kind": "int", "value": 16, "type": "i64"}
    assert attrs["use_bias"] == {"kind": "bool", "value": True}


def test_symbolic_config_fields_named_by_symbol(tmp_path):
    _export(tmp_path, toys.flax_port())
    text = (tmp_path / "forward" / "module.mlir").read_text()
    assert 'seq = "S"' in text and 'batch = "B"' in text


def test_size_variant_phase_identical(tmp_path):
    s = _export(tmp_path, toys.flax_port())
    assert s["phases"]["forward"]["hash"] == s["phases"]["forward_b1"]["hash"]
    assert s["checks"]["size_variants"]["ok"]


def test_invariance_catches_branch_on_config(tmp_path):
    s = _export(tmp_path, toys.flax_port(branch_on_cfg=True))
    inv = s["phases"]["forward"]["checks"]["invariance"]
    assert not inv["ok"]
    assert inv["differences"], inv


def test_python_shape_arithmetic_fails_export(tmp_path):
    s = _export(tmp_path, toys.flax_port(shape_arith=True))
    assert "InconclusiveDimensionOperation" in s["phases"]["forward"]["error"] or "error" in s["phases"]["forward"]


def test_symbols_recorded(tmp_path):
    _export(tmp_path, toys.flax_port())
    sy = json.loads((tmp_path / "forward" / "symbols.json").read_text())
    assert set(sy["symbols"]) == {"B", "S"}
    assert sy["symbols"]["S"]["inputs"] == [{"path": "[1]", "axis": 1, "expr": "S"}]
    assert sy["constraints"] == ["S >= 2"]
    assert sy["concrete_input_axes"] == [{"path": "[1]", "axis": 2, "size": 4}]
    assert {n: d["size"] for n, d in sy["symbols"].items()} == {"B": 2, "S": 8}
    assert _forward_checks(tmp_path)["symbol_sizes"] == {"ok": True, "errors": []}


def _forward_checks(tmp_path):
    return json.loads((tmp_path / "export.json").read_text())["phases"]["forward"]["checks"]


def test_solve_sizes_scaled_and_checked():
    where = {
        "H": [{"path": "[1]", "axis": 1, "expr": "32*H"}, {"path": "[2]", "axis": 0, "expr": "floordiv(H + 1, 2)"}],
        "B": [{"path": "[1]", "axis": 0, "expr": "B"}],
    }
    sizes, errors = symbols.solve_sizes(where, {"[1]": (3, 640), "[2]": (10,)})
    assert sizes == {"B": 3, "H": 20} and errors == []
    _, errors = symbols.solve_sizes(where, {"[1]": (3, 640), "[2]": (11,)})
    assert errors == ["[2][0] `floordiv(H + 1, 2)` = 10, concrete 11"]
    _, errors = symbols.solve_sizes({"N": [{"path": "[1]", "axis": 0, "expr": "N + 1"}]}, {"[1]": (5,)})
    assert errors[0] == "N: no input axis of the form `N` or `c*N`"


def test_no_symbols_fails_invariance_note(tmp_path):
    s = _export(tmp_path, toys.flax_port(symbolic=False))
    inv = s["phases"]["forward"]["checks"]["invariance"]
    assert inv["ok"] and "no symbols" in inv["note"]


def test_haiku_single_root_passes_through(tmp_path):
    s = _export(tmp_path, toys.haiku_port(single_root=True))
    _all_checks_ok(s)
    g = _graph(tmp_path)
    paths = {c["path"] for c in g["composites"]}
    assert s["phases"]["forward"]["roots"] == ["net"]
    assert "net" not in paths
    assert {"net/b0", "net/b0/up", "net/b0/norm", "net/b1/down"} <= paths
    names = {c["name"] for c in g["composites"]}
    assert {"cc.haiku.HkBlock", "cc.haiku.Linear", "cc.haiku.LayerNorm"} <= names


def test_haiku_several_top_level_modules_are_components(tmp_path):
    s = _export(tmp_path, toys.haiku_port(single_root=False))
    _all_checks_ok(s)
    paths = {c["path"] for c in _graph(tmp_path)["composites"]}
    assert {"b0", "b1", "b0/up"} <= paths


def test_keras_nested_layers(tmp_path):
    s = _export(tmp_path, toys.keras_port())
    _all_checks_ok(s)
    g = _graph(tmp_path)
    names = [c["name"] for c in g["composites"]]
    assert names.count("cc.keras.Dense") == 4 and names.count("cc.keras.Cross") == 1
    cross = next(c for c in g["composites"] if c["name"] == "cc.keras.Cross")
    kids = [c for c in g["composites"] if c["parent"] == cross["instance"]]
    assert len(kids) == 2


def test_census_matches_verify_hlo(tmp_path):
    s = _export(tmp_path, toys.flax_port())
    cc = s["phases"]["forward"]["checks"]["census_crosscheck"]
    assert cc["ok"]
    assert cc["verify_hlo"]["dot_general f32xf32->f32"]["count"] == 7
    con = json.loads((tmp_path / "forward" / "census_concrete.json").read_text())
    assert con["summary"]["dot_general f32xf32->f32"]["count"] == 7
    assert con["flops"] == con["summary"]["dot_general f32xf32->f32"]["flops"] > 0
    assert s["phases"]["forward"]["checks"]["census_concrete"] == {"ok": True}


def test_exported_bin_roundtrips(tmp_path):
    from jax import export

    _export(tmp_path, toys.flax_port())
    exp = export.deserialize(bytearray((tmp_path / "forward" / "exported.bin").read_bytes()))
    assert "B" in str(exp.in_avals[-1].shape[0])


def test_flax_mutable_cache_written_back(tmp_path):
    s = _export(tmp_path, toys.cache_port())
    _all_checks_ok(s)
    g = _graph(tmp_path)
    assert {c["path"] for c in g["composites"]} == {"attn", "attn/k"}
    # the updated cache leaves the composite and is a result of the phase
    attn_op = next(c for c in g["composites"] if c["path"] == "attn")
    funcs = {f["name"]: f for f in g["module"]["funcs"]}
    op = next(o for o in funcs[attn_op["func"]]["body"]["blocks"][0]["ops"] if o["id"] == attn_op["op"])
    assert len(op["results"]) == 2


def test_haiku_global_context_is_hoisted(tmp_path):
    s = _export(tmp_path, toys.haiku_port())
    _all_checks_ok(s)
    assert s["phases"]["forward"]["roots"] == ["net"]
    paths = {c["path"] for c in _graph(tmp_path)["composites"]}
    assert "net/scale" in paths
    assert json.loads((tmp_path / "forward" / "graph.json").read_text())["interception"]["hoisted"] > 0


def test_composite_under_vmap_and_dce():
    """vmap of a composite is the composite of the vmapped decomposition; dead code inside a composite is
    removed (both rules are missing from jax 0.12 and registered by cc_export)."""
    import jax
    from jax import export

    from cc_export import evaluate

    def comp(x, w):
        def dec(x, w):
            dead = jnp.sin(w) @ w            # result unused: must not survive
            del dead
            return jnp.tanh(x @ w), x * 2.0  # second output unused by the caller

        return composite.cc_composite(dec, "cc.toy")(x, w)[0]

    f = jax.vmap(comp, in_axes=(0, None))
    x = jax.random.normal(jax.random.key(0), (5, 3, 4))
    w = jax.random.normal(jax.random.key(1), (4, 4))
    exp = export.export(jax.jit(f), platforms=("cpu",))(x, w)
    text = exp.mlir_module()
    assert text.count('stablehlo.composite "cc.toy"') == 1
    assert "stablehlo.sine" not in text
    np.testing.assert_allclose(evaluate.evaluable(exp).call(x, w), jnp.tanh(x @ w), rtol=1e-6)


TINY = 4  # config.toml [testing] tiny_dim


@pytest.mark.parametrize("make", [toys.flax_port, toys.cache_port, toys.haiku_port,
                                  lambda: toys.haiku_port(getter=True), toys.keras_port])
def test_interception_preserves_semantics(tmp_path, make):
    """The exported module (composites, symbolic shapes) computes what the phase function computes, at
    tiny sizes with generated data (design §5.3)."""
    import jax

    from cc_export import evaluate

    port = make()
    _export(tmp_path, port, invariance=False)
    spec = port.build(port.CONFIGS["reduced"])
    ph = spec.phases[0]
    key = jax.random.key(7)
    leaves, tree = jax.tree.flatten((spec.param_specs, *ph.inputs))
    concrete = []
    for i, leaf in enumerate(leaves):
        shape = tuple(TINY if d in (2, 8) else d for d in leaf.shape)   # toy batch/seq -> tiny
        concrete.append(jax.random.normal(jax.random.fold_in(key, i), shape, leaf.dtype))
    args = jax.tree.unflatten(tree, concrete)
    got = evaluate.load(tmp_path / "forward" / "exported.bin").call(*args)
    want = ph.fn(*args)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "value, expected",
    [
        (jnp.float32, "float32"),
        (np.dtype("int8"), "int8"),
        ((1, (2, 3)), (1, (2, 3))),
        (None, None),
        ({"b": 1, "a": None}, {"b": 1}),
        (2**70, str(2**70)),
        (np.arange(3), (0, 1, 2)),
        (np.zeros((4, 4)), "array<4x4xfloat64>"),
    ],
)
def test_attr_value(value, expected):
    assert composite.attr_value(value) == expected


def test_attr_value_callables_are_names():
    assert composite.attr_value(jnp.tanh).endswith("tanh")
    assert "0x" not in composite.attr_value(lambda x: x)
