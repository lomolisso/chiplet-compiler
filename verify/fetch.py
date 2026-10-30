"""Fetch upstream metadata without downloading weights.

* `hf_manifest(repo, ...)`: tensor name -> (shape, dtype) from the safetensors headers
  of a HuggingFace checkpoint, read with HTTP range requests (8-byte length + JSON
  header per shard, never the tensor data). Sharded checkpoints are resolved through
  `*.safetensors.index.json`.
* `hf_json(repo, path)`, `raw(url)`: small config files.

Responses are cached under verify/.cache/ (gitignored), so reruns are offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import urllib.request

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
HF = "https://huggingface.co"


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    key = hashlib.sha1(f"{url}|{rng}".encode()).hexdigest()
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": "dnn-graph-decomp-verify"})
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith(HF):
        req.add_header("Authorization", f"Bearer {tok}")
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def raw(url: str) -> bytes:
    return _get(url)


def byte_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end] (inclusive) of a remote file, cached (headers of large files)."""
    return _get(url, (start, end))


def hf_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF}/{repo}/resolve/{revision}/{path}"


def hf_json(repo: str, path: str, revision: str = "main"):
    return json.loads(_get(hf_url(repo, path, revision)))


def safetensors_header(url: str) -> dict:
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    return json.loads(_get(url, (8, 8 + n - 1)))


def hf_manifest(repo: str, filename: str = "model.safetensors", revision: str = "main",
                subfolder: str | None = None) -> dict[str, tuple[tuple[int, ...], str]]:
    """{tensor name: (shape, dtype)} of a (possibly sharded) safetensors checkpoint."""
    pre = f"{subfolder}/" if subfolder else ""
    try:
        idx = hf_json(repo, f"{pre}{filename}.index.json", revision)
        shards = sorted(set(idx["weight_map"].values()))
    except Exception:
        shards = [filename]
    out = {}
    for shard in shards:
        hdr = safetensors_header(hf_url(repo, pre + shard, revision))
        for name, meta in hdr.items():
            if name != "__metadata__":
                out[name] = (tuple(meta["shape"]), meta["dtype"])
    return out
