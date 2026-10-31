# Third-party submodules (`third_party/`)

Git submodules (shallow). Each pin is recorded in `config.toml` `[tools.*]`. `chipc doctor` checks that the checkout, the commit the superproject records and the Docker image all agree. Tools are built only inside the image (`docker/`), never on the host.

| Submodule | Upstream | Pin (commit) | `config.toml` key | Used by |
|---|---|---|---|---|
| `grami/` | https://github.com/ehab-abdelhamid/GraMi | `d2875ae80afa2f141323a3f6b89a387787fcda38` (2018-06-03, last upstream commit) | `tools.grami.commit` | `mine` (directed build) |
| `allo/` | https://github.com/cornell-zhang/allo | `ac21bae49ca5f912ecebea00908b56976d2361a2` (`main`, 2026-10-12) | `tools.allo.commit` | `softfloat`, `allo_emit`, `xls` |
| `xls/` | https://github.com/google/xls | `8ddae8bf466742285aa2d405470512cc73a9aeec` = release tag `v0.0.0-10898-g8ddae8bf4` (2026-10-24) | `tools.xls.commit`, `tools.xls.tag` | `xls`, `characterize` |

The Allo pin also fixes LLVM. Allo's own submodule `externals/llvm-project` records `6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb` (`tools.allo.llvm_commit`). Nested submodules are **not** checked out on the host: `/home` is NFS and nearly full, and LLVM is large. The image build fetches LLVM at that commit itself. Allo's other nested submodule, `past-python-bindings`, is not needed: its `requirements.txt` installs the pinned `past` wheel.

## Patches

`patches/<submodule>/*.patch` are applied inside the image build only. A submodule is never edited in place.

| Patch | Why |
|---|---|
| `patches/grami/0001-drop-unused-javax-imports.patch` | Removes `import javax.jws.Oneway;` and `import javax.rmi.CORBA.Util;` (both unused) from the two `*_SUBGRAPHS` trees. Java 11 removed those modules (JEP 320), so GraMi does not compile with the image's Java 17 otherwise (`docs/tools/grami.md` §3). |

`chipc doctor` checks that every patch still applies (`git apply --check`).

## Fresh clone

```bash
git submodule update --init --depth 1        # non-recursive: no LLVM on the host
```

## Updating a pin

1. Fetch the new commit into the submodule and check it out:
   ```bash
   git -C third_party/<name> fetch --depth 1 origin <sha>
   git -C third_party/<name> checkout <sha>
   ```
   XLS: take the commit of a release tag, so the release tarball and the Bazel-built tools come from the same source.
2. Update `config.toml`: the commit. XLS also needs `tag` and `release_sha256`, which is the release asset's digest from the GitHub releases API. For Allo, set `llvm_commit` to `git -C third_party/allo ls-tree HEAD externals/llvm-project`.
3. Check that the patches still apply: `git -C third_party/grami apply --check ../patches/grami/*.patch`.
4. Bump `[tools.docker] tag` and rebuild the image (`docker/build.sh`, see `docs/toolchain-image.md`).
5. `git add third_party/<name> config.toml`, then run `chipc doctor` until it is green.
