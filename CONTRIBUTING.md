# Contributing

## Workflow

1. **One change, one branch, one worktree.** Branches are named `<type>/<topic>`, for example `feat/import`:
   ```bash
   git pull --ff-only
   git worktree add .worktrees/feat-import -b feat/import
   scripts/build-area init feat-import          # the worktree's build dir on the build host
   ```
   Submodules only when the change needs them: `git -C .worktrees/feat-import submodule update --init`
   (never `--recursive`).
2. **Build and test on the build host** through `scripts/on-build-host`, never on the orchestration host (see
   [docs/development.md](docs/development.md)).
3. **Push the branch for review:** `git push -u origin feat/import`. Review fixes are new commits on the same
   branch; never force-push a branch under review. To undo a commit, use `git revert <sha>`.
4. **Open a pull request** to `main` once the change is approved. CI (`lint` and `test`) must pass before it can be
   merged. Pull requests are squash-merged, so the pull request title becomes the commit on `main`.
5. **Clean up** after the merge (the remote branch is deleted automatically):
   ```bash
   git pull --ff-only
   scripts/build-area clean feat-import
   git worktree remove .worktrees/feat-import
   git branch -D feat/import
   ```
6. **Undo a merged change** with a pull request that reverts it (`revert: ...`). History on `main` is never
   rewritten.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org): `type(scope): subject`, imperative, lower case, at most
72 characters, no trailing period. Add a body only when the subject is not enough.

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`, `revert`. A breaking change adds
`!` after the type (`feat(export)!: rename the census file`). Typical scopes: a pass name (`export`, `import`), a
crate area (`core`, `pipeline`, `cli`, `env`), `models`, `docker`, `scripts`.

The pull request title follows the same format, because it becomes the commit message on `main`.

## Changelog and versions

The project follows [Semantic Versioning](https://semver.org) (`0.x` while interfaces change). Every `feat` or
`fix` pull request (and every breaking change) adds one short line under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md), in the section that fits (`Added`, `Changed`, `Fixed`, `Removed`). CI checks this.

A release is a `chore(release): vX.Y.Z` pull request that sets the version in `Cargo.toml` and `pyproject.toml` and
moves the unreleased lines under `## [X.Y.Z] - <date>`. After the merge, tagging the merge commit `vX.Y.Z` and pushing
the tag publishes the GitHub release.

## Documentation

Detailed documentation lives in [docs/](docs/README.md); folder READMEs elsewhere stay short. A new pass adds
`docs/passes/<pass>.md` (sections listed in [docs/development.md](docs/development.md)).
