//! The environment: what `chipc doctor` checks and builds outside the pass DAG. It writes only
//! `build/.env/` (manifest, verify report, liberty `.db` files). See `README.md` in this directory.

pub mod doctor;
pub mod pdk_db;

pub use doctor::{DoctorOptions, doctor};

use cc_core::{Config, CoreError};

use crate::error::PipelineError;

/// `KEY=VALUE` lines for `docker/build.sh`: the Dockerfile `ARG`s, all from `config.toml`, so the
/// config is the single source of every pin. Keys starting with `CAP_` are `docker build` flags
/// (memory, CPUs), not build args. An empty pin is an error, never a silent default.
pub fn build_args(cfg: &Config) -> Result<Vec<(String, String)>, PipelineError> {
    let b = &cfg.tools.docker.build;
    let pins = [
        ("UBUNTU_2404", b.ubuntu_2404.as_str()),
        ("UBUNTU_2204", b.ubuntu_2204.as_str()),
        ("APT_SNAPSHOT", b.apt_snapshot.as_str()),
        ("PYPI_EXCLUDE_NEWER", b.pypi_exclude_newer.as_str()),
        ("UV_VERSION", b.uv_version.as_str()),
        ("UV_SHA256", b.uv_sha256.as_str()),
        ("BAZELISK_VERSION", b.bazelisk_version.as_str()),
        ("BAZELISK_SHA256", b.bazelisk_sha256.as_str()),
        ("Z3_VERSION", b.z3_version.as_str()),
        ("Z3_SHA256", b.z3_sha256.as_str()),
        ("BITWUZLA_VERSION", b.bitwuzla_version.as_str()),
        ("BITWUZLA_SHA256", b.bitwuzla_sha256.as_str()),
        ("XLS_TAG", cfg.tools.xls.tag.as_str()),
        ("XLS_COMMIT", cfg.tools.xls.commit.as_str()),
        ("XLS_RELEASE_SHA256", cfg.tools.xls.release_sha256.as_str()),
        ("ALLO_COMMIT", cfg.tools.allo.commit.as_str()),
        ("LLVM_COMMIT", cfg.tools.allo.llvm_commit.as_str()),
        ("GRAMI_COMMIT", cfg.tools.grami.commit.as_str()),
        ("IMAGE_TAG", cfg.tools.docker.tag.as_str()),
    ];
    let mut out = Vec::new();
    for (k, v) in pins {
        if v.trim().is_empty() || v.contains(char::is_whitespace) {
            return Err(PipelineError::Core(CoreError::Config {
                layer: "merged config".into(),
                message: format!("build arg {k} is empty or contains whitespace (config.toml [tools.*])"),
            }));
        }
        out.push((k.to_string(), v.to_string()));
    }
    out.push(("JOBS".into(), b.cpus.to_string()));
    out.push(("LINK_JOBS".into(), b.link_jobs.to_string()));
    out.push(("BAZEL_RAM_MB".into(), b.bazel_ram_mb.to_string()));
    out.push(("CAP_MEMORY_GB".into(), b.memory_gb.to_string()));
    out.push(("CAP_CPUS".into(), b.cpus.to_string()));
    out.push((
        "CAP_IMAGE".into(),
        format!("{}:{}", cfg.tools.docker.image, cfg.tools.docker.tag),
    ));
    Ok(out)
}
