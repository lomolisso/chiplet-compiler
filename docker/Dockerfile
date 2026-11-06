# chiplet-compiler toolchain image. Build only with docker/build.sh, which passes every pin
# from config.toml (`chipc doctor --build-args`) and the memory/CPU caps (docs/toolchain-image.md).
#
# This host's Docker has no buildx, so this file targets the legacy builder: `docker build
# --memory/--memory-swap/--cpu-quota` caps every RUN step; no BuildKit-only syntax.
# Stages run in file order (legacy builder): cheap downloads first, the two long builds last.
# Every stage carries the chiplet-compiler.stage label so our dangling stage images can be pruned
# by label without touching other users' images on the shared daemon.

ARG UBUNTU_2404
ARG UBUNTU_2204

# ---------------------------------------------------------------------------------------------
# base: Ubuntu 24.04 (Python 3.12, glibc 2.39), apt frozen to one snapshot.ubuntu.com date.
# ---------------------------------------------------------------------------------------------
FROM ${UBUNTU_2404} AS base
LABEL chiplet-compiler.stage=base
ARG APT_SNAPSHOT
ARG UV_VERSION
ARG UV_SHA256
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# snapshot.ubuntu.com is https-only, so ca-certificates comes first from the regular archive (apt
# verifies the archive signatures either way). Then APT::Snapshot makes every later apt call in
# this and derived stages use the same snapshot.
RUN test -n "${APT_SNAPSHOT}" \
 && apt-get update --error-on=any && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && echo "APT::Snapshot \"${APT_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50chipc-snapshot \
 && apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip python3.12 python3.12-venv \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /tmp/uv.tgz "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
 && echo "${UV_SHA256}  /tmp/uv.tgz" | sha256sum -c - \
 && tar -xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
 && rm /tmp/uv.tgz && uv --version

# build: compilers for the source builds (not in the final image except what Verilator needs).
FROM base AS build
LABEL chiplet-compiler.stage=build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
# Downloads (checksummed).
# ---------------------------------------------------------------------------------------------
FROM base AS xls-release
LABEL chiplet-compiler.stage=xls-release
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG XLS_TAG
ARG XLS_RELEASE_SHA256
RUN curl -fsSLo /tmp/xls.tgz "https://github.com/google/xls/releases/download/${XLS_TAG}/xls-${XLS_TAG}-linux-x64.tar.gz" \
 && echo "${XLS_RELEASE_SHA256}  /tmp/xls.tgz" | sha256sum -c - \
 && mkdir -p /opt/xls/bin /opt/xls/share \
 && tar -xzf /tmp/xls.tgz -C /tmp \
 && cd "/tmp/xls-${XLS_TAG}-linux-x64" \
 && mv codegen_main dslx_fmt dslx_ls interpreter_main ir_converter_main opt_main \
       parse_and_typecheck_dslx_main proto_to_dslx_main prove_quickcheck_main /opt/xls/bin/ \
 && mv xls LICENSE THIRD_PARTY_NOTICES.txt /opt/xls/share/ \
 && rm -rf /tmp/xls.tgz "/tmp/xls-${XLS_TAG}-linux-x64"

FROM base AS solvers
LABEL chiplet-compiler.stage=solvers
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG Z3_VERSION
ARG Z3_SHA256
ARG BITWUZLA_VERSION
ARG BITWUZLA_SHA256
# z3's release build needs glibc 2.39 (= Ubuntu 24.04). The "static" bitwuzla zip links its own
# libraries statically but the binary still needs libgmp and libmpfr (objdump -p: NEEDED).
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends libgmp10 libmpfr6 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/solvers/bin \
 && curl -fsSLo /tmp/z3.zip "https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/z3-${Z3_VERSION}-x64-glibc-2.39.zip" \
 && echo "${Z3_SHA256}  /tmp/z3.zip" | sha256sum -c - \
 && unzip -q /tmp/z3.zip -d /tmp/z3 \
 && cp /tmp/z3/z3-*/bin/z3 /opt/solvers/bin/ \
 && curl -fsSLo /tmp/bitwuzla.zip "https://github.com/bitwuzla/bitwuzla/releases/download/${BITWUZLA_VERSION}/Bitwuzla-Linux-x86_64-static.zip" \
 && echo "${BITWUZLA_SHA256}  /tmp/bitwuzla.zip" | sha256sum -c - \
 && unzip -q /tmp/bitwuzla.zip -d /tmp/bitwuzla \
 && find /tmp/bitwuzla -type f -name bitwuzla -exec cp {} /opt/solvers/bin/ \; \
 && chmod 0755 /opt/solvers/bin/* && /opt/solvers/bin/z3 --version && /opt/solvers/bin/bitwuzla --version \
 && rm -rf /tmp/z3 /tmp/z3.zip /tmp/bitwuzla /tmp/bitwuzla.zip

# ---------------------------------------------------------------------------------------------
# GraMi from the pinned submodule; the patch drops two imports removed in Java 11 (JEP 320).
# ---------------------------------------------------------------------------------------------
FROM base AS grami
LABEL chiplet-compiler.stage=grami
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends openjdk-17-jdk-headless patch \
 && rm -rf /var/lib/apt/lists/*
COPY third_party/grami /src/grami
COPY third_party/patches/grami /src/patches
RUN cd /src/grami && for p in /src/patches/*.patch; do patch -p1 --forward < "$p"; done \
 && for t in GRAMI_DIRECTED_SUBGRAPHS GRAMI_UNDIRECTED_SUBGRAPHS; do \
      mkdir -p "/opt/grami/$t/bin" \
      && find "$t/src" -name '*.java' > /tmp/sources.txt \
      && javac -nowarn -encoding ISO-8859-1 -d "/opt/grami/$t/bin" @/tmp/sources.txt; \
    done \
 && cp LICENCE.txt /opt/grami/

# ---------------------------------------------------------------------------------------------
# FloPoCo at its pinned release tag (the commit is checked), against the apt snapshot's Sollya,
# GMP/MPFR/MPFI, Boost and LAPACK. docker/patches/flopoco/ lets it build without the optional
# PAGSuite and ScaLP libraries (shift-and-add constant multipliers, ILP); the float operators we use
# (FPExp, FPLog, FPPow) do not need them.
# ---------------------------------------------------------------------------------------------
FROM build AS flopoco
LABEL chiplet-compiler.stage=flopoco
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG FLOPOCO_TAG
ARG FLOPOCO_COMMIT
ARG JOBS=16
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends flex pkg-config libboost-random-dev libboost-filesystem-dev \
      libgmp-dev libmpfr-dev libmpfi-dev liblapack-dev libsollya-dev libtbb-dev \
 && rm -rf /var/lib/apt/lists/*
COPY docker/patches/flopoco /src/patches/flopoco
RUN git init -q /src/flopoco && cd /src/flopoco \
 && git remote add origin https://gitlab.com/flopoco/flopoco.git \
 && git fetch -q --depth 1 origin "refs/tags/${FLOPOCO_TAG}" && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${FLOPOCO_COMMIT}" \
 && for p in /src/patches/flopoco/*.patch; do patch -p1 --forward < "$p"; done \
 && cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/flopoco \
      -DFLOPOCO_ENABLE_CCACHE=OFF \
 && ninja -C build -j"${JOBS}" && cmake --install build \
 && cp COPYING /opt/flopoco/ && /opt/flopoco/bin/flopoco FPExp wE=5 wF=10 outputFile=/tmp/t.vhdl >/dev/null \
 && rm -rf /src/flopoco /tmp/t.vhdl

# ---------------------------------------------------------------------------------------------
# Berkeley HardFloat (Chisel, the pinned submodule) and our generator (docker/hardfloat): Chisel from Maven
# Central at the pinned release, built with the pinned sbt on the snapshot's JDK 17; firtool (CIRCT) at
# the release that Chisel version is tested with. The runtime is the jars plus firtool; the final image
# has the JRE.
# ---------------------------------------------------------------------------------------------
FROM base AS hardfloat
LABEL chiplet-compiler.stage=hardfloat
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SBT_VERSION
ARG SBT_SHA256
ARG FIRTOOL_VERSION
ARG FIRTOOL_SHA256
ARG CHISEL_VERSION
ARG SCALA_VERSION
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends openjdk-17-jdk-headless \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /tmp/firtool.tgz "https://github.com/llvm/circt/releases/download/firtool-${FIRTOOL_VERSION}/firrtl-bin-ubuntu-20.04.tar.gz" \
 && echo "${FIRTOOL_SHA256}  /tmp/firtool.tgz" | sha256sum -c - \
 && mkdir -p /opt/firtool && tar -xzf /tmp/firtool.tgz -C /opt/firtool --strip-components=1 \
 && /opt/firtool/bin/firtool --version && rm /tmp/firtool.tgz \
 && curl -fsSLo /tmp/sbt.tgz "https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz" \
 && echo "${SBT_SHA256}  /tmp/sbt.tgz" | sha256sum -c - \
 && tar -xzf /tmp/sbt.tgz -C /opt && rm /tmp/sbt.tgz
COPY docker/hardfloat /src/hfgen
COPY third_party/hardfloat/hardfloat/src/main/scala /src/hfgen/hardfloat-src
COPY third_party/hardfloat/LICENSE /opt/hardfloat/LICENSE
# sbt's caches live under /tmp/sbt and are dropped; the generator's run-time classpath (Chisel, Scala,
# HardFloat and our classes, as jars) goes to /opt/hardfloat/lib.
RUN cd /src/hfgen && mkdir -p project && echo "sbt.version=${SBT_VERSION}" > project/build.properties \
 && export HOME=/tmp/sbt/home CHISEL_VERSION SCALA_VERSION \
 && /opt/sbt/bin/sbt -batch -no-colors -J-Xmx3g -Dsbt.boot.directory=/tmp/sbt/boot \
      -Dsbt.global.base=/tmp/sbt/global -Dsbt.ivy.home=/tmp/sbt/ivy "export Runtime/fullClasspathAsJars" \
      | tee /tmp/sbt.log | tail -n 1 > /tmp/classpath.txt \
 && mkdir -p /opt/hardfloat/lib /opt/hardfloat/bin \
 && tr ':' '\n' < /tmp/classpath.txt | grep '\.jar$' | xargs -I{} cp {} /opt/hardfloat/lib/ \
 && ls /opt/hardfloat/lib | grep -q '^chisel_' && ls /opt/hardfloat/lib | grep -q '^hardfloat-gen' \
 && install -m 0755 hardfloat-gen /opt/hardfloat/bin/hardfloat-gen \
 && PATH=/opt/firtool/bin:$PATH /opt/hardfloat/bin/hardfloat-gen add --exp 5 --sig 10 --module hf_add_f16 --out /tmp/t.v \
 && grep -q '^module hf_add_f16' /tmp/t.v \
 && rm -rf /tmp/sbt /tmp/t.v /tmp/sbt.log /tmp/classpath.txt /opt/sbt /src/hfgen

# ---------------------------------------------------------------------------------------------
# LLVM at Allo's pin (cmake flags from Allo's setup docs) + Allo, editable, into /opt/cc/venv.
# Caps: ninja -j${JOBS}, LLVM_PARALLEL_LINK_JOBS (links are the memory peak), docker --memory.
# ---------------------------------------------------------------------------------------------
FROM build AS llvm-allo
LABEL chiplet-compiler.stage=llvm-allo
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG LLVM_COMMIT
ARG JOBS=16
ARG LINK_JOBS=2
ARG PYPI_EXCLUDE_NEWER
# Resolve PyPI as of one date so rebuilds pick the same versions.
ENV UV_EXCLUDE_NEWER=${PYPI_EXCLUDE_NEWER} UV_PYTHON_DOWNLOADS=never UV_CACHE_DIR=/tmp/uv-cache
RUN git init -q /opt/llvm-project && cd /opt/llvm-project \
 && git remote add origin https://github.com/llvm/llvm-project.git \
 && git fetch -q --depth 1 origin "${LLVM_COMMIT}" && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${LLVM_COMMIT}" && rm -rf .git
RUN uv venv --python /usr/bin/python3.12 /opt/cc/venv \
 && uv pip install --python /opt/cc/venv/bin/python -r /opt/llvm-project/mlir/python/requirements.txt \
      setuptools wheel "nanobind>=2.9,<3.0" \
 && rm -rf "${UV_CACHE_DIR}"
RUN cmake -G Ninja -S /opt/llvm-project/llvm -B /opt/llvm-project/build \
      -DLLVM_ENABLE_PROJECTS="clang;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=/opt/cc/venv/bin/python \
      -DLLVM_PARALLEL_LINK_JOBS="${LINK_JOBS}" \
 && ninja -C /opt/llvm-project/build -j"${JOBS}"
COPY third_party/allo /opt/allo
# Our pinned patches (third_party/patches/allo, docs/third-party.md), applied before Allo's C++ is built.
COPY third_party/patches/allo /src/patches/allo
RUN cd /opt/allo && for p in /src/patches/allo/*.patch; do patch -p1 --forward < "$p"; done
# allo/_mlir is a symlink into mlir/build, so Allo stays an editable install of /opt/allo.
RUN cd /opt/allo && test -L allo/_mlir \
 && LLVM_BUILD_DIR=/opt/llvm-project/build NUM_THREADS="${JOBS}" \
    uv pip install --python /opt/cc/venv/bin/python --no-build-isolation -e . \
 && rm -rf "${UV_CACHE_DIR}" \
 && find /opt/allo/mlir/build \( -name '*.o' -o -name '*.a' \) -delete
# MLIR's Python build links its sources into the package (allo/_mlir/ir.py -> /opt/llvm-project/
# mlir/python/mlir/ir.py, ~200 links); the final image has no LLVM source tree, so copy those
# targets in place. Links inside /opt/allo (e.g. the .so version link) stay links.
RUN find /opt/allo -type l -lname '/opt/llvm-project/*' \
      -exec sh -c 'for l; do t="$(readlink "$l")"; rm "$l"; cp -r "$t" "$l"; done' _ {} + \
 && test -z "$(find /opt/allo -xtype l -lname '/opt/llvm-project/*' -print -quit)" \
 && test -f /opt/allo/allo/_mlir/ir.py && test ! -L /opt/allo/allo/_mlir/ir.py
# What Allo needs from LLVM at run time: the shared libraries it hands to ExecutionEngine
# (allo/backend/llvm.py, simulator.py) plus a few CLI tools. Same path as the build dir.
RUN mkdir -p /opt/llvm-rt/lib /opt/llvm-rt/bin \
 && cp -a /opt/llvm-project/build/lib/*.so* /opt/llvm-rt/lib/ \
 && cp -a /opt/llvm-project/build/bin/llvm-config /opt/llvm-project/build/bin/mlir-opt \
          /opt/llvm-project/build/bin/mlir-translate /opt/llvm-rt/bin/

# ---------------------------------------------------------------------------------------------
# XLS tools that are not in the release tarball, built with Bazel at the same commit.
# Ubuntu 22.04 is XLS's documented build platform (third_party/xls/Dockerfile-ubuntu-22.04).
# ---------------------------------------------------------------------------------------------
FROM ${UBUNTU_2204} AS xls-bazel
LABEL chiplet-compiler.stage=xls-bazel
ARG APT_SNAPSHOT
ARG BAZELISK_VERSION
ARG BAZELISK_SHA256
ARG JOBS=16
ARG BAZEL_RAM_MB=49152
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# apt 2.4 (22.04) has no APT::Snapshot; point the sources at the (https-only) snapshot instead,
# after installing ca-certificates from the regular archive.
RUN test -n "${APT_SNAPSHOT}" \
 && apt-get update --error-on=any && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu/?#https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/#" /etc/apt/sources.list \
 && apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-dev python-is-python3 \
      libtinfo6 build-essential libxml2-dev liblapack-dev libblas-dev gfortran zip unzip default-jdk \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /usr/local/bin/bazel "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" \
 && echo "${BAZELISK_SHA256}  /usr/local/bin/bazel" | sha256sum -c - && chmod 0755 /usr/local/bin/bazel
# Upstream builds as a non-root user (its Dockerfile); hermetic rules_python rejects root.
RUN useradd -m xls
COPY --chown=xls third_party/xls /home/xls/xls
USER xls
WORKDIR /home/xls/xls
# Explicit resources: on cgroups v1 Bazel would size itself from the host (64 CPUs, 502 GiB).
# The build is its own step so its layer (with Bazel's output base) is cached on its own. No
# `bazel shutdown`: PID 1 of a legacy-builder RUN container does not reap the daemonized server,
# so a killed server stays a zombie and shutdown fails ("did not die", exit 36). The server ends
# with the step's container.
RUN bazel --host_jvm_args=-Xmx4g build -c opt --jobs="${JOBS}" \
      --local_resources=cpu="${JOBS}" --local_resources=memory="${BAZEL_RAM_MB}" \
      //xls/contrib/xlscc:xlscc //xls/tools:eval_ir_main //xls/dev_tools:check_ir_equivalence_main
# Our pinned patches (third_party/patches/xls, docs/third-party.md) in their own step: the unpatched
# build above stays cached, so a patch change rebuilds only what it touches. The patches' unit tests
# run here.
COPY --chown=xls third_party/patches/xls /home/xls/patches
RUN for p in /home/xls/patches/*.patch; do patch -p1 --forward < "$p"; done \
 && bazel --host_jvm_args=-Xmx4g test -c opt --jobs="${JOBS}" \
      --local_resources=cpu="${JOBS}" --local_resources=memory="${BAZEL_RAM_MB}" --test_output=errors \
      //xls/contrib/xlscc:xlscc //xls/tools:eval_ir_main //xls/dev_tools:check_ir_equivalence_main \
      //xls/contrib/xlscc/unit_tests:translator_extern_verilog_test
# No Bazel call here (it would start a server over the stale one): bazel-bin is a plain path under
# the output base. set -e: a failing command substitution must stop the step (pipefail alone does
# not). xls_int.h includes "ac_datatypes/include/ac_int.h", but none of our three targets depends
# on that repo, so Bazel never fetched it: download the archive XLS pins (URL and sha256 read from
# its module extension at this commit) and keep its include/ tree.
ARG AC_TYPES_EXT=dependency_support/com_github_hlslibs_ac_types/extension.bzl
RUN set -eu \
 && mkdir -p /home/xls/out/bin /home/xls/out/include/xls/contrib/xlscc /home/xls/out/include/ac_datatypes \
 && cp -L bazel-bin/xls/contrib/xlscc/xlscc bazel-bin/xls/tools/eval_ir_main \
          bazel-bin/xls/dev_tools/check_ir_equivalence_main /home/xls/out/bin/ \
 && cp -rL xls/contrib/xlscc/synth_only /home/xls/out/include/xls/contrib/xlscc/ \
 && url="$(sed -n 's/.*urls = \["\([^"]*\)"\].*/\1/p' "${AC_TYPES_EXT}")" \
 && sha="$(sed -n 's/.*sha256 = "\([0-9a-f]*\)".*/\1/p' "${AC_TYPES_EXT}")" \
 && prefix="$(sed -n 's/.*strip_prefix = "\([^"]*\)".*/\1/p' "${AC_TYPES_EXT}")" \
 && echo "ac_types: ${url} ${sha} ${prefix}" && test -n "${url}" && test -n "${sha}" && test -n "${prefix}" \
 && curl -fsSLo /tmp/ac_types.tgz "${url}" && echo "${sha}  /tmp/ac_types.tgz" | sha256sum -c - \
 && tar -xzf /tmp/ac_types.tgz -C /tmp \
 && cp -r "/tmp/${prefix}/include" /home/xls/out/include/ac_datatypes/ \
 && test -f /home/xls/out/include/ac_datatypes/include/ac_int.h && rm -rf /tmp/ac_types.tgz "/tmp/${prefix}"

# ---------------------------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------------------------
# The project's locked JAX stack (pyproject.toml + uv.lock: jax, flax, haiku, keras, ...) for the export
# pass, in its own venv: Allo's /opt/cc/venv pins its own numpy and MLIR bindings.
FROM base AS jax
LABEL chiplet-compiler.stage=jax
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV UV_PYTHON_DOWNLOADS=never UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/opt/cc/jax-venv \
    KERAS_BACKEND=jax
COPY pyproject.toml uv.lock /src/jax/
RUN cd /src/jax && uv sync --frozen --no-install-project --no-dev --python /usr/bin/python3.12 \
 && /opt/cc/jax-venv/bin/python -c "import jax, flax, haiku, keras, flatbuffers; print('jax', jax.__version__)" \
 && rm -rf /tmp/uv-cache

FROM base AS final
LABEL chiplet-compiler.stage=final
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Verilator compiles C++ models at run time, so the final image keeps a compiler.
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends build-essential openjdk-17-jre-headless \
      yosys iverilog verilator libtinfo6 zlib1g libgmp10 libmpfr6 \
 && rm -rf /var/lib/apt/lists/*
# GHDL and the ghdl-yosys plugin (VHDL, e.g. FloPoCo's operators, into Yosys; the plugin reads the
# ghdl-gcc library tree), and the libraries the FloPoCo binary links (ldd).
RUN apt-get update --error-on=any \
 && apt-get install -y --no-install-recommends ghdl ghdl-gcc yosys-plugin-ghdl \
      libfplll8t64 libgmpxx4ldbl libmpfi0 libsollya8 libtbb12 libxml2 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=xls-release /opt/xls /opt/xls
COPY --from=xls-bazel /home/xls/out/bin/ /opt/xls/bin/
COPY --from=xls-bazel /home/xls/out/include /opt/xls/include
COPY --from=solvers /opt/solvers /opt/solvers
COPY --from=grami /opt/grami /opt/grami
COPY --from=flopoco /opt/flopoco /opt/flopoco
COPY --from=hardfloat /opt/hardfloat /opt/hardfloat
COPY --from=hardfloat /opt/firtool /opt/firtool
# XLS's DSLX float libraries (sine/cosine, exponential and their imports) from the pinned XLS commit;
# imported as `third_party.xls_go_math.<file>` with --dslx_path=$XLS_DSLX_PATH.
COPY third_party/xls/third_party/xls_go_math/*.x third_party/xls/third_party/xls_go_math/LICENSE \
     /opt/xls/share/third_party/xls_go_math/
COPY third_party/xls/third_party/xls_berkeley_softfloat/*.x third_party/xls/third_party/xls_berkeley_softfloat/LICENSE \
     /opt/xls/share/third_party/xls_berkeley_softfloat/
COPY --from=llvm-allo /opt/llvm-rt /opt/llvm-project/build
COPY --from=llvm-allo /opt/allo /opt/allo
COPY --from=llvm-allo /opt/cc/venv /opt/cc/venv
COPY --from=jax /opt/cc/jax-venv /opt/cc/jax-venv
COPY docker/cc-probe.py /opt/cc/bin/cc-probe
COPY docker/selftest.sh /opt/cc/bin/cc-selftest
# DockerRunner runs as the host uid with HOME=/tmp/home (docs/tools/docker_resources.md §3).
RUN mkdir -p /tmp/home && chmod 1777 /tmp/home && chmod 0755 /opt/cc/bin/cc-probe /opt/cc/bin/cc-selftest
ENV PATH=/opt/cc/bin:/opt/cc/venv/bin:/opt/xls/bin:/opt/solvers/bin:/opt/flopoco/bin:/opt/hardfloat/bin:/opt/firtool/bin:/opt/llvm-project/build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/cc/venv \
    LLVM_BUILD_DIR=/opt/llvm-project/build \
    XLS_STDLIB=/opt/xls/share/xls/dslx/stdlib \
    XLS_DSLX_PATH=/opt/xls/share \
    XLSCC_INCLUDE_DIRS=/opt/xls/include,/opt/xls/include/xls/contrib/xlscc/synth_only \
    XLSCC_DEFINES=__SYNTHESIS__,__xlscc__,__AC_OVERRIDE_OVF_UPDATE_BODY=,__AC_OVERRIDE_OVF_UPDATE2_BODY= \
    GRAMI_DIRECTED_CP=/opt/grami/GRAMI_DIRECTED_SUBGRAPHS/bin \
    GRAMI_UNDIRECTED_CP=/opt/grami/GRAMI_UNDIRECTED_SUBGRAPHS/bin \
    HOME=/tmp/home
ARG XLS_TAG
ARG XLS_COMMIT
ARG ALLO_COMMIT
ARG LLVM_COMMIT
ARG GRAMI_COMMIT
ARG FLOPOCO_COMMIT
ARG HARDFLOAT_COMMIT
ARG CHISEL_VERSION
ARG FIRTOOL_VERSION
ARG SBT_VERSION
ARG Z3_VERSION
ARG BITWUZLA_VERSION
ARG UV_VERSION
ARG BAZELISK_VERSION
ARG APT_SNAPSHOT
ARG PYPI_EXCLUDE_NEWER
ARG IMAGE_TAG
# The unit tests of the Allo patches (third_party/patches/allo), now that xlscc and codegen are on PATH.
RUN cd /tmp && python3 -m pytest -q -p no:cacheprovider /opt/allo/tests/test_xls_extensions.py
# Fail the build if any tool is broken, then record versions + pins for `chipc doctor`.
RUN cc-selftest \
 && cc-probe --write /opt/cc/versions.json \
      --pin image_tag="${IMAGE_TAG}" --pin xls_tag="${XLS_TAG}" --pin xls_commit="${XLS_COMMIT}" \
      --pin allo_commit="${ALLO_COMMIT}" --pin llvm_commit="${LLVM_COMMIT}" --pin grami_commit="${GRAMI_COMMIT}" \
      --pin flopoco_commit="${FLOPOCO_COMMIT}" --pin hardfloat_commit="${HARDFLOAT_COMMIT}" \
      --pin chisel_version="${CHISEL_VERSION}" --pin firtool_version="${FIRTOOL_VERSION}" --pin sbt_version="${SBT_VERSION}" \
      --pin z3_version="${Z3_VERSION}" --pin bitwuzla_version="${BITWUZLA_VERSION}" \
      --pin uv_version="${UV_VERSION}" --pin bazelisk_version="${BAZELISK_VERSION}" \
      --pin apt_snapshot="${APT_SNAPSHOT}" --pin pypi_exclude_newer="${PYPI_EXCLUDE_NEWER}" \
 && cat /opt/cc/versions.json
WORKDIR /tmp/home
