# CUDA container build for ds4 / DwarfStar (https://github.com/antirez/ds4)
#
# ds4 is a self-contained native C/CUDA inference engine for DeepSeek V4
# Flash/PRO and GLM 5.2 GGUFs.  It does not link GGML and has no Python
# runtime, so the shipped image is just the binaries plus the CUDA runtime.
#
# Three things about ds4's Makefile shape this build and are easy to get
# wrong -- see README.md "Build constraints" for the long version:
#
#   1. CFLAGS default to ``-march=native`` / ``-mcpu=native``.  Left alone,
#      a CI build bakes the *runner's* ISA into the binary.  DS4_CPU_FLAG
#      overrides ``NATIVE_CPU_FLAG`` with an explicit baseline.
#   2. ``CUDA_ARCH`` produces a SINGLE ``-gencode`` with no ``code=compute_*``
#      PTX fallback, and ``-DDS4_CUDA_HAVE_MXF4=1`` is only valid for
#      sm_120a/sm_121a.  One image == one GPU architecture; there is no
#      fatbin that spans Blackwell and Hopper.
#   3. RDMA is ``dlopen``'d behind ``#if __has_include(<infiniband/verbs.h>)``
#      (ds4_tp.c).  The headers must be present at BUILD time or tensor-parallel
#      RDMA transport is silently compiled out, and libibverbs must be present
#      at RUN time or the dlopen fails.

ARG CUDA_VERSION="13.1.1"
ARG UBUNTU_VERSION="24.04"

# ---------------------------------------------------------------------------
# Stage 1: build ds4 from source
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG BUILD_JOBS=4
ARG DS4_REPO="https://github.com/antirez/ds4.git"
ARG DS4_REF="main"
# nvcc -arch value.  sm_121 -> GB10 (DGX Spark), sm_120 -> RTX/PRO Blackwell.
# The Makefile maps both onto their ``a`` variants and enables MXFP4.
ARG DS4_CUDA_ARCH="sm_121"
# Overrides the Makefile's NATIVE_CPU_FLAG (default: -march=native).
ARG DS4_CPU_FLAG="-mcpu=neoverse-v2"

ENV DEBIAN_FRONTEND=noninteractive \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=10G \
    CCACHE_COMPRESS=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        git \
        libibverbs-dev \
        librdmacm-dev \
        rdma-core \
    # ccache ships masquerade symlinks for cc/gcc/g++ but not nvcc.  ds4's
    # heaviest translation units are the vendored cuda/mmq/ templates, so
    # caching nvcc is what makes a rebuild tolerable.
    && ln -sf /usr/bin/ccache /usr/lib/ccache/nvcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# DS4_REF may be a 40-hex commit SHA (full clone + checkout, what CI pins) or
# a branch/tag name (shallow clone).  Upstream publishes neither tags nor
# releases, so CI always pins a SHA.
RUN if echo "${DS4_REF}" | grep -qE '^[0-9a-fA-F]{40}$'; then \
        git clone "${DS4_REPO}" ds4 && \
        cd ds4 && \
        git checkout "${DS4_REF}"; \
    else \
        git clone -b "${DS4_REF}" --depth 1 "${DS4_REPO}" ds4; \
    fi

WORKDIR /build/ds4

# ds4_tp.c gates the verbs include on ``__has_include(<infiniband/verbs.h>)``,
# so a missing libibverbs-dev degrades to a working TCP-only binary rather
# than a build error.  Tensor-parallel over CX7 is a headline feature on a
# DGX Spark cluster, so assert the precondition instead of discovering it in
# production.
RUN test -f /usr/include/infiniband/verbs.h \
    || (echo "ERROR: infiniband/verbs.h missing; ds4 would build without RDMA transport" >&2 && exit 1)

# Build only the server plus the two diagnostics binaries -- ``make cuda-spark``
# additionally builds ds4-eval and ds4-agent and passes ``-B`` (force rebuild),
# neither of which we want in an image.
#
# CC/NVCC resolve through /usr/lib/ccache.  DS4_LINK is pinned to the *real*
# nvcc: it is a link command, not a compile, and there is nothing for ccache
# to do but add a fallback hop.  It is single-quoted so $(NVCCFLAGS) is
# expanded by make (lazily, after CUDA_ARCH is applied), not by the shell.
#
# CFLAGS is deliberately NOT overridden: the Makefile does ``CFLAGS +=`` for
# the Linux-only flags, and make ignores ``+=`` against a command-line
# variable, so setting it here would silently drop -D_GNU_SOURCE and
# -fno-finite-math-only.  Only NATIVE_CPU_FLAG is overridden.
RUN --mount=type=cache,id=ccache-ds4,target=/root/.ccache \
    make -j"${BUILD_JOBS}" ds4-server ds4 ds4-bench \
        CC=cc \
        NVCC=nvcc \
        DS4_LINK='/usr/local/cuda/bin/nvcc $(NVCCFLAGS)' \
        CUDA_ARCH="${DS4_CUDA_ARCH}" \
        NATIVE_CPU_FLAG="${DS4_CPU_FLAG}" \
    && ccache --show-stats

RUN mkdir -p /opt/ds4/bin && cp ds4-server ds4 ds4-bench /opt/ds4/bin/

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        # ds4_tp.c dlopen()s libibverbs at run time for --transport rdma.
        libibverbs1 \
        librdmacm1 \
        rdma-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/ds4/bin/ /usr/local/bin/

# ds4-server's default model path is the relative ``ds4flash.gguf``; /data is
# the conventional mount point for a weights directory.
WORKDIR /data

ENV PATH=/usr/local/bin:$PATH

# No ENTRYPOINT, deliberately.  sparkrun launches workloads as CMD arguments
# (``docker run <image> bash -c <cmd>``); a consuming ENTRYPOINT would parse
# those as its own flags and the workload would never start.
CMD ["ds4-server", "--help"]
