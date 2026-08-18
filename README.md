# dgx-ds4 — CUDA containers for ds4 / DwarfStar

Prebuilt CUDA container images for [ds4 (DwarfStar)](https://github.com/antirez/ds4),
Salvatore Sanfilippo's native C/CUDA inference engine for **DeepSeek V4 Flash/PRO**
and **GLM 5.2** GGUFs.

Images are published to **`ghcr.io/spark-arena/dgx-ds4`** and are what the
[sparkrun](https://github.com/spark-arena/sparkrun) `ds4` runtime launches by
default.

ds4 is self-contained — no PyTorch, no Python, no GGML linkage — so the runtime
image is just the binaries plus the CUDA runtime.

---

## Quick start

```bash
docker run --rm -it \
  --gpus all \
  --network host --ipc=host \
  -v ~/models:/data \
  ghcr.io/spark-arena/dgx-ds4:latest \
  ds4-server \
    --cuda \
    -m /data/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
    -c 131072 \
    --host 0.0.0.0 --port 8000
```

The server exposes OpenAI-compatible `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, an OpenAI `/v1/responses` endpoint and an Anthropic-compatible
`/v1/messages`. There is **no** `/health` endpoint and **no** authentication —
`GET /v1/models` is the readiness probe.

Model weights come from
[`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf).
ds4 is deliberately not a general GGUF loader; arbitrary GGUFs will not load.

## Tags

| Tag | Meaning |
|---|---|
| `latest` | Multi-arch. The latest tracked upstream `main` commit. |
| `<version>-cu131` | Multi-arch, immutable. `<version>` is `<commit-date>-<sha7>`. **Pin this** for anything reproducible. |
| `<version>-sm121a-cu131` | arm64 only, GB10 / DGX Spark. |
| `<version>-sm120a-cu131` | amd64 only, RTX / PRO Blackwell. |
| `sm121a` / `sm120a` | Moving per-architecture tags for the latest build. |

Upstream publishes **no tags and no releases**, so the version axis is the
upstream commit's *committer date* plus a short SHA — stable and reproducible,
unlike a build date. The full 40-hex commit is pinned in `recipes/` and
recorded in the image as `dev.scitrera.ds4_ref`.

```bash
docker inspect ghcr.io/spark-arena/dgx-ds4:latest --format '{{json .Config.Labels}}' | jq
```

## Build constraints

Four properties of ds4's Makefile drive this repo's shape. All four fail
*silently* — a wrong build produces a working-looking image.

**1. One image is one GPU architecture.** `CUDA_ARCH=sm_121` expands to a
single `-gencode arch=compute_121a,code=sm_121a`. There is no `code=compute_*`
entry, so there is **no PTX fallback** and no forward-compatible JIT. On top of
that, `-DDS4_CUDA_HAVE_MXF4=1` is a compile-time global set only for
`sm_120a`/`sm_121a`, so a fatbin spanning Blackwell and Hopper cannot be
produced without patching upstream.

The multi-arch `latest` and `<version>-cu131` manifests therefore pair **arm64 = `sm_121a`
(GB10)** with **amd64 = `sm_120a` (RTX / PRO Blackwell)** — both Blackwell,
both MXFP4-capable, so the two platforms have equivalent capability. Ada
(`sm_89`), Hopper (`sm_90`) and Grace-Hopper are out of scope; add a matrix
entry and a separate tag name if they are ever needed.

**2. `-march=native` / `-mcpu=native` is the Makefile default.** Building in CI
without overriding `NATIVE_CPU_FLAG` bakes the *runner's* ISA into the binary —
an illegal-instruction crash on x86, a silent de-optimization on arm64. The
builds pin explicit baselines: `-mcpu=neoverse-v2` (armv9, covers GB10 and
Grace) and `-march=x86-64-v3` (AVX2/FMA, covers everything that can host a
Blackwell card).

`CFLAGS` itself is deliberately **not** overridden — the Makefile appends
`-D_GNU_SOURCE -fno-finite-math-only` with `+=`, and make ignores `+=` against
a command-line variable, so setting `CFLAGS` would silently drop them.

**3. RDMA is `dlopen`'d behind `__has_include`.** `ds4_tp.c` gates the verbs
include on the header being present, so a builder without `libibverbs-dev`
produces a perfectly functional TCP-only binary with `--transport rdma` quietly
gone. The Dockerfile asserts `/usr/include/infiniband/verbs.h` exists before
compiling, and the runtime stage installs `libibverbs1` / `rdma-core` so the
`dlopen` resolves. This matters: two-machine tensor parallelism over ConnectX-7
is the reason to run ds4 on a Spark cluster at all.

**4. There is no ENTRYPOINT.** sparkrun launches workloads as CMD *arguments*
(`docker run <image> bash -c <cmd>`), so a consuming ENTRYPOINT would parse
them as its own flags and the workload would never start. `CMD` is set to
`ds4-server --help` purely so a bare `docker run` is informative.

## Repository layout

```
Dockerfile                    2-stage CUDA build (devel builder -> runtime)
ds4.parameters                shared build parameters (CUDA version, jobs, repo)
build-image.sh                local/manual builds from a recipe
recipes/ds4-<version>.recipe  the currently-tracked upstream commit
recipes/archive/              superseded recipes
.github/workflows/build.yml   commit tracker -> per-arch build -> manifest
```

A **recipe** pins one upstream commit. The GPU architecture is not part of it:
one commit is built once per architecture, so arch lives in the CI matrix (and
in `build-image.sh --arch`).

## Building locally

On a DGX Spark (defaults to `sm_121` / `-mcpu=neoverse-v2` from `uname -m`):

```bash
./build-image.sh ds4-20260809-84cc882
./build-image.sh --dry-run ds4-20260809-84cc882      # show resolved config
./build-image.sh --list
```

On an x86 Blackwell workstation it defaults to `sm_120` / `-march=x86-64-v3`.
Override either explicitly:

```bash
./build-image.sh --arch sm_120 --cpu-flag '-march=x86-64-v3' ds4-20260809-84cc882
```

Expect a long first build — the vendored `cuda/mmq/` templates dominate. ccache
is wired through a buildx cache mount (including a masquerade symlink for
`nvcc`, which ccache does not install itself), so rebuilds of the same commit
are fast. `BUILD_JOBS` defaults to 4; lower it first if nvcc is OOM-killed.

## CI

`build.yml` runs twice daily:

1. **check** — resolve `antirez/ds4` HEAD via the GitHub API, compare the SHA
   against the pinned recipe.
2. **build** — a 2-entry matrix on *native* runners (`ubuntu-24.04-arm` and
   `ubuntu-24.04`, no QEMU), pushing immutable per-arch tags.
3. **manifest** — `buildx imagetools create` combines them into the multi-arch
   `<version>-cu131` and `latest`.
4. **commit-recipe** — write the new recipe, archive the old one, commit.
   Skipped for a forced `workflow_dispatch` ref so an ad-hoc build of a side
   branch never repoints the tracked commit.

`workflow_dispatch` accepts any branch, tag or SHA — useful for upstream's
`glm5.2`, `ds4f-mxfp4` and `responses-api` branches.

## Licence

The build tooling in this repository is MIT (see `LICENSE`). ds4 itself is MIT,
Copyright (c) 2026 The ds4.c authors and (c) 2023-2026 The ggml authors — see
[the upstream LICENSE](https://github.com/antirez/ds4/blob/main/LICENSE).

This project is not affiliated with NVIDIA, DeepSeek, or the ds4 authors.
