#!/bin/bash
#
# Build a ds4 image from a recipe file.  Mirrors the ergonomics of
# scitrera/cuda-containers' build-image.sh, trimmed to this repo's single
# Dockerfile.
#
# The GPU architecture is NOT part of the recipe: one recipe pins one upstream
# ds4 commit, and that commit is built once per architecture (see README).
# The arch is chosen here, defaulting from the host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="${SCRIPT_DIR}/recipes"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <recipe>

Build the ds4 CUDA image from a recipe file.

ARGUMENTS:
  recipe              Recipe name (without .recipe) or a path to a recipe file

OPTIONS:
  -h, --help          Show this help
  -l, --list          List available recipes
  -n, --dry-run       Print the resolved build configuration and exit
      --arch ARCH     GPU arch: sm_121 (GB10 / DGX Spark) or sm_120 (RTX /
                      PRO Blackwell).  Default: sm_121 on aarch64, sm_120 on
                      x86_64.
      --cpu-flag F    Override the compiler CPU baseline (see README).
                      Default: -mcpu=neoverse-v2 on aarch64,
                               -march=x86-64-v3 on x86_64.
      --tag TAG       Full image tag to produce.  Default is derived from
                      IMAGE_REPO, DS4_VERSION, the arch and the CUDA version.
      --no-cache      Build without the Docker layer cache
      --push          Push instead of loading into the local image store
EOF
}

list_recipes() {
    echo "Available recipes in ${RECIPES_DIR}:"
    echo ""
    local recipe name ref
    for recipe in "${RECIPES_DIR}"/*.recipe; do
        [[ -f "$recipe" ]] || continue
        name=$(basename "$recipe" .recipe)
        ref=$(grep -E '^DS4_REF=' "$recipe" | head -1 | cut -d= -f2- || echo "")
        printf "  %-32s -> %s\n" "$name" "${ref:0:12}"
    done
}

# Keys a recipe or the shared parameters file may set.
KNOWN_KEYS=(
    CUDA_VERSION UBUNTU_VERSION BUILD_JOBS
    DS4_REPO DS4_REF DS4_VERSION
    IMAGE_REPO
)

declare -A CONFIG=()

load_kv_file() {
    local file="$1" key value known
    [[ -f "$file" ]] || return 0
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        key=$(echo "$key" | xargs)
        [[ -z "$key" ]] && continue
        value=$(echo "$value" | xargs)
        for known in "${KNOWN_KEYS[@]}"; do
            if [[ "$key" == "$known" ]]; then
                CONFIG["$key"]="$value"
                break
            fi
        done
    done < "$file"
}

DRY_RUN=false
NO_CACHE=false
PUSH=false
RECIPE_ARG=""
ARCH=""
CPU_FLAG=""
IMAGE_TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -l|--list) list_recipes; exit 0 ;;
        -n|--dry-run) DRY_RUN=true; shift ;;
        --no-cache) NO_CACHE=true; shift ;;
        --push) PUSH=true; shift ;;
        --arch) [[ $# -ge 2 ]] || { echo "Error: --arch requires a value" >&2; exit 1; }; ARCH="$2"; shift 2 ;;
        --cpu-flag) [[ $# -ge 2 ]] || { echo "Error: --cpu-flag requires a value" >&2; exit 1; }; CPU_FLAG="$2"; shift 2 ;;
        --tag) [[ $# -ge 2 ]] || { echo "Error: --tag requires a value" >&2; exit 1; }; IMAGE_TAG="$2"; shift 2 ;;
        -*) echo "Error: unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)
            [[ -z "$RECIPE_ARG" ]] || { echo "Error: only one recipe may be given" >&2; exit 1; }
            RECIPE_ARG="$1"; shift ;;
    esac
done

if [[ -z "$RECIPE_ARG" ]]; then
    echo "Error: no recipe specified." >&2
    echo "" >&2
    usage >&2
    exit 1
fi

if [[ -f "$RECIPE_ARG" ]]; then
    RECIPE_FILE="$RECIPE_ARG"
elif [[ -f "${RECIPES_DIR}/${RECIPE_ARG}.recipe" ]]; then
    RECIPE_FILE="${RECIPES_DIR}/${RECIPE_ARG}.recipe"
else
    echo "Error: recipe not found: ${RECIPE_ARG}" >&2
    exit 1
fi

# Shared parameters first, recipe second -- the recipe wins.
load_kv_file "${SCRIPT_DIR}/ds4.parameters"
load_kv_file "$RECIPE_FILE"

for required in DS4_REPO DS4_REF DS4_VERSION CUDA_VERSION IMAGE_REPO; do
    [[ -n "${CONFIG[$required]:-}" ]] || { echo "Error: recipe must define ${required}" >&2; exit 1; }
done

HOST_ARCH="$(uname -m)"
if [[ -z "$ARCH" ]]; then
    case "$HOST_ARCH" in
        aarch64|arm64) ARCH="sm_121" ;;
        x86_64|amd64)  ARCH="sm_120" ;;
        *) echo "Error: unsupported host arch ${HOST_ARCH}; pass --arch explicitly" >&2; exit 1 ;;
    esac
fi

case "$ARCH" in
    sm_121|sm_120) ;;
    *) echo "Error: --arch must be sm_121 or sm_120 (got ${ARCH})" >&2; exit 1 ;;
esac

if [[ -z "$CPU_FLAG" ]]; then
    case "$HOST_ARCH" in
        aarch64|arm64) CPU_FLAG="-mcpu=neoverse-v2" ;;
        *)             CPU_FLAG="-march=x86-64-v3" ;;
    esac
fi

CUDA_SHORT=$(echo "${CONFIG[CUDA_VERSION]}" | cut -d. -f1,2 | tr -d '.')
ARCH_SUFFIX="${ARCH/sm_/sm}a"   # sm_121 -> sm121a

if [[ -z "$IMAGE_TAG" ]]; then
    IMAGE_TAG="${CONFIG[IMAGE_REPO]}:${CONFIG[DS4_VERSION]}-${ARCH_SUFFIX}-cu${CUDA_SHORT}"
fi

cat <<EOF
========================================
ds4 build configuration
========================================

Recipe:      ${RECIPE_FILE}
ds4 repo:    ${CONFIG[DS4_REPO]}
ds4 ref:     ${CONFIG[DS4_REF]}
Version:     ${CONFIG[DS4_VERSION]}
CUDA:        ${CONFIG[CUDA_VERSION]} (cu${CUDA_SHORT})
Ubuntu:      ${CONFIG[UBUNTU_VERSION]:-24.04}
GPU arch:    ${ARCH} -> ${ARCH_SUFFIX}
CPU flag:    ${CPU_FLAG}
Build jobs:  ${CONFIG[BUILD_JOBS]:-4}
Image tag:   ${IMAGE_TAG}

========================================
EOF

if $DRY_RUN; then
    echo "[DRY RUN] Not building."
    exit 0
fi

BUILD_CMD=(docker buildx build)
$NO_CACHE && BUILD_CMD+=(--no-cache)
BUILD_CMD+=(
    -f "${SCRIPT_DIR}/Dockerfile"
    --target runtime
    --build-arg "CUDA_VERSION=${CONFIG[CUDA_VERSION]}"
    --build-arg "UBUNTU_VERSION=${CONFIG[UBUNTU_VERSION]:-24.04}"
    --build-arg "BUILD_JOBS=${CONFIG[BUILD_JOBS]:-4}"
    --build-arg "DS4_REPO=${CONFIG[DS4_REPO]}"
    --build-arg "DS4_REF=${CONFIG[DS4_REF]}"
    --build-arg "DS4_CUDA_ARCH=${ARCH}"
    --build-arg "DS4_CPU_FLAG=${CPU_FLAG}"
    --label "maintainer=spark-arena <open-source-team@scitrera.com>"
    --label "dev.scitrera.ds4_version=${CONFIG[DS4_VERSION]}"
    --label "dev.scitrera.ds4_ref=${CONFIG[DS4_REF]}"
    --label "dev.scitrera.cuda_version=${CONFIG[CUDA_VERSION]}"
    --label "dev.scitrera.cuda_arch=${ARCH}"
    -t "$IMAGE_TAG"
)

if $PUSH; then
    BUILD_CMD+=(--push)
else
    BUILD_CMD+=(--load)
fi

BUILD_CMD+=("${SCRIPT_DIR}")

echo "=== Building ${IMAGE_TAG} ==="
"${BUILD_CMD[@]}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE_TAG}"
