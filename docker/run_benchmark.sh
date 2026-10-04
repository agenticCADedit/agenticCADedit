#!/usr/bin/env bash
#
# AgenticCADedit - benchmark runner
#
# Serves an open-weight model with vLLM (optional) and runs one of the two
# benchmark arms inside the project Docker image.
#
# Usage:
#   docker/run_benchmark.sh {script|mcp}
#
# Everything is configured through environment variables; see README.

set -euo pipefail

# ---------------------------------------------------------------------------
# Arm selection
# ---------------------------------------------------------------------------
ARM="${1:-}"

MODEL_TAG="${MODEL_TAG:-my-model}"

case "${ARM}" in
    script)
        HARNESS="src/harnesses/cadquery_script.py"
        USER_ID="${USER_ID:-${MODEL_TAG}_cadquery-script}"
        ;;
    mcp)
        HARNESS="src/harnesses/cadquery_mcp/server.py"
        USER_ID="${USER_ID:-${MODEL_TAG}_cadquery-mcp}"
        ;;
    *)
        echo "Usage: $0 {script|mcp}" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-agentic-cad-edit:latest}"
CONTAINER_PROJECT_DIR="${CONTAINER_PROJECT_DIR:-/workspace/AgenticCADedit}"

# Host directory that contains the benchmark dataset, e.g.
#   $DATA_DIR/edit_192_external/parquets
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing the benchmark dataset}"
DATASET_NAME="${DATASET_NAME:-edit_192_external}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/data}"

CONFIG="${CONFIG:-src/config/${DATASET_NAME}.json}"
INPUT_SUBDIR="${INPUT_SUBDIR:-parquets}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-model_edits}"
N_ROWS="${N_ROWS:-999999}"
REQUIRED_EXTENSIONS="${REQUIRED_EXTENSIONS:-step}"

# vLLM serving (set SKIP_VLLM=1 to use an already running / remote endpoint)
SKIP_VLLM="${SKIP_VLLM:-0}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
MODEL_REVISION="${MODEL_REVISION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_IMAGES="${MAX_IMAGES:-256}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_READY_ATTEMPTS="${VLLM_READY_ATTEMPTS:-360}"
VLLM_READY_INTERVAL="${VLLM_READY_INTERVAL:-10}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"

GPUS="${GPUS:-all}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

# Paths as seen from inside the container
DATASET_DIR_HOST="${DATA_DIR}/${DATASET_NAME}"
DATASET_DIR_CONTAINER="${CONTAINER_DATA_DIR}/${DATASET_NAME}"
INPUT_CONTAINER="${DATASET_DIR_CONTAINER}/${INPUT_SUBDIR}"
OUTPUT_CONTAINER="${DATASET_DIR_CONTAINER}/${OUTPUT_SUBDIR}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
command -v "${DOCKER_BIN}" >/dev/null 2>&1 || {
    echo "ERROR: '${DOCKER_BIN}' not found in PATH." >&2
    exit 1
}

if [[ ! -d "${DATASET_DIR_HOST}/${INPUT_SUBDIR}" ]]; then
    cat >&2 <<EOF
ERROR: benchmark input not found at
       ${DATASET_DIR_HOST}/${INPUT_SUBDIR}

Download the dataset and place it so that this path exists, then set
DATA_DIR accordingly. See the README section "Download and visualise data".
EOF
    exit 1
fi

mkdir -p "${DATASET_DIR_HOST}/${OUTPUT_SUBDIR}" "${HF_CACHE}"

echo "=== AgenticCADedit benchmark ==="
echo "    host              : $(hostname)"
echo "    arm               : ${ARM}"
echo "    harness           : ${HARNESS}"
echo "    userId            : ${USER_ID}"
echo "    project           : ${PROJECT_DIR}"
echo "    dataset (host)    : ${DATASET_DIR_HOST}"
echo "    image             : ${IMAGE}"
echo "    rows              : ${N_ROWS}"
if [[ "${SKIP_VLLM}" == "1" ]]; then
    echo "    endpoint          : ${VLLM_BASE_URL:-<from config / provider keys>}"
else
    echo "    model             : ${MODEL}"
    echo "    vLLM port         : ${VLLM_PORT}"
fi
echo

# ---------------------------------------------------------------------------
# Optional: start vLLM
# ---------------------------------------------------------------------------
VLLM_CONTAINER=""

cleanup() {
    if [[ -n "${VLLM_CONTAINER}" ]]; then
        echo "Stopping vLLM container ${VLLM_CONTAINER}..."
        "${DOCKER_BIN}" rm -f "${VLLM_CONTAINER}" >/dev/null 2>&1 || true
        VLLM_CONTAINER=""
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${SKIP_VLLM}" != "1" ]]; then
    VLLM_CONTAINER="agentic-cad-vllm-${ARM}-$$"

    VLLM_ARGS=(
        --model "${MODEL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --tensor-parallel-size "${TENSOR_PARALLEL}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --dtype auto
        --max-num-seqs 1
        --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}"
    )
    [[ -n "${MODEL_REVISION}" ]] && VLLM_ARGS+=(--revision "${MODEL_REVISION}")

    # Tool calling is required for the MCP arm.
    if [[ "${ARM}" == "mcp" ]]; then
        VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
    fi
    [[ -n "${REASONING_PARSER}" ]] && VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")

    VLLM_ENV=()
    [[ -n "${HF_TOKEN}" ]] && VLLM_ENV+=(-e "HF_TOKEN=${HF_TOKEN}")

    echo "1. Starting vLLM server (${MODEL})..."
    "${DOCKER_BIN}" run -d --rm \
        --name "${VLLM_CONTAINER}" \
        --gpus "${GPUS}" \
        --ipc=host \
        -p "${VLLM_PORT}:8000" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        "${VLLM_ENV[@]+"${VLLM_ENV[@]}"}" \
        "${VLLM_IMAGE}" \
        "${VLLM_ARGS[@]}" >/dev/null

    echo "2. Waiting for vLLM (the initial model download can take a while)..."
    READY=false
    for ((i = 1; i <= VLLM_READY_ATTEMPTS; i++)); do
        if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            READY=true
            break
        fi
        if ! "${DOCKER_BIN}" ps -q --filter "name=^${VLLM_CONTAINER}$" | grep -q .; then
            echo "ERROR: vLLM exited before becoming ready. Logs:" >&2
            "${DOCKER_BIN}" logs "${VLLM_CONTAINER}" 2>&1 | tail -n 50 >&2 || true
            exit 1
        fi
        echo "   attempt ${i}/${VLLM_READY_ATTEMPTS} - retrying in ${VLLM_READY_INTERVAL}s..."
        sleep "${VLLM_READY_INTERVAL}"
    done

    [[ "${READY}" == true ]] || {
        echo "ERROR: vLLM did not become ready within the configured timeout." >&2
        exit 1
    }
    echo "   vLLM ready at http://localhost:${VLLM_PORT}/v1"
    VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
fi

# ---------------------------------------------------------------------------
# Run the benchmark
# ---------------------------------------------------------------------------
echo "3. Running the ${ARM} benchmark..."

BENCH_ENV=()
[[ -n "${VLLM_BASE_URL:-}"      ]] && BENCH_ENV+=(-e "VLLM_BASE_URL=${VLLM_BASE_URL}")
[[ -n "${OPENAI_API_KEY:-}"     ]] && BENCH_ENV+=(-e "OPENAI_API_KEY=${OPENAI_API_KEY}")
[[ -n "${ANTHROPIC_API_KEY:-}"  ]] && BENCH_ENV+=(-e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
[[ -n "${GOOGLE_API_KEY:-}"     ]] && BENCH_ENV+=(-e "GOOGLE_API_KEY=${GOOGLE_API_KEY}")

set +e
"${DOCKER_BIN}" run --rm \
    --network=host \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    "${BENCH_ENV[@]+"${BENCH_ENV[@]}"}" \
    -v "${PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    -v "${DATA_DIR}:${CONTAINER_DATA_DIR}" \
    -w "${CONTAINER_PROJECT_DIR}" \
    "${IMAGE}" \
    xvfb-run -a -s "-screen 0 1920x1080x24" \
    python src/scripts_benchmark_inference/run_harness.py \
        --config "${CONFIG}" \
        --input "${INPUT_CONTAINER}" \
        --output_dir "${OUTPUT_CONTAINER}" \
        --harness "${HARNESS}" \
        --userId "${USER_ID}" \
        --required-extensions "${REQUIRED_EXTENSIONS}" \
        --n-rows "${N_ROWS}"
BENCHMARK_EXIT=$?
set -e

echo "   benchmark exit code: ${BENCHMARK_EXIT}"
echo "4. Results written to ${DATASET_DIR_HOST}/${OUTPUT_SUBDIR}/${USER_ID}"
echo "=== Finished ${ARM} arm ==="

exit "${BENCHMARK_EXIT}"