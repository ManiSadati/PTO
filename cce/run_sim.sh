#!/usr/bin/env bash
set -euo pipefail

# Run the softmax CCE kernel through CANN's msprof op simulator.
# Use from an activated PTOAS/CANN shell, e.g.:
#   bash -ic 'activate_ptoas >/dev/null; cd ~/cce; bash run_sim.sh'

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "error: ASCEND_HOME_PATH is not set; run activate_ptoas or source CANN setenv first" >&2
  exit 1
fi

ASCEND_HOME="${ASCEND_HOME:-${ASCEND_HOME_PATH}/x86_64-linux}"
LLVM_DIR="${LLVM_DIR:-${ASCEND_HOME_PATH}/bin}"
DEVICE_ARCH="${DEVICE_ARCH:-dav-c310-vec}"
# Matches PTOAS/PTOAS_Markham/test/tilelang_st/script/run_all_st.py for A5.
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CXX="${CXX:-g++}"
MSPROF_OUT="${MSPROF_OUT:-${HOME}/.cache/cce-softmax-msprof}"
MSPROF_TIMEOUT_MIN="${MSPROF_TIMEOUT_MIN:-5}"

GEN_SCRIPT="${ROOT_DIR}/script/gen_softmax.py"
DEVICE_SRC="${ROOT_DIR}/softmax_a5.cce"
HOST_SRC="${ROOT_DIR}/host_softmax.cpp"

DEVICE_OBJ="${BUILD_DIR}/softmax.o"
HOST_EXE="${BUILD_DIR}/softmax_host"
INPUT_BIN="${BUILD_DIR}/softmax_input_fp16.bin"
GOLDEN_BIN="${BUILD_DIR}/softmax_golden_fp16.bin"

CCEC="${LLVM_DIR}/ccec"
SIM_LIB_DIR="${ASCEND_HOME}/simulator/${SOC_VERSION}/lib"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "required file not found: $1"
}

require_executable() {
  [[ -x "$1" ]] || fail "required executable not found or not executable: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_file "$GEN_SCRIPT"
require_file "$DEVICE_SRC"
require_file "$HOST_SRC"
require_executable "$CCEC"
require_executable "${ASCEND_HOME_PATH}/bin/msprof"
require_command "$PYTHON_BIN"
require_command "$CXX"

[[ -d "${ASCEND_HOME}/include" ]] || fail "CANN include directory not found under ASCEND_HOME=${ASCEND_HOME}"
[[ -d "${ASCEND_HOME}/lib64" ]] || fail "CANN lib64 directory not found under ASCEND_HOME=${ASCEND_HOME}"
[[ -d "${SIM_LIB_DIR}" ]] || fail "simulator library directory not found: ${SIM_LIB_DIR}"

mkdir -p "$BUILD_DIR"
mkdir -p "$MSPROF_OUT"
chmod 700 "$MSPROF_OUT"

export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${ASCEND_HOME}/lib64:${ASCEND_HOME}/devlib:${ASCEND_HOME}/devlib/linux/x86_64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/runtime/lib64:${ASCEND_HOME_PATH}/fwkacllib/lib64:${LD_LIBRARY_PATH:-}"

echo "==> [1/4] Generating softmax input and golden output"
"$PYTHON_BIN" "$GEN_SCRIPT" \
  --input "$INPUT_BIN" \
  --golden "$GOLDEN_BIN"

echo "==> [2/4] Compiling device kernel"
"$CCEC" -c --cce-aicore-only -O2 \
  -std=c++17 \
  --cce-aicore-arch="$DEVICE_ARCH" \
  -DREGISTER_BASE \
  -mllvm -cce-aicore-function-stack-size=16000 \
  -mllvm -cce-aicore-fp-ceiling=2 \
  -mllvm -cce-aicore-record-overflow=false \
  --cce-auto-sync \
  -mllvm -api-deps-filter \
  "$DEVICE_SRC" \
  -o "$DEVICE_OBJ"

echo "==> [3/4] Compiling host launcher"
"$CXX" -O2 -std=c++17 "$HOST_SRC" \
  -I "${ASCEND_HOME}/include" \
  -I "${ASCEND_HOME}/include/experiment/runtime" \
  -I "${ASCEND_HOME}/pkg_inc/runtime" \
  -I "${ASCEND_HOME}/include/experiment/msprof" \
  -L "${ASCEND_HOME}/lib64" \
  -L "${ASCEND_HOME}/devlib" \
  -L "${ASCEND_HOME}/devlib/linux/x86_64" \
  -L "${ASCEND_HOME_PATH}/runtime/lib64" \
  -Wl,-rpath,"${ASCEND_HOME}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME}/devlib" \
  -Wl,-rpath,"${ASCEND_HOME}/devlib/linux/x86_64" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/runtime/lib64" \
  -lascendcl \
  -lruntime \
  -lregister \
  -lerror_manager \
  -lascend_trace \
  -lc_sec \
  -o "$HOST_EXE"

echo "==> [4/4] Running under msprof op simulator"
MSPROF_LOG="${MSPROF_OUT}/msprof_softmax.log"
set +e
"${ASCEND_HOME_PATH}/bin/msprof" op simulator \
  --application="${HOST_EXE} ${DEVICE_OBJ} ${INPUT_BIN} ${GOLDEN_BIN}" \
  --kernel-name=softmax__kernel0 \
  --launch-count=1 \
  --soc-version="$SOC_VERSION" \
  --timeout="$MSPROF_TIMEOUT_MIN" \
  --output="$MSPROF_OUT" \
  2>&1 | tee "$MSPROF_LOG"
MSPROF_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "$MSPROF_STATUS" -ne 0 ]]; then
  fail "msprof exited with status ${MSPROF_STATUS}; see ${MSPROF_LOG}"
fi

if grep -Eq 'Child process exited|Profiling kernels result is: 0 success|Profiling data parse failed' "$MSPROF_LOG"; then
  fail "simulator reported a failed kernel run; see ${MSPROF_LOG}"
fi

echo
echo "Simulator output: $MSPROF_OUT"
