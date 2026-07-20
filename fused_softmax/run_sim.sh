#!/usr/bin/env bash
set -euo pipefail

# Direct A5 camodel run for the fused softmax CCE kernel.
#
# This follows the local PTO/TileLang ST pattern:
#   1. Build the CCE kernel + launch wrapper into a fatobj-linked shared lib.
#   2. Link a host verifier against runtime_camodel.
#   3. Run the host directly, without msprof/opprof.
#
# Usage:
#   bash -ic 'activate_ptoas >/dev/null; cd ~/mani-PTO/fused_softmax; bash run_sim.sh'

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "error: ASCEND_HOME_PATH is not set; run activate_ptoas or source CANN setenv first" >&2
  exit 1
fi

CANN_HOME="${ASCEND_HOME_PATH}"
ASCEND_X86_HOME="${ASCEND_X86_HOME:-${ASCEND_HOME_PATH}/x86_64-linux}"
LLVM_DIR="${LLVM_DIR:-${ASCEND_HOME_PATH}/bin}"
DEVICE_ARCH="${DEVICE_ARCH:-dav-c310-vec}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CXX="${CXX:-g++}"
SIM_TIMEOUT_SEC="${SIM_TIMEOUT_SEC:-120}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BUILD_DIR}/sim_artifacts}"

GEN_SCRIPT="${ROOT_DIR}/script/gen_softmax.py"
DEVICE_SRC="${ROOT_DIR}/softmax_a5.cce"
LAUNCH_SRC="${ROOT_DIR}/softmax_launch.cpp"
HOST_SRC="${ROOT_DIR}/host_softmax_sim.cpp"

KERNEL_SO="${BUILD_DIR}/libsoftmax_kernel.so"
HOST_EXE="${BUILD_DIR}/softmax_host_sim"
INPUT_BIN="${BUILD_DIR}/softmax_input_fp16.bin"
GOLDEN_BIN="${BUILD_DIR}/softmax_golden_fp16.bin"

CCEC="${LLVM_DIR}/ccec"
SIM_LIB_DIR="${CANN_HOME}/tools/simulator/${SOC_VERSION}/lib"

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
require_file "$LAUNCH_SRC"
require_file "$HOST_SRC"
require_executable "$CCEC"
require_command "$PYTHON_BIN"
require_command "$CXX"
require_command timeout

[[ -d "${CANN_HOME}/include" ]] || fail "CANN include directory not found: ${CANN_HOME}/include"
[[ -d "${CANN_HOME}/lib64" ]] || fail "CANN lib64 directory not found: ${CANN_HOME}/lib64"
[[ -d "${SIM_LIB_DIR}" ]] || fail "simulator library directory not found: ${SIM_LIB_DIR}"

mkdir -p "$BUILD_DIR"
mkdir -p "$ARTIFACT_DIR"

export LD_LIBRARY_PATH="${BUILD_DIR}:${SIM_LIB_DIR}:${CANN_HOME}/lib64:${ASCEND_X86_HOME}/lib64:${ASCEND_X86_HOME}/devlib:${ASCEND_X86_HOME}/devlib/linux/x86_64:${LD_LIBRARY_PATH:-}"

echo "==> [1/4] Generating softmax input and golden output"
"$PYTHON_BIN" "$GEN_SCRIPT" \
  --input "$INPUT_BIN" \
  --golden "$GOLDEN_BIN"

echo "==> [2/4] Building kernel shared library for camodel"
"$CCEC" -fPIC -shared --cce-fatobj-link \
  --cce-aicore-arch="$DEVICE_ARCH" \
  -DREGISTER_BASE \
  -std=c++17 \
  -I "${CANN_HOME}/pkg_inc" \
  -I "${CANN_HOME}/pkg_inc/profiling" \
  -I "${CANN_HOME}/pkg_inc/runtime/runtime" \
  -x cce "$DEVICE_SRC" \
  -x cce "$LAUNCH_SRC" \
  -L "$SIM_LIB_DIR" \
  -L "${CANN_HOME}/lib64" \
  -Wl,-rpath,"$SIM_LIB_DIR" \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -lruntime_camodel \
  -o "$KERNEL_SO"

echo "==> [3/4] Building simulator host verifier"
"$CXX" -O2 -std=c++17 "$HOST_SRC" \
  -I "${CANN_HOME}/include" \
  -L "$BUILD_DIR" \
  -L "$SIM_LIB_DIR" \
  -L "${CANN_HOME}/lib64" \
  -Wl,-rpath,"$BUILD_DIR" \
  -Wl,-rpath,"$SIM_LIB_DIR" \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -lsoftmax_kernel \
  -lruntime_camodel \
  -lascendcl \
  -ltiling_api \
  -lplatform \
  -lc_sec \
  -ldl \
  -lnnopbase \
  -lpthread \
  -o "$HOST_EXE"

echo "==> [4/4] Running direct camodel verifier"
(
  cd "$ARTIFACT_DIR"
  timeout "${SIM_TIMEOUT_SEC}s" "$HOST_EXE" "$INPUT_BIN" "$GOLDEN_BIN"
)

echo
echo "Fused softmax camodel workflow completed successfully."
echo "  Kernel shared library: $KERNEL_SO"
echo "  Host executable: $HOST_EXE"
echo "  Input: $INPUT_BIN"
echo "  Golden: $GOLDEN_BIN"
echo "  Simulator artifacts: $ARTIFACT_DIR"
