#!/usr/bin/env bash
set -euo pipefail

# Complete A5 fused softmax workflow:
#   1. Generate FP16 input and NumPy golden output.
#   2. Compile the CCE device kernel to an AIVector ELF object.
#   3. Compile the host launcher/verifier.
#   4. Run the kernel and verify its output.
#
# The defaults below match the paths used on the original x86 A5 machine.
# Override them without editing this file, for example:
#   ASCEND_HOME=/path/to/CANN LLVM_DIR=/path/to/llvm/bin ./run.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

ASCEND_HOME="${ASCEND_HOME:-/usr/local/CANN/cann/x86_64-linux}"
LLVM_DIR="${LLVM_DIR:-/home/w00555904/melika/llvm-project/build_HiIPU/bin}"
DEVICE_ARCH="${DEVICE_ARCH:-dav-c310-vec}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CXX="${CXX:-g++}"

GEN_SCRIPT="${ROOT_DIR}/script/gen_softmax.py"
DEVICE_SRC="${ROOT_DIR}/softmax_a5.cce"
HOST_SRC="${ROOT_DIR}/host_softmax.cpp"

DEVICE_OBJ="${BUILD_DIR}/softmax.o"
HOST_EXE="${BUILD_DIR}/softmax_host"
INPUT_BIN="${BUILD_DIR}/softmax_input_fp16.bin"
GOLDEN_BIN="${BUILD_DIR}/softmax_golden_fp16.bin"

CCEC="${LLVM_DIR}/ccec"

# Add the CANN libraries used by the host executable. Keep any existing entries
# after the desired A5/CANN paths rather than replacing them entirely.
export LD_LIBRARY_PATH="${ASCEND_HOME}/lib64:${ASCEND_HOME}/devlib:${ASCEND_HOME}/devlib/linux/x86_64:${ASCEND_HOME}/runtime/lib64:${LD_LIBRARY_PATH:-}"

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
require_command "$PYTHON_BIN"
require_command "$CXX"

[[ -d "${ASCEND_HOME}/include" ]] || fail "CANN include directory not found under ASCEND_HOME=${ASCEND_HOME}"
[[ -d "${ASCEND_HOME}/lib64" ]] || fail "CANN lib64 directory not found under ASCEND_HOME=${ASCEND_HOME}"

mkdir -p "$BUILD_DIR"

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
  -L "${ASCEND_HOME}/runtime/lib64" \
  -Wl,-rpath,"${ASCEND_HOME}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME}/devlib" \
  -Wl,-rpath,"${ASCEND_HOME}/devlib/linux/x86_64" \
  -Wl,-rpath,"${ASCEND_HOME}/runtime/lib64" \
  -lascendcl \
  -lruntime \
  -lregister \
  -lerror_manager \
  -lascend_trace \
  -lc_sec \
  -o "$HOST_EXE"

echo "==> [4/4] Running and verifying softmax"
"$HOST_EXE" "$DEVICE_OBJ" "$INPUT_BIN" "$GOLDEN_BIN"

echo
echo "Softmax workflow completed successfully."
echo "  Device object: $DEVICE_OBJ"
echo "  Host executable: $HOST_EXE"
echo "  Input: $INPUT_BIN"
echo "  Golden: $GOLDEN_BIN"
