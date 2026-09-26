#!/usr/bin/env bash
# Rootless HIP build of the PrismML llama.cpp fork for gfx1101 (RX 7800 XT).
# Usage: [SRC=/path/to/fork] [COMPILER=clang|amdclang] build.sh <build-dir> [extra cmake -D args...]
#   build-dir is absolute or relative to SRC. Builds llama-server, llama-bench, llama-perplexity,
#   test-backend-ops. Needs, all under this directory (no sudo):
#     lld/  — Arch 'lld' 22.1.8 extracted (clang's amdgcn link step needs ld.lld)
#     hbc/  — Arch 'hipblas-common' 7.2.4 extracted (ROCm runtime's hipblas.h includes it)
#     rocm-llvm/ — Arch 'rocm-llvm' 7.2.4 extracted (only for COMPILER=amdclang)
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-/home/evank/llama-prism-src}"
B="${1:?build dir}"; shift
[[ "$B" = /* ]] || B="$SRC/$B"
ROCM=/home/evank/rocm-runtime/opt/rocm
VENV=/home/evank/dev/inference/.venv/bin
case "${COMPILER:-clang}" in
  clang)    CXX=/usr/bin/clang++; DEVLIB=/opt/rocm/amdgcn/bitcode ;;
  amdclang) CXX="$T/rocm-llvm/opt/rocm/lib/llvm/bin/clang++"; DEVLIB="$T/rocm-llvm/opt/rocm/lib/llvm/lib/clang/$(ls "$T/rocm-llvm/opt/rocm/lib/llvm/lib/clang")/../../../../amdgcn/bitcode"
            [ -d "$DEVLIB" ] || DEVLIB=/opt/rocm/amdgcn/bitcode ;;
esac
cat > "$T/hipwrap-${COMPILER:-clang}" <<EOF
#!/usr/bin/env bash
export PATH="$T/lld/usr/bin:\$PATH" LD_LIBRARY_PATH="$T/lld/usr/lib:$T/rocm-llvm/opt/rocm/lib/llvm/lib:\${LD_LIBRARY_PATH:-}"
exec "$CXX" "\$@"
EOF
chmod +x "$T/hipwrap-${COMPILER:-clang}"
export PATH="$T/lld/usr/bin:$VENV:$PATH" LD_LIBRARY_PATH="$T/lld/usr/lib:${LD_LIBRARY_PATH:-}"
"$VENV/cmake" -S "$SRC" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$VENV/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1101 -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_HIP_COMPILER="$T/hipwrap-${COMPILER:-clang}" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM --rocm-device-lib-path=$DEVLIB -I$T/hbc/opt/rocm/include" \
  -DCMAKE_PREFIX_PATH="$T/hbc/opt/rocm;$ROCM;/opt/rocm" "$@"
"$VENV/ninja" -C "$B" -j8 llama-server llama-bench llama-perplexity test-backend-ops
