#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Header-only CK support; no precompiled device-instance library is installed.
set -euo pipefail

readonly CK_REVISION=39d8d60365d2272a5366f76ccb3bfba24b219eaf
readonly CK_REPOSITORY=https://github.com/ROCm/rocm-libraries.git
readonly CK_TARGET=gfx1200
readonly CK_PREFIX=/opt/rocm
ck_tmp=$(mktemp -d)
trap 'rm -rf -- "$ck_tmp"' EXIT

git init -q "$ck_tmp/source"
git -C "$ck_tmp/source" remote add origin "$CK_REPOSITORY"
git -C "$ck_tmp/source" sparse-checkout set projects/composablekernel
timeout 300 git -C "$ck_tmp/source" fetch -q --depth 1 --filter=blob:none origin "$CK_REVISION"
timeout 300 git -C "$ck_tmp/source" checkout -q --detach FETCH_HEAD
test "$(git -C "$ck_tmp/source" rev-parse HEAD)" = "$CK_REVISION"
ck_source="$ck_tmp/source/projects/composablekernel"

cmake -S "$ck_source" -B "$ck_tmp/build" \
    -DCMAKE_CXX_COMPILER="$CK_PREFIX/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER="$CK_PREFIX/llvm/bin/clang++" \
    -DGPU_TARGETS="$CK_TARGET" -DCMAKE_HIP_ARCHITECTURES="$CK_TARGET" \
    -DCMAKE_INSTALL_PREFIX="$CK_PREFIX" \
    -DBUILD_CK_DEVICE_INSTANCES=OFF -DBUILD_CK_PROFILER=OFF \
    -DBUILD_CK_EXAMPLES=OFF -DBUILD_CK_TUTORIALS=OFF \
    -DBUILD_TESTING=OFF -DENABLE_CLANG_CPP_CHECKS=OFF
# Upstream's instance-free install only installs generated config/version files.
# Copy the unmodified public headers as well, retaining the upstream license.
cp -a "$ck_source/include/." "$CK_PREFIX/include/"
cp -a "$ck_source/library/include/." "$CK_PREFIX/include/"
cmake --install "$ck_tmp/build"
printf '%s\n' "$CK_REPOSITORY" "$CK_REVISION" "$CK_TARGET" \
    > "$CK_PREFIX/share/doc/composablekernel/source.txt"
test -f "$CK_PREFIX/include/ck/ck.hpp"
test -f "$CK_PREFIX/include/ck/config.h"
