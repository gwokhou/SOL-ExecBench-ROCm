#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
evidence_root="$(mktemp -d)"
first="${evidence_root}/first"
second="${evidence_root}/second"
requested_output="${1:-${repository_root}/data/outputs/orojenesis-reproducible}"
mkdir -p "$(dirname "${requested_output}")"
output_parent="$(cd "$(dirname "${requested_output}")" && pwd)"
output="${output_parent}/$(basename "${requested_output}")"
publish_staging=""

cleanup() {
    rm -rf -- "${evidence_root}"
    if [ -n "${publish_staging}" ] && [ -d "${publish_staging}" ]; then
        rm -rf -- "${publish_staging}"
    fi
}
trap cleanup EXIT

if [ -e "${output}" ]; then
    echo "ERROR: output already exists: ${output}" >&2
    exit 2
fi

docker build \
    --no-cache \
    --target artifact \
    --output "type=local,dest=${first}" \
    --file "${repository_root}/docker/orojenesis/Dockerfile" \
    "${repository_root}"
docker build \
    --no-cache \
    --target artifact \
    --output "type=local,dest=${second}" \
    --file "${repository_root}/docker/orojenesis/Dockerfile" \
    "${repository_root}"

first_home="${first}/opt/orojenesis"
second_home="${second}/opt/orojenesis"
cmp \
    "${first_home}/bin/timeloop-mapper" \
    "${second_home}/bin/timeloop-mapper"
cmp \
    "${first_home}/orojenesis-provenance.json" \
    "${second_home}/orojenesis-provenance.json"

publish_staging="$(mktemp -d "${output_parent}/.orojenesis-publish.XXXXXX")"
cp -a "${first_home}/." "${publish_staging}/"
mv "${publish_staging}" "${output}"
publish_staging=""

sha256sum "${output}/bin/timeloop-mapper"
echo "Published reproducible Orojenesis artifact: ${output}"
