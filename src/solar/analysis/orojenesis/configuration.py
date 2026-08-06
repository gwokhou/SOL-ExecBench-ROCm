"""Pinned Orojenesis toolchain and composition configuration."""

import os

from solar.analysis.orojenesis.identity import OrojenesisIdentityPolicy

OROJENESIS_COMMIT = "97d52178bf9a9c209bf79be96b87c164bcd35625"
OROJENESIS_REPOSITORY = "https://github.com/NVlabs/timeloop.git"
OROJENESIS_TREE_OID = "05b05ec5a2a2979b1fe92046b937556d9ad99847"
OROJENESIS_SOURCE_ARCHIVE_SHA256 = (
    "3a254ab201d92b7eba993d3c7dcf0bb148a31dc9e57ece020fbaa38ad67c7873"
)
OROJENESIS_BUILDER_IMAGE = (
    "ubuntu:24.04@sha256:"
    "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
)
OROJENESIS_UBUNTU_SNAPSHOT = "20260718T000000Z"
OROJENESIS_OPENSSL_BOOTSTRAP_SHA256 = (
    "9c79333ab21bce0fb8dd92304cd76b3b1c427b0f2fedc897257fb5cced37c39e"
)
OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256 = (
    "641de77d8f142cfd62a1a6f964ba67b20754d3337c480efb529d086075a06c9a"
)
OROJENESIS_SOURCE_DATE_EPOCH = 1753058729
OROJENESIS_COMPILER_WRAPPER_SHA256 = (
    "04363ce239f76a4763490c049de1d69e2265d59578d51bed753f688c6f75278d"
)
# Reproduced by two no-cache builds from the pinned source and toolchain.
OROJENESIS_TRUSTED_MAPPER_SHA256: frozenset[str] = frozenset(
    {"18591892b1ecec3264ec729b0e457ec9f22422993f656ece40dba809c032d77a"},
)
OROJENESIS_PROVENANCE_FILENAME = "orojenesis-provenance.json"
OROJENESIS_FALLBACK_MAPPER_THREADS = 1

MULTI_EINSUM_SOLVER = "NVlabs/Orojenesis tiled-fusion"
MULTI_EINSUM_COMPOSITION = "linear_matmul_compatible_tiles_sum_capacity_v1"
MULTI_EINSUM_LAYOUT_COMPOSITION = "linear_matmul_axis_map_tile_shape_v2"
MULTI_EINSUM_BATCH_COMPOSITION = "broadcast_batch_linear_tile_shape_v1"
MULTI_EINSUM_FANOUT_COMPOSITION = "matmul_fanout_tree_tile_shape_v1"

IDENTITY_POLICY = OrojenesisIdentityPolicy(
    repository=OROJENESIS_REPOSITORY,
    commit=OROJENESIS_COMMIT,
    tree_oid=OROJENESIS_TREE_OID,
    source_archive_sha256=OROJENESIS_SOURCE_ARCHIVE_SHA256,
    compiler_wrapper_sha256=OROJENESIS_COMPILER_WRAPPER_SHA256,
    builder_image=OROJENESIS_BUILDER_IMAGE,
    ubuntu_snapshot=OROJENESIS_UBUNTU_SNAPSHOT,
    openssl_sha256=OROJENESIS_OPENSSL_BOOTSTRAP_SHA256,
    ca_certificates_sha256=OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256,
    source_date_epoch=OROJENESIS_SOURCE_DATE_EPOCH,
    provenance_filename=OROJENESIS_PROVENANCE_FILENAME,
    trusted_mapper_sha256=OROJENESIS_TRUSTED_MAPPER_SHA256,
)


def available_logical_cpu_count(
    *,
    cpu_ids: frozenset[int] | None = None,
) -> int | None:
    """Return logical CPUs available to this process, respecting affinity."""
    available = _process_cpu_ids() if cpu_ids is None else cpu_ids
    return len(available) or None


def orojenesis_mapper_thread_count(
    *,
    cpu_ids: frozenset[int] | None = None,
) -> int:
    """Use all process-visible logical CPUs, with a conservative fallback."""
    return (
        available_logical_cpu_count(cpu_ids=cpu_ids)
        or OROJENESIS_FALLBACK_MAPPER_THREADS
    )


def _process_cpu_ids() -> frozenset[int]:
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical_cpus = os.cpu_count()
        return frozenset(range(logical_cpus)) if logical_cpus else frozenset()
