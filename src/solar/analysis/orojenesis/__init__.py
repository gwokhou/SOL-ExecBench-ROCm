# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned Orojenesis integration used by formal analysis."""

from solar.analysis.orojenesis.configuration import (
    MULTI_EINSUM_BATCH_COMPOSITION,
    MULTI_EINSUM_COMPOSITION,
    MULTI_EINSUM_FANOUT_COMPOSITION,
    MULTI_EINSUM_LAYOUT_COMPOSITION,
    MULTI_EINSUM_SOLVER,
    OROJENESIS_BUILDER_IMAGE,
    OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256,
    OROJENESIS_COMMIT,
    OROJENESIS_COMPILER_WRAPPER_SHA256,
    OROJENESIS_OPENSSL_BOOTSTRAP_SHA256,
    OROJENESIS_PROVENANCE_FILENAME,
    OROJENESIS_REPOSITORY,
    OROJENESIS_SOURCE_ARCHIVE_SHA256,
    OROJENESIS_SOURCE_DATE_EPOCH,
    OROJENESIS_TREE_OID,
    OROJENESIS_TRUSTED_MAPPER_SHA256,
    OROJENESIS_UBUNTU_SNAPSHOT,
)
from solar.analysis.orojenesis.curves import (
    compose_multi_einsum_curve,
    parse_multi_einsum_curve,
    parse_multi_einsum_region_curve,
    parse_multi_mapping_records,
)
from solar.analysis.orojenesis.errors import OrojenesisError
from solar.analysis.orojenesis.multi_einsum import (
    find_multi_einsum_chains,
    multi_einsum_layer_problem,
    multi_einsum_mapper_role,
    multi_einsum_problem,
)
from solar.analysis.orojenesis.regions import (
    compose_multi_einsum_region_curve,
    find_multi_einsum_regions,
    multi_einsum_region_mapper_role,
    multi_einsum_region_problem,
)
from solar.analysis.orojenesis.runner import (
    OrojenesisRunner,
    select_capacity_point,
)

__all__ = [
    "MULTI_EINSUM_BATCH_COMPOSITION",
    "MULTI_EINSUM_COMPOSITION",
    "MULTI_EINSUM_FANOUT_COMPOSITION",
    "MULTI_EINSUM_LAYOUT_COMPOSITION",
    "MULTI_EINSUM_SOLVER",
    "OROJENESIS_BUILDER_IMAGE",
    "OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256",
    "OROJENESIS_COMMIT",
    "OROJENESIS_COMPILER_WRAPPER_SHA256",
    "OROJENESIS_OPENSSL_BOOTSTRAP_SHA256",
    "OROJENESIS_PROVENANCE_FILENAME",
    "OROJENESIS_REPOSITORY",
    "OROJENESIS_SOURCE_ARCHIVE_SHA256",
    "OROJENESIS_SOURCE_DATE_EPOCH",
    "OROJENESIS_TREE_OID",
    "OROJENESIS_TRUSTED_MAPPER_SHA256",
    "OROJENESIS_UBUNTU_SNAPSHOT",
    "OrojenesisError",
    "OrojenesisRunner",
    "compose_multi_einsum_curve",
    "compose_multi_einsum_region_curve",
    "find_multi_einsum_chains",
    "find_multi_einsum_regions",
    "multi_einsum_layer_problem",
    "multi_einsum_mapper_role",
    "multi_einsum_problem",
    "multi_einsum_region_mapper_role",
    "multi_einsum_region_problem",
    "parse_multi_einsum_curve",
    "parse_multi_einsum_region_curve",
    "parse_multi_mapping_records",
    "select_capacity_point",
]
