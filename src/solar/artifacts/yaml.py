# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""YAML serialization primitives for canonical SOLAR artifacts."""

import yaml


class NoAliasDumper(yaml.SafeDumper):
    """Disable YAML anchors so canonical artifacts remain reviewable."""

    def ignore_aliases(self, data: object) -> bool:
        """Disable aliases for every serialized value."""
        del data
        return True


__all__ = ["NoAliasDumper"]
