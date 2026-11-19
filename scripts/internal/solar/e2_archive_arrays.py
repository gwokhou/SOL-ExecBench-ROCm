"""Deduplicate identical output arrays without changing any observed bytes."""

import numpy as np
from numpy.lib.npyio import NpzFile

_ALIASES = "e2_output_aliases"
_OUTPUTS = ("reference", "actual", "oracle_rounded")


def compact_output_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Store byte-identical outputs once and describe the omitted fields."""
    result = arrays.copy()
    retained: list[str] = []
    aliases: list[tuple[str, str]] = []
    for name in _OUTPUTS:
        if name not in result:
            continue
        output = result[name]
        original = next(
            (
                key
                for key in retained
                if output.dtype == result[key].dtype
                and output.shape == result[key].shape
                and output.tobytes() == result[key].tobytes()
            ),
            None,
        )
        if original is None:
            retained.append(name)
        else:
            aliases.append((name, original))
            del result[name]
    if aliases:
        result[_ALIASES] = np.array(aliases)
    return result


def read_output_arrays(
    archive: NpzFile, *, with_oracle: bool = False
) -> dict[str, np.ndarray]:
    """Resolve output members in legacy, compact, and oracle-free archives."""
    aliases: dict[str, str] = {}
    if _ALIASES in archive.files:
        stored = archive[_ALIASES]
        if stored.dtype.kind != "U" or stored.shape not in {(1, 2), (2, 2)}:
            raise ValueError("E2 output aliases have an invalid shape or dtype")
        for raw_name, raw_source in stored:
            name, source = str(raw_name), str(raw_source)
            if (
                name not in _OUTPUTS[1:]
                or source not in _OUTPUTS[: _OUTPUTS.index(name)]
                or name in aliases
                or name in archive.files
                or source not in archive.files
            ):
                raise ValueError("E2 archive has an invalid output alias")
            aliases[name] = source
    array_names = _OUTPUTS if with_oracle else _OUTPUTS[:2]
    outputs: dict[str, np.ndarray] = {}
    loaded: dict[str, np.ndarray] = {}
    for name in array_names:
        source = aliases.get(name, name)
        if name == "oracle_rounded" and source not in archive.files:
            continue
        if source not in loaded:
            loaded[source] = archive[source]
        outputs[name] = loaded[source]
    return outputs
