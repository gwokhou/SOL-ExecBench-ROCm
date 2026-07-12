from __future__ import annotations

import re

from sol_execbench.core.timestamps import utc_timestamp


def test_utc_timestamp_uses_second_precision_zulu_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_timestamp())
