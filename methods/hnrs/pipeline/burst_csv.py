from __future__ import annotations

from pathlib import Path

import pandas as pd


UNIT_BURST_COLUMNS = [
    "Electrode",
    "Cluster",
    "burst_start_ms",
    "burst_end_ms",
    "burst_duration_ms",
    "num_onsets",
    "RS",
    "RS_aggregation",
    "Method",
]

REGION_BURST_COLUMNS = [
    "Region",
    "rs_core_start_ms",
    "rs_core_end_ms",
    "burst_start_ms",
    "burst_end_ms",
    "burst_duration_ms",
    "num_onsets",
    "RS",
    "Method",
]

NETWORK_BURST_COLUMNS = [
    "rs_core_start_ms",
    "rs_core_end_ms",
    "burst_start_ms",
    "burst_end_ms",
    "burst_duration_ms",
    "num_onsets",
    "RS",
]


def write_burst_csv(
    rows: list[dict] | pd.DataFrame,
    path: str | Path,
    columns: list[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        frame = rows.reindex(columns=columns)
    else:
        frame = pd.DataFrame.from_records(rows, columns=columns)
    frame.to_csv(output_path, index=False)
