"""Parquet 저장 헬퍼."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_parquet(
    path: Path,
    new_df: pd.DataFrame,
    *,
    dedupe_subset: list[str] | tuple[str, ...] | None = None,
    sort_by: str | list[str] | tuple[str, ...] | None = None,
    compression: str = "snappy",
) -> pd.DataFrame:
    """기존 parquet가 있으면 이어 붙이고, 없으면 새로 저장한다."""
    if path.exists():
        try:
            existing_df = pd.read_parquet(path)
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if existing_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = existing_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    if dedupe_subset and not merged.empty:
        merged = merged.drop_duplicates(subset=list(dedupe_subset), keep="last")

    if sort_by and not merged.empty:
        if isinstance(sort_by, (list, tuple)):
            merged = merged.sort_values(list(sort_by))
        else:
            merged = merged.sort_values(sort_by)

    merged = merged.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, compression=compression, index=False)
    return merged