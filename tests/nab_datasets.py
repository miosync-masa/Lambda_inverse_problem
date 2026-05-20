"""
NAB (Numenta Anomaly Benchmark) データローダ。

CSV (timestamp, value) と combined_windows.json を読み、
Lambda3 detector 用の (n, 1) numpy 配列＋窓インデックスを返す。

Usage::
    for sample in iter_category('realKnownCause'):
        result = detector.analyze(sample.values)
        ...

NAB_ROOT のデフォルトはリポジトリ親 ``../NAB`` (this file: tests/ → ../../NAB)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

NAB_ROOT_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'NAB')
)


@dataclass
class NABSample:
    name: str                                              # "realKnownCause/nyc_taxi.csv"
    values: np.ndarray                                     # (n, 1) float64
    timestamps: List[pd.Timestamp]                         # length n
    windows_ts: List[Tuple[pd.Timestamp, pd.Timestamp]]    # raw label windows
    window_indices: List[Tuple[int, int]]                  # inclusive (si, ei)

    @property
    def n(self) -> int:
        return len(self.timestamps)

    @property
    def labels(self) -> np.ndarray:
        """0/1 binary labels per index (1 = inside any anomaly window)."""
        y = np.zeros(self.n, dtype=np.int32)
        for si, ei in self.window_indices:
            y[si:ei + 1] = 1
        return y


def _to_indices(ts_series: pd.Series,
                windows_ts: List[Tuple[pd.Timestamp, pd.Timestamp]]
                ) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    n = len(ts_series)
    for s, e in windows_ts:
        si = int(ts_series.searchsorted(s, side='left'))
        ei = int(ts_series.searchsorted(e, side='right')) - 1
        si = max(0, min(si, n - 1))
        ei = max(si, min(ei, n - 1))
        indices.append((si, ei))
    return indices


def load_nab_sample(rel_path: str,
                    windows_raw: List[List[str]],
                    nab_root: str = NAB_ROOT_DEFAULT) -> NABSample:
    csv_path = os.path.join(nab_root, 'data', rel_path)
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    ts_series = df['timestamp']
    timestamps = ts_series.tolist()
    values = df['value'].to_numpy(dtype=np.float64).reshape(-1, 1)

    windows_ts = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in windows_raw]
    window_indices = _to_indices(ts_series, windows_ts)

    return NABSample(
        name=rel_path,
        values=values,
        timestamps=timestamps,
        windows_ts=windows_ts,
        window_indices=window_indices,
    )


def iter_category(category: str,
                  nab_root: str = NAB_ROOT_DEFAULT,
                  windows_file: str = 'combined_windows.json',
                  include_empty: bool = False) -> Iterator[NABSample]:
    """指定カテゴリ (例: 'realKnownCause') の全ファイルを iterate。

    include_empty=False の場合、anomaly window を持たないファイルはスキップ。
    """
    with open(os.path.join(nab_root, 'labels', windows_file)) as f:
        all_windows = json.load(f)
    prefix = category + '/'
    for rel_path, wins in all_windows.items():
        if not rel_path.startswith(prefix):
            continue
        if not wins and not include_empty:
            continue
        yield load_nab_sample(rel_path, wins, nab_root)
