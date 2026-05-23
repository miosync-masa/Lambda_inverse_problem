"""StreamingScorer 抽象基底。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StreamingScorer(ABC):
    """Streaming anomaly scorer の基底クラス。

    使い方:
        s = MyStreamingScorer(...)
        s.calibrate(events_cal)         # 異常無しと仮定する先頭区間で baseline 固定
        for t in range(cal_end, n):
            raw_score = s.score(events, t)  # events[:t+1] のみを使う (future leakage 禁止)
            if raw_score > s.threshold:
                # フレーム t は異常
                ...

    実装上のルール:
      - calibrate() でのみ baseline/threshold を変更する
      - score() は state を持たず、events と t だけから決定論的に値を返す
        (内部キャッシュ可、ただし t に依存しない部分のみ)
      - lookback ウィンドウは events[max(0, t-W):t+1] の形で参照
    """

    @abstractmethod
    def calibrate(self, events_cal: np.ndarray) -> None:
        """calibration 区間 (events_cal: (n_cal, d) または (n_cal,)) から
        内部 baseline を確立する。同時に self.threshold を決定する。
        """
        ...

    @abstractmethod
    def score(self, events: np.ndarray, t: int) -> float:
        """events[:t+1] のみを参照して、フレーム t の raw anomaly score を返す。
        events[t+1:] へのアクセスは禁止 (future leakage)。
        """
        ...

    @property
    @abstractmethod
    def threshold(self) -> float:
        """calibrate 後に固定される binary 判定閾値。
        raw_score > threshold で「フラグ」とみなす。
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
