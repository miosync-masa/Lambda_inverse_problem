"""
分割前後の数値同値性テスト。

`lambda3_detector_v2` (旧モノリス) と新パッケージ `lambda3_detector`
が、同じ入力・同じseedに対して同じ ``Lambda3Result.paths`` と
``detect_anomalies`` スコアを返すことを確認する。

pytest:
    cd Lambda_inverse_problem
    python -m pytest tests/test_split_equivalence.py -v
"""

import importlib
import sys

import numpy as np
import pytest


def _reload_modules_with_fresh_globals():
    """旧と新で独立した module state（特にmutable globals）を確保。"""
    # 旧側
    if 'lambda3_detector_v2' in sys.modules:
        importlib.reload(sys.modules['lambda3_detector_v2'])
    # 新側のconfigは module attribute をそのまま読むので reload は不要だが、
    # キャッシュ汚染を避けるため新規Detectorを毎回作る。


def _make_events(seed: int = 42, n_events: int = 80, n_features: int = 8):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, size=(n_events, n_features))
    # 数イベントだけ強い外れ値を入れて構造を作る（解が決定論的になる程度に）
    for idx in (10, 30, 55, 70):
        base[idx] += rng.normal(0, 4.0, size=n_features)
    return base


@pytest.fixture(scope='module')
def events():
    return _make_events()


def _run(detector_cls, events: np.ndarray, *, legacy: bool = False):
    """Detectorインスタンスを作って analyze + detect_anomalies を実行。

    Args:
        legacy: True なら新パッケージを legacy 設定 (use_sparse_solver=False)
            で動かす。旧モノリスは常に full solver なので、新側もこのモードで
            比較しないと数値一致しない。
    """
    np.random.seed(0)
    if legacy:
        # 新パッケージのみ legacy mode 対応。旧monolithは元から full のみ。
        try:
            from lambda3_detector import L3Config
            detector = detector_cls(L3Config(use_sparse_solver=False))
        except TypeError:
            # 旧monolithは L3Config に use_sparse_solver を持たない
            detector = detector_cls()
    else:
        detector = detector_cls()
    np.random.seed(0)
    result = detector.analyze(events)
    np.random.seed(0)
    scores = detector.detect_anomalies(result, events, use_adaptive_weights=False)
    return result, scores


def test_paths_equivalence(events):
    """同一seed下で paths（構造テンソル）が数値一致することを確認"""
    _reload_modules_with_fresh_globals()
    from lambda3_detector_v2 import Lambda3ZeroShotDetector as OldDetector
    from lambda3_detector import Lambda3ZeroShotDetector as NewDetector

    old_result, _ = _run(OldDetector, events, legacy=True)
    new_result, _ = _run(NewDetector, events, legacy=True)

    assert set(old_result.paths.keys()) == set(new_result.paths.keys())
    for i in old_result.paths:
        np.testing.assert_allclose(
            old_result.paths[i], new_result.paths[i],
            rtol=1e-5, atol=1e-7,
            err_msg=f"paths[{i}] mismatch",
        )


def test_topological_quantities_equivalence(events):
    """topological charges / stabilities / energies の同値性"""
    _reload_modules_with_fresh_globals()
    from lambda3_detector_v2 import Lambda3ZeroShotDetector as OldDetector
    from lambda3_detector import Lambda3ZeroShotDetector as NewDetector

    old_result, _ = _run(OldDetector, events, legacy=True)
    new_result, _ = _run(NewDetector, events, legacy=True)

    for i in old_result.topological_charges:
        assert np.isclose(
            old_result.topological_charges[i],
            new_result.topological_charges[i],
            rtol=1e-5,
        ), f"Q_Λ[{i}] mismatch"
        assert np.isclose(
            old_result.stabilities[i], new_result.stabilities[i], rtol=1e-5,
        ), f"σ_Q[{i}] mismatch"
        assert np.isclose(
            old_result.energies[i], new_result.energies[i], rtol=1e-5,
        ), f"E[{i}] mismatch"


def test_anomaly_scores_equivalence(events):
    """detect_anomalies の最終スコアが数値一致することを確認"""
    _reload_modules_with_fresh_globals()
    from lambda3_detector_v2 import Lambda3ZeroShotDetector as OldDetector
    from lambda3_detector import Lambda3ZeroShotDetector as NewDetector

    _, old_scores = _run(OldDetector, events, legacy=True)
    _, new_scores = _run(NewDetector, events, legacy=True)

    np.testing.assert_allclose(
        old_scores, new_scores, rtol=1e-5, atol=1e-7,
        err_msg="detect_anomalies scores diverged",
    )


if __name__ == "__main__":
    # スクリプト実行: pytestを介さずシンプルに確認したい用途
    ev = _make_events()
    test_paths_equivalence(ev)
    test_topological_quantities_equivalence(ev)
    test_anomaly_scores_equivalence(ev)
    print("All equivalence tests passed.")
