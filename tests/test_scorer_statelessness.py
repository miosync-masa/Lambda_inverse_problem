"""
Scorerが純粋関数として振る舞うことを確認するテスト。

各scorerについて:
  1. 同じ入力で2回呼んで同じ出力が返る（idempotency）
  2. 別データで呼んだあと元データで呼んでも結果がdriftしない（no leaked state）
  3. インスタンス内部状態が呼び出しで変化しない（vars(scorer) 不変）
"""

import copy

import numpy as np
import pytest

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    HybridScorer,
    JumpScorer,
    KernelScorer,
    StructuralScorer,
)

SCORERS = [
    ('jump', JumpScorer),
    ('hybrid', HybridScorer),
    # Polynomialで固定（auto-selectはnp.random.choiceに依存するのでstatelessのチェック対象外）
    ('kernel', lambda: KernelScorer(kernel_type=1, degree=3, coef0=1.0)),
    ('structural', StructuralScorer),
]


def _make_events(seed: int, n_events: int = 60, n_features: int = 6):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, size=(n_events, n_features))
    for idx in (5, 25, 45):
        base[idx] += rng.normal(0, 4.0, size=n_features)
    return base


@pytest.fixture(scope='module')
def analyzed():
    """同じdetectorで2セット解析しておく（呼び出し側はimmutableに使う）"""
    events_a = _make_events(seed=1)
    events_b = _make_events(seed=2)

    det_a = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0)
    result_a = det_a.analyze(events_a)

    det_b = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0)
    result_b = det_b.analyze(events_b)

    return events_a, result_a, events_b, result_b


@pytest.mark.parametrize('name,build', SCORERS)
def test_idempotent_same_input(name, build, analyzed):
    """同じ入力で繰り返し呼んでも出力は同じ。"""
    events_a, result_a, _, _ = analyzed
    scorer = build()
    s1 = scorer.score(events_a, result_a)
    s2 = scorer.score(events_a, result_a)
    np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-12,
                               err_msg=f"{name}: 同じ入力で出力がブレた")


@pytest.mark.parametrize('name,build', SCORERS)
def test_no_state_leak_between_inputs(name, build, analyzed):
    """別データで呼んだあと元データに戻しても結果が変わらない。"""
    events_a, result_a, events_b, result_b = analyzed
    scorer = build()

    s_a_baseline = scorer.score(events_a, result_a)
    # 別データを挟む（ステートを持つ実装ならここで内部が汚染される）
    scorer.score(events_b, result_b)
    s_a_after = scorer.score(events_a, result_a)

    np.testing.assert_allclose(
        s_a_baseline, s_a_after, rtol=1e-10, atol=1e-12,
        err_msg=f"{name}: 別データを挟むと結果がdriftした（state leak疑い）",
    )


@pytest.mark.parametrize('name,build', SCORERS)
def test_scorer_instance_attributes_immutable(name, build, analyzed):
    """score()呼び出しでインスタンス属性が増減しない。"""
    events_a, result_a, events_b, result_b = analyzed
    scorer = build()

    before = copy.deepcopy(vars(scorer))
    scorer.score(events_a, result_a)
    scorer.score(events_b, result_b)
    after = vars(scorer)

    # キーが変わってない（_cache, _fitted_ 等が増えていない）
    assert set(before.keys()) == set(after.keys()), (
        f"{name}: score()呼び出しで属性集合が変化した: "
        f"before={set(before.keys())} after={set(after.keys())}"
    )
    # 元の属性も値が変わっていない（dtype比較を避けるためrepr一致でチェック）
    for k in before:
        assert repr(before[k]) == repr(after[k]), (
            f"{name}: 属性 {k!r} がscore()で変わった"
        )


@pytest.mark.parametrize('name,build', SCORERS)
def test_independent_instances_agree(name, build, analyzed):
    """同設定のインスタンスを2つ作って結果が一致する（コンストラクタ純粋性）。"""
    events_a, result_a, _, _ = analyzed
    s1 = build().score(events_a, result_a)
    s2 = build().score(events_a, result_a)
    np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-12,
                               err_msg=f"{name}: 別インスタンス間で結果がずれた")


if __name__ == "__main__":
    # pytestなしで簡易確認
    events_a = _make_events(seed=1)
    events_b = _make_events(seed=2)
    det = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0); result_a = det.analyze(events_a)
    det2 = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0); result_b = det2.analyze(events_b)

    for name, build in SCORERS:
        scorer = build()
        s1 = scorer.score(events_a, result_a)
        scorer.score(events_b, result_b)
        s2 = scorer.score(events_a, result_a)
        ok = np.allclose(s1, s2, rtol=1e-10, atol=1e-12)
        print(f"  {name:<12}  stateless={'OK' if ok else 'LEAK!'}  "
              f"max|Δ|={np.max(np.abs(s1 - s2)):.2e}")
