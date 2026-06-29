# Lambda³ NNNU — Future Work

Lambda³-S (Tier 0) と Lambda³-R (Tier 2) の **時間解像度拡張** と **空間次元拡張** に相当する 2 つの実装提案。本文書では実装方針・API 設計・思想的位置付けを記述する。具体実装はこのリポジトリには含まれていない (将来作業)。

---

## 0. 一貫した設計原理

NNNU (Neural Network Non-Use) の原理は単一:

> **正常を数学的に構造化し、構造からの逸脱を検知する。**

本ページで提案する 2 機能は、この原理の **時間解像度拡張** と **空間次元拡張** であり、新しい検出手法の追加ではない。既存の Tier 0 (streaming) / Tier 2 (regime-aware) と **同じ思想の実装バリエーション**として設計する。

異常パターンは **一切学習しない**。

```
NNNU 原理
   │
   ├── Tier 0 Streaming (現状)          ─ 短期 (frame-level) 検知
   ├── Tier 2 Regime-aware (現状)        ─ 中期 (regime-level) 検知
   │
   ├── Future: BatchDriftDetector       ─ 長期 (shift/day-level) 検知
   └── Future: MultiChannelDiagnostics  ─ 多 channel 統合・故障モード推定
```

---

## 1. BatchDriftDetector — 長期ドリフト検知

### 1.1 目的

ストリーミング scorer が苦手な **長期ドリフト** (季節変動、経年劣化、ベアリングの徐々の摩耗、化学プロセスの触媒劣化) を **定期バッチ** で検知する。

- リアルタイムアラートではなく **シフト終了時レポート**
- 突発異常 (Tier 0 が拾う) と責務分離
- NAB の `ambient_temperature_system_failure` や `realTweets/PFE,UPS` のような **「Tier 0 / Tier 2 では構造的に拾えない」失敗モード**への補完

### 1.2 思想的位置付け

「直前 N 時間の中央値」が「正常の構造化」に相当する。そこからの逸脱率を見ているだけで、異常パターンは一切学習しない。Tier 0 streaming の「window mean からの z-score」を、**時間スケールだけ massive に拡大** した実装バリエーション。

### 1.3 アルゴリズム

```
入力:
  events     : (n,) 1-D 時系列 (1 センサー分)
  timestamps : (n,) datetime 配列
  window_hours: int       (例: 8h、1 シフト)
  threshold_pct: float    (例: 10.0 = 10%)
  overlap     : float ∈ [0, 1)  (sliding 度、0 = 非重複)

処理 (現在時刻 t_now):
  now_start  = t_now - window_hours
  prev_start = t_now - 2 * window_hours
  prev_end   = t_now - window_hours

  seg_now    = events[ timestamps ∈ [now_start, t_now) ]
  seg_prev   = events[ timestamps ∈ [prev_start, prev_end) ]

  median_now  = np.median(seg_now)
  median_prev = np.median(seg_prev)
  change_rate = |median_now - median_prev| / (|median_prev| + ε)

  flagged = (change_rate > threshold_pct / 100.0)

出力 DriftReport:
  flagged     : bool
  change_rate : float
  median_now  : float
  median_prev : float
  window_start: datetime
  window_end  : datetime
  n_samples_now, n_samples_prev : int  (信頼度の補助指標)
```

### 1.4 sliding 実行イメージ

```
t=0h      t=8h      t=16h     t=24h
|---prev---|---now----|
           |---prev---|---now----|
                      |---prev---|---now----|

各 8h 境界で 1 回 check() を実行 → DriftReport を発行。
```

### 1.5 なぜ median か

- **スパイク (突発異常) に引っ張られない**: Tier 0 streaming が拾うべき短期 spike と責務分離。長期 baseline 比較は中央値が適切。
- **計算が軽い**: `np.median` は O(n log n)、毎時実行でも問題なし。
- **頑健**: 平均が outlier に敏感なのに対し、median は中央 50% で決まる。長期統計量として safe。

### 1.6 API 案

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DriftReport:
    flagged: bool
    change_rate: float
    median_now: float
    median_prev: float
    window_start: datetime
    window_end: datetime
    n_samples_now: int
    n_samples_prev: int


class BatchDriftDetector:
    """長期ドリフトの定期バッチ検知。NNNU 原理の時間解像度拡張。

    異常パターンは学習しない。
    中央値による「正常の構造化」からの逸脱率だけを見る。
    """

    def __init__(self,
                 window_hours: float,
                 threshold_pct: float,
                 overlap: float = 0.0,
                 eps: float = 1e-10):
        ...

    def check(self,
              events: np.ndarray,
              timestamps: np.ndarray,
              t_now: datetime | None = None) -> DriftReport:
        """t_now (省略時 = timestamps の最後) を「現在」として直前 2 ウィンドウを比較。"""
        ...

    def check_history(self,
                      events: np.ndarray,
                      timestamps: np.ndarray) -> list[DriftReport]:
        """series 全体に sliding 適用、各 boundary で DriftReport を返す。"""
        ...
```

### 1.7 多重スケール並列実行

複数 window_hours の並列インスタンスを想定:

```python
detectors = [
    BatchDriftDetector(window_hours=1,  threshold_pct=20),   # 短期 trend break
    BatchDriftDetector(window_hours=8,  threshold_pct=10),   # 1 shift
    BatchDriftDetector(window_hours=24, threshold_pct=5),    # 1 day
    BatchDriftDetector(window_hours=168, threshold_pct=3),   # 1 week (季節 drift)
]
reports = [d.check(events, timestamps) for d in detectors]
```

### 1.8 Tier 0 / Tier 2 との関係

- **独立に動く**。同じ events を食うが実行タイミングが違う (毎 frame vs シフト境界)。
- 出力先も違う: Tier 0/2 は **per-frame anomaly score**、本 detector は **shift-level DriftReport**。
- レポートであって即時停止判断には使わない。**オペレーターへの状況通知**。

### 1.9 期待される効果 (推測)

NAB 上で実装すれば、以下の失敗モードが救えると推測する (実機がないので推測のまま記述):

| File | 現状 (Tier 2) | 期待 |
|---|---|---|
| ambient_temperature_system_failure | 39.29 | drift 検知でシフト前半に baseline 変化を報告 → 補助情報として有用 |
| realTweets/PFE | 7.18 | 長期 trend break を別軸で報告 |
| realTweets/UPS | 17.64 | 同上 |

ただし NAB は per-frame label なので **本検出器を NAB score に直接統合するには評価方式の変更が必要**。本機能は実機 (継続稼働の産業ライン) を主要 target とする。

---

## 2. MultiChannelDiagnostics — 多 channel 故障モード推定

### 2.1 目的

独立した N 個の NNNU インスタンス (各センサー 1 つ) の出力を統合し、**どのセンサーの組み合わせが flag を立てたか** から **故障モード**を推定する。

産業例 (4 ch モーター):
- ch0 = vibration (振動)
- ch1 = temperature (温度)
- ch2 = strain (歪み)
- ch3 = current (電流)

flag の組み合わせから:
- `{vibration}` のみ → ベアリング劣化
- `{vibration, temperature}` → 潤滑系障害
- `{current, temperature}` → 過負荷
- 全 ch → 重大機械故障

### 2.2 思想的位置付け

**検知レイヤーと診断レイヤーの分離**:
- 各 NNNU は完全独立に検知する。
- MultiChannelDiagnostics は **検知結果だけを受け取る**。NNNU の内部には一切手を入れない。

NNNU 原理 (「正常の構造化からの逸脱」) は各 channel で個別に成立。組み合わせ → 故障モードへの translation はドメイン知識 (ルール) であり、これも anomaly pattern の学習ではない。

```
       channel 0          channel 1            channel N
       ┌──────┐           ┌──────┐             ┌──────┐
       │NNNU 0│           │NNNU 1│   ...       │NNNU N│
       └───┬──┘           └───┬──┘             └───┬──┘
           │                  │                    │
           │binary,score      │binary,score        │binary,score
           ▼                  ▼                    ▼
       ┌──────────────────────────────────────────────┐
       │      MultiChannelDiagnostics                 │
       │  (rule-based combination → fault mode)       │
       └──────────────────────┬───────────────────────┘
                              ▼
                      DiagnosisReport
                      (fault_mode, severity, flagged_channels)
```

### 2.3 故障モード推定 (ルールベース)

ML 不要、ルールベース。工場のドメイン知識を **明示的** に入れることで信頼性を確保。ブラックボックス化を避ける。

```python
FAULT_RULES = {
    frozenset(['vibration']):                          'bearing_degradation',
    frozenset(['vibration', 'temperature']):           'lubrication_failure',
    frozenset(['current']):                            'electrical_fault',
    frozenset(['current', 'temperature']):             'overload',
    frozenset(['vibration', 'current', 'temperature']):'critical_mechanical',
    frozenset(['strain']):                             'structural_deformation',
    # ... 工場側エンジニアが YAML/JSON で設定可能にする
}
```

**未知の組み合わせ** (テーブルにない) は `'unknown_fault'` として報告。これ自体が重要な情報 (新規故障モード candidate)。

### 2.4 API 案

```python
from dataclasses import dataclass

@dataclass
class NNNUResult:
    """各 channel の NNNU 出力 (Tier 0 / Tier 2 共通)。"""
    binary: np.ndarray         # (n,) 0/1
    score:  np.ndarray         # (n,) continuous (max-normalized)
    # 必要なら per_scorer dict / regimes など


@dataclass
class DiagnosisReport:
    flagged_channels: list[str]
    fault_mode:       str       # FAULT_RULES のキー、または 'unknown_fault' / 'normal'
    channel_scores:   dict[str, float]      # channel → 当該 frame の score
    confidence:       float     # = flag 立った channel 数 / 全 channel 数 (重症度)


class MultiChannelDiagnostics:
    """N 個の独立した NNNU の検知結果を統合し故障モードを推定。
    NNNU の内部には触れない (検知と診断の責務分離)。
    """

    def __init__(self,
                 channel_names: list[str],
                 fault_rules: dict[frozenset, str]):
        ...

    @classmethod
    def from_yaml(cls, channel_names, yaml_path):
        """fault_rules を YAML から読む (工場側で設定変更可)。"""
        ...

    def diagnose_frame(self,
                       channel_results: dict[str, NNNUResult],
                       t: int) -> DiagnosisReport:
        """frame t の全 channel 結果から 1 件の DiagnosisReport を返す。"""
        flagged = [ch for ch, r in channel_results.items() if r.binary[t]]
        key = frozenset(flagged)

        if not flagged:
            mode = 'normal'
        elif key in self.fault_rules:
            mode = self.fault_rules[key]
        else:
            mode = 'unknown_fault'

        return DiagnosisReport(
            flagged_channels=flagged,
            fault_mode=mode,
            channel_scores={ch: float(r.score[t]) for ch, r in channel_results.items()},
            confidence=len(flagged) / len(channel_results),
        )

    def diagnose_series(self,
                        channel_results: dict[str, NNNUResult]) -> list[DiagnosisReport]:
        """全 frame を一括処理。"""
        ...
```

### 2.5 BatchDriftDetector との統合

`BatchDriftDetector` の `DriftReport.flagged` も **1 つの channel** として MultiChannelDiagnostics に投入可能にする。長期 drift flag も故障モード推定に使える設計。

```python
# 例: drift channel を pseudo-NNNUResult として包む
drift_pseudo = NNNUResult(
    binary=drift_flagged_per_frame,  # DriftReport.flagged を expand
    score=drift_change_rate,         # 連続値として
)
channel_results['vibration_drift'] = drift_pseudo

# FAULT_RULES に "vibration_drift" を含むルールを追加可能
FAULT_RULES[frozenset(['vibration', 'vibration_drift'])] = 'progressive_bearing_failure'
```

### 2.6 並列実行特性

- 各 NNNU は **完全独立**、相互依存なし → embarrassingly parallel。
- N channel = N プロセス (or N GPU stream) で並列化可能。
- MultiChannelDiagnostics は per-frame O(N) で軽量。

### 2.7 期待されるユースケース

- **モーター監視** (Nidec / 製造ライン): 4 ch (vib/temp/strain/current) で故障モード予測
- **発電所タービン**: 振動 + 軸位置 + 温度 + 油圧
- **化学プラント**: 流量 + 温度 + 圧力 + pH
- **半導体 fab 装置**: パワー + ガス流量 + 温度 + 振動

NAB 単一 channel ベンチでは評価できないため、本機能は **実機ベンチマーク** で別途検証する想定。

---

## 3. 全体アーキテクチャ (実装後の姿)

```
                各センサー (N 個、独立)
                       │
   ┌───────────────────┼────────────────────┐
   │                   │                    │
   ▼                   ▼                    ▼
NNNU Streaming     NNNU Regime-aware    BatchDriftDetector
(Tier 0)            (Tier 2)            (Future §1)
per-channel独立     per-channel独立     per-channel独立
   │                   │                    │
   │ binary+score      │ binary+score       │ DriftReport
   │                   │                    │
   └─────────┬─────────┴────────────────────┘
             │
             ▼
   MultiChannelDiagnostics (Future §2)
             │
             ▼
   DiagnosisReport
     - flagged_channels: ['vibration', 'temperature']
     - fault_mode:       'lubrication_failure'
     - confidence:       0.50  (2/4 ch flag)
     - channel_scores:   {...}
```

### 3.1 思想の一貫性

全レイヤーが **「正常を構造化 → 逸脱を検知」** の NNNU 原理で動く:

| Layer | 「正常の構造化」 | 「逸脱検知」 |
|---|---|---|
| Tier 0 Streaming | 先頭 15% calibration 区間の per-scorer baseline | OR voting で raw/threshold > 1 |
| Tier 2 Regime-aware | clean data の GMM + per-regime trimmed percentile | OR voting で raw/threshold > 1 |
| Future BatchDrift | 直前 N 時間の median | 変化率 > X% |
| Future MultiChannel | (上位 NNNU の結果) | rule-based 組み合わせ → fault mode |

どのレイヤーも **異常パターンを学習しない**。BatchDrift は時間スケール、MultiChannel は channel 次元、それぞれの拡張軸が違うだけ。

---

## 4. 実装上の注意 (Claude Code セッションへの引き継ぎ用)

これらの機能を実装する場合の指針:

### 4.1 BatchDriftDetector

- 新規モジュール: `lambda3_detector/drift/batch_drift.py`
- 公開 API: `BatchDriftDetector`, `DriftReport` (dataclass)
- 既存 streaming / regime には **手を入れない**
- テスト: 1 つの NAB ファイル (例: `ambient_temperature_system_failure`) で `check_history()` を走らせて DriftReport の系列を `tests/test_batch_drift.py` で検証
- 統合 ablation は不要 (NAB score は別軸であるため)

### 4.2 MultiChannelDiagnostics

- 新規モジュール: `lambda3_detector/diagnostics/multi_channel.py`
- 公開 API: `MultiChannelDiagnostics`, `NNNUResult` (dataclass), `DiagnosisReport` (dataclass)
- 既存 `RegimeAwareDetector.fit_predict` の result dict から `NNNUResult` を構築するヘルパー `to_nnnu_result(result)` を提供
- `fault_rules` の YAML loader を `MultiChannelDiagnostics.from_yaml(channels, path)` として実装
- テスト: 4 ch ダミーデータで完全独立 NNNU を 4 つ走らせ、同期させて `diagnose_series()` を呼ぶ `tests/test_multi_channel.py`

### 4.3 思想的ガード

実装時に **絶対にやってはいけないこと**:
- 異常パターン (anomaly shape) を学習しない (DL 含む)
- 故障モード推定で ML 分類器を使わない (ルールベースのみ)
- 既存 Tier 0 / Tier 2 の内部に多 channel 統合ロジックを混ぜない (層分離)
- BatchDriftDetector を frame-level alarm として使わない (シフト境界の **report** のみ)

これらを破ると NNNU の honest framing が崩れる。

### 4.4 推奨実装順序

1. `BatchDriftDetector` (単独機能、テスト容易)
2. `NNNUResult` dataclass と `to_nnnu_result` ヘルパー
3. `MultiChannelDiagnostics` (rule-based core)
4. `MultiChannelDiagnostics.from_yaml`
5. 4 ch ダミーデータでの end-to-end テスト

各ステップ独立に commit 可能。

---

## 5. なぜ「今は」実装しないか

- **NAB benchmark で評価できない**: NAB は frame-level の per-instance 異常 label のみ、長期 drift report や fault mode 分類のラベルが無い。
- **実機データが必要**: 産業ライン (モーター、タービン、プラント) の継続稼働データと、故障モードラベル付きの periods が必要。
- **論文範囲が広がりすぎる**: 現状の Lambda³-R NAB 72.02 (HTM 超え) を主張する論文と、産業応用拡張は別論文にすべき。
- **本リポジトリの focus は research**: 産業実装は提携先 (e.g., Nidec) との別プロジェクトで実施が妥当。

これらが整った段階で、本ドキュメントを起点に実装を開始する。

---

## 6. 関連ドキュメント

- [architecture.md](architecture.md) — 現状の Tier 0 / Tier 2 詳細
- [scoreboard.md](scoreboard.md) — NAB 結果と experimental progression
- [abstract.md](abstract.md) — 論文 abstract draft
- [README.md](README.md) — doc/ navigation index
