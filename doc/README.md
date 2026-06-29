# Lambda³ Documentation

Paper / publication drafts and detailed benchmarking artifacts.

## Contents

- **[architecture.md](architecture.md)** — Complete system architecture: 6 streaming scorers (math + algorithms), Tier 0 / Tier 2 workflow, BIC K selection, trimmed percentile threshold, anomaly mask handling, module layout, data flow diagram, Lambda³ theory connection
- **[abstract.md](abstract.md)** — Paper abstract drafts (long/short/tagline) + title candidates + method paragraph + limitations
- **[scoreboard.md](scoreboard.md)** — Full NAB scoreboard comparison: per-category, per-file, Tier 0 vs Tier 2, experimental progression, honest framing, reproducibility recipe
- **[future_work.md](future_work.md)** — Future-work proposals: BatchDriftDetector (long-term drift, time-resolution extension) + MultiChannelDiagnostics (multi-channel fault mode, spatial-dimension extension). Implementation specs for a future Claude Code session, not yet implemented.

## Quick navigation

| Question | See |
|---|---|
| What's the NAB score? | [scoreboard.md §1](scoreboard.md#1-headline-result) — **72.02** |
| Per-category breakdown? | [scoreboard.md §2](scoreboard.md#2-per-category-results-lambda³-r-nab-7202-final-config) |
| Why these numbers (experiments)? | [scoreboard.md §5](scoreboard.md#5-experimental-progression) |
| Why dichotomous behavior? | [scoreboard.md §6](scoreboard.md#6-honest-framing-of-the-result) |
| How do the 6 scorers work? | [architecture.md §2](architecture.md#2-the-six-streaming-scorers) |
| How does the OR voting work? | [architecture.md §3.2](architecture.md#32-the-or-voting-integration) |
| How does Tier 2 use anomaly labels? | [architecture.md §4.5](architecture.md#45-semi-supervised-normal-label-only) |
| Connection to Lambda³ theory? | [architecture.md §11](architecture.md#11-connection-to-lambda³-theory-background) |
| Abstract for paper? | [abstract.md](abstract.md) (long / short / tagline) |
| Method paragraph? | [abstract.md `Method overview`](abstract.md#method-overview-paragraph) |
| Limitations? | [abstract.md `Limitations`](abstract.md#limitations-for-honest-paper) |
| Long-term drift detection (future)? | [future_work.md §1](future_work.md#1-batchdriftdetector--長期ドリフト検知) |
| Multi-channel fault diagnosis (future)? | [future_work.md §2](future_work.md#2-multichanneldiagnostics--多-channel-故障モード推定) |
| Why not implement now? | [future_work.md §5](future_work.md#5-なぜ今は実装しないか) |

## Reproducing the result

See [scoreboard.md §7](scoreboard.md#7-reproducibility) — single config across all 6 categories produces 72.02 ± 0.01 deterministically.
