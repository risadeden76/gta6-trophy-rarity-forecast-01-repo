# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/risadeden76/gta6-trophy-rarity-forecast-01-repo/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/risadeden76/gta6-trophy-rarity-forecast-01-repo/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python 3.13, standard-library-only program that forecasts every Grand Theft Auto VI (PS5) trophy's Exophase earn rate at the case target, as five quantiles per trophy, from the house's recorded reads before the checkpoint.

The model fits each trophy's logit earn rate against log-time since launch, blends that trend with a dilution model (a trophy keeps its share of all earned trophies while the tracked-player count grows), and — crucially — does not take a flat history at face value: on launch night Exophase's page sat frozen at 638 players while the real counts grew, and the first refresh cut every rate. A stale page therefore forecasts a dilution drop, not persistence, with a wide spread; the final's ~4.5-day horizon gets a much wider band than the 90-minute practice horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per case to the output directory. Deterministic: two runs write byte-identical output. No network, no dependencies.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ekkx10fx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
