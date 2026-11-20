"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, from the house's recorded reads before the checkpoint.

Model (written for this entry, stdlib only, deterministic):
  * Time is measured in log-time since launch midnight 2026-11-19T00:00Z:
    x = ln(1 + hours since launch).  Earn-rate growth saturates in x, so a
    straight line of logit(rate) against x is the trend model.
  * Per trophy, an OLS line of logit(rate) on x over the distinct reads.
    Its slope is damped (x0.65) and its move from the last read is capped
    at 2 logit units, so a short, noisy history cannot throw the median
    far from the last observed rate.
  * A second, dilution model: Exophase's tracked-player count N and the
    total trophies earned E both grow in log-time; if a trophy keeps its
    share of all earned trophies, its rate scales by (E_target/E_last) /
    (N_target/N_last).  This is what carries the forecast when the history
    is flat (launch night: every read identical, 638 players) -- a flat
    history does NOT mean rates stay flat, it means Exophase has not
    refreshed yet, and the next refresh dilutes every rate as new,
    less-progressed players flood in.  On the practice cases the refresh
    took players 638 -> 3,868 while earned trophies only grew ~4.5x, so
    rates fell to ~75% of their stale values; the flat-history refresh
    prior in forecast_case encodes exactly that jump, decaying as the
    launch matures, so a stale page forecasts a drop, not persistence.
  * Median: the trend model when the history actually moves, blended
    65/35 with the dilution model; the dilution model alone when flat.
  * Spread (logit sigma): the house random-walk term 0.5*sqrt(days), plus
    a large term when the history is flat (a refresh we have not seen is
    the dominant uncertainty), plus the trend fit's residual, plus a term
    growing with the horizon in days -- the final horizon is ~4.5 days,
    far longer than the 90-minute practice horizon, so its band must be
    much wider than the practice band.  Rare trophies get a small extra.
  * Tier caps keep medians sane (platinum is a week-one rarity, not a
    common trophy); a trophy already at 100% stays at 100%.

Usage: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input) and
writing $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Launch midnight; log-time is measured from here.
LAUNCH = "2026-11-19T00:00:00Z"

# Median caps by tier (percent): a week-one platinum cannot be common.
TIER_CAP = {"platinum": 3.0, "gold": 20.0, "silver": 50.0, "bronze": 99.5}
# Fallback median (percent) for a trophy never observed.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 2.0, "bronze": 1.0}


def parse_time(text):
    """ISO-8601 UTC string -> epoch seconds (no datetime import needed
    beyond the stdlib one; kept explicit for determinism)."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt.timestamp()


LAUNCH_EPOCH = parse_time(LAUNCH)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / n) if n else 0.0
    return a, b, sd


def log_time(at_epoch):
    """x = ln(1 + hours since launch midnight); >= 0 for reads after it."""
    hours = max(0.0, (at_epoch - LAUNCH_EPOCH) / 3600.0)
    return math.log1p(hours)


def forecast_case(case):
    horizon_days = case["horizon_minutes"] / 1440.0
    target_epoch = parse_time(case["target"])
    x_target = log_time(target_epoch)
    history = case["history"]
    trophies = case["trophies"]

    # --- Shared series: players tracked and total trophies earned. ---
    # Distinct (x, value) points, oldest first; duplicates collapse.
    def distinct_series(getter):
        pts = {}
        for read in history:
            v = getter(read)
            if v is None or v <= 0:
                continue
            pts[log_time(parse_time(read["at"]))] = v
        return sorted(pts.items())

    n_pts = distinct_series(lambda r: r.get("players_tracked"))
    e_pts = distinct_series(lambda r: r.get("trophies_earned"))
    x_last = log_time(parse_time(history[-1]["at"]))

    def growth_ratio(pts):
        """Predicted value at target / value at last read, from a
        log-log line; 1.0 when there is no usable trend."""
        if len(pts) < 2:
            return 1.0
        xs = [p[0] for p in pts]
        ys = [math.log(p[1]) for p in pts]
        a, b, _ = ols(xs, ys)
        if b <= 0:
            return 1.0
        ratio = math.exp(b * (x_target - x_last))
        return min(30.0, max(1.0, ratio))

    n_ratio = growth_ratio(n_pts)
    e_ratio = growth_ratio(e_pts)
    # Dilution factor for a trophy holding its share of earned trophies.
    share_factor = e_ratio / n_ratio if n_ratio > 0 else 1.0

    # Flat-history refresh prior.  When every read is identical, Exophase
    # has not refreshed since the first read: the page is stale, not the
    # truth.  The one refresh the kit documents (launch night, 638 ->
    # 3,868 players tracked, 943 -> 4,276 trophies earned, over the ~4.7 h
    # from the first probe to the 04:30 UTC refresh) says a refresh
    # multiplies tracked players ~6x and earned trophies ~4.5x while the
    # page looks frozen -- so rates fall to ~75% of their stale values.
    # That jump decays as launch matures (day-two growth is far slower
    # than launch-night growth), and a longer horizon spans more
    # refreshes, so the log-jump scales mildly with horizon.
    flat_history = len(history) >= 1 and all(
        read["players_tracked"] == history[0]["players_tracked"]
        and read["rates"] == history[0]["rates"]
        for read in history
    )
    flat_share_factor = share_factor
    if flat_history:
        hours_since_launch = max(
            0.0, (parse_time(case["at"]) - LAUNCH_EPOCH) / 3600.0
        )
        decay = math.exp(-hours_since_launch / 24.0)
        n_jump = 1.0 + 5.0 * decay
        e_jump = 1.0 + 3.5 * decay
        horizon_scale = min(2.0, math.sqrt(horizon_days / 0.0625))
        flat_share_factor = (e_jump / n_jump) ** horizon_scale

    rows = []
    for i, trophy in enumerate(trophies):
        # Per-trophy series of (x, logit(rate)), deduplicated by x.
        pts = {}
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            pts[log_time(parse_time(read["at"]))] = logit(rate)
        xs = sorted(pts)
        ys = [pts[x] for x in xs]

        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break

        tier = trophy.get("tier", "bronze")
        cap = TIER_CAP.get(tier, 99.5)

        if last_rate is None:
            median = TIER_PRIOR.get(tier, 1.0)
            sigma = 2.0
            rows.append((trophy["id"], median, sigma))
            continue

        if last_rate >= 99.99:
            # Everybody already has it; it stays earned by everybody.
            rows.append((trophy["id"], 100.0, 0.0))
            continue

        flat = (len(ys) < 2) or (max(ys) - min(ys) < 0.05)
        last_logit = logit(last_rate)

        # Dilution-model median: last rate scaled by the share factor
        # (the flat-history refresh prior when the page is stale, the
        # fitted N/E growth ratios when the history actually moves).
        factor = flat_share_factor if flat else share_factor
        share_median = clamp_pct(last_rate * factor)

        if flat:
            median = share_median
            resid_sd = 0.0
            has_trend = False
        else:
            a, b, resid_sd = ols(xs, ys)
            has_trend = len(xs) >= 3 and abs(b) > 1e-9
            if has_trend:
                delta = 0.65 * b * (x_target - x_last)
                delta = max(-2.0, min(2.0, delta))
                trend_median = inv_logit(last_logit + delta)
                # Blend in logit space: trend leads, dilution disciplines.
                median = inv_logit(
                    0.65 * logit(trend_median) + 0.35 * logit(share_median)
                )
            else:
                median = share_median

        # Tier cap (never below the last observed rate for a trophy that
        # is already above it -- the cap binds the forecast, not history).
        median = min(median, cap)
        median = clamp_pct(median)

        # --- Spread, in logit units. ---
        base = 0.5 * math.sqrt(horizon_days)
        flat_extra = 0.95 if flat else 0.0
        resid_extra = min(0.8, 2.0 * resid_sd) if has_trend else 0.0
        horizon_extra = 0.18 * horizon_days
        rare_extra = 0.15 if last_rate < 1.0 else 0.0
        sigma = math.sqrt(
            base * base
            + flat_extra * flat_extra
            + resid_extra * resid_extra
            + horizon_extra * horizon_extra
            + rare_extra * rare_extra
        )
        sigma = max(0.10, min(3.0, sigma))
        rows.append((trophy["id"], median, sigma))

    out_rows = []
    for tid, median, sigma in rows:
        if sigma == 0.0:
            qs = [100.0] * 5 if median >= 99.99 else [round4(median)] * 5
        else:
            m = logit(median)
            qs = []
            prev = None
            for z in Z:
                q = round4(clamp_pct(inv_logit(m + z * sigma)))
                if prev is not None and q < prev:
                    q = prev
                qs.append(q)
                prev = q
        out_rows.append({"id": tid, "q": qs})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
