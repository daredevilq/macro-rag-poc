from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prototype import config
from prototype.macro import data_sources as ds
from prototype.macro.analyze import synthesize
from prototype.macro.indicators import REGISTRY, _value_asof
from prototype.expert_kb.vector_store import RuleVectorStore
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

INDICATORS = ["yield_curve", "vix", "real_rates"]
START = "2004-01-01"
END = date.today().isoformat()
CATEGORY = "macro"
HORIZONS = [3, 6, 12]
NEUTRAL_POS = 0.5 # equity weight on a neutral call
PPY = 4 # quarterly grid

OUTLOOK_NUM = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
CONF_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}

BACKTEST_DIR = config.BACKTEST_DIR

def _llm_cache_path(dt: pd.Timestamp) -> Path:
    safe = config.FORGE_MODEL.replace("/", "_")
    d = config.LLM_CACHE_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dt.date()}_{'-'.join(INDICATORS)}.json"


def _llm_decision(states, hits_by_key, dt) -> tuple[str, float]:
    path = _llm_cache_path(dt)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        verdict, _ = synthesize(states, hits_by_key)
        data = verdict.model_dump()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    outlook = str(data.get("equities_outlook", "neutral")).lower().strip()
    if outlook not in ("bullish", "bearish", "neutral"):
        outlook = "neutral"
    conf = str(data.get("confidence", "medium")).lower().strip()
    score = OUTLOOK_NUM.get(outlook, 0.0) * CONF_WEIGHT.get(conf, 0.6)
    return outlook, score


def build_signal() -> pd.DataFrame:
    store = RuleVectorStore()
    series_by_key = {key: REGISTRY[key].fetch() for key in INDICATORS}

    rows = []
    for dt in pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="QE"):
        states, hits_by_key = [], {}
        for key in INDICATORS:
            try:
                st = REGISTRY[key].compute(series_by_key[key], dt.date())
            except Exception as e:
                print(e)
                continue
            states.append(st)
            hits_by_key[key] = store.retrieve(st.query_text, k=config.TOP_K, category=CATEGORY)

        if not states:
            continue
        outlook, score = _llm_decision(states, hits_by_key, dt)
        print(f"[{dt.date()}] outlook={outlook:<7} score={score:+.2f}")
        rows.append({"date": dt, "outlook": outlook, "score": score})

    return pd.DataFrame(rows).set_index("date")

def attach_returns(df: pd.DataFrame, sp: pd.Series) -> pd.DataFrame:
    last = sp.index[-1]
    df = df.copy()
    df["sp_close"] = [_value_asof(sp, dt) for dt in df.index]
    for h in HORIZONS:
        col = []
        for dt in df.index:
            c0 = _value_asof(sp, dt)
            future = dt + pd.DateOffset(months=h)
            c1 = _value_asof(sp, future)
            col.append((c1 / c0 - 1.0) if (c0 and c1 and future <= last) else float("nan"))
        df[f"fwd_{h}m"] = col
    return df


def attach_excess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rf = ds.fetch_fred_series("TB3MS", start="1990-01-01")
    for h in HORIZONS:
        exc = []
        for dt in df.index:
            r = df.at[dt, f"fwd_{h}m"]
            if pd.isna(r):
                exc.append(float("nan"))
                continue
            rfa = _value_asof(rf, dt) or 0.0
            rf_h = (1 + rfa / 100.0) ** (h / 12.0) - 1
            exc.append(float(r) - rf_h)
        df[f"exc_{h}m"] = exc
    return df


def forecast_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        sub = df.dropna(subset=[f"fwd_{h}m"])
        if sub.empty:
            continue
        ic = float(sub["score"].corr(sub[f"fwd_{h}m"], method="spearman"))
        out[h] = {"n": int(len(sub)), "IC_spearman": round(ic, 3)}
    return out


def classification_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        sub = df.dropna(subset=[f"exc_{h}m"])
        sub = sub[sub["outlook"].isin(["bullish", "bearish"])]
        if len(sub) < 5:
            continue
        y_true = (sub[f"exc_{h}m"] > 0).map({True: "up", False: "down"})
        y_pred = sub["outlook"].map({"bullish": "up", "bearish": "down"})
        out[h] = {
            "n_used": int(len(sub)),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 3),
            "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 3),
        }
    return out


def perf_stats(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    eq = (1 + returns).cumprod()
    yrs = len(returns) / PPY
    vol = float(returns.std() * math.sqrt(PPY))
    return {
        "CAGR": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
        "Sharpe": round(float((returns.mean() * PPY) / vol), 3) if vol > 0 else float("nan"),
        "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
    }


def strategy_returns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pos_map = {"bullish": 1.0, "neutral": NEUTRAL_POS, "bearish": 0.0}
    closes = pd.to_numeric(df["sp_close"], errors="coerce").dropna()
    market = closes.pct_change()
    pos = df["outlook"].map(pos_map).reindex(closes.index)
    strat = (pos.shift(1) * market).dropna()
    return strat, market.reindex(strat.index)

def run():
    df = build_signal()
    sp = ds.fetch_sp500_monthly(start="1995-01-01")
    df = attach_returns(df, sp)
    df = attach_excess(df)

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(INDICATORS)}_{START}_{END}"
    df.to_csv(BACKTEST_DIR / f"backtest_{tag}.csv")

    print("\n OUTLOOK DISTRIBUTION")
    print(df["outlook"].value_counts().to_string())

    print("\n FORECAST: Spearman IC")
    print(json.dumps(forecast_metrics(df), indent=2))

    print("\n CLASSIFICATION:")
    print(json.dumps(classification_metrics(df), indent=2))

    print("\n COMPARISON:")
    strat, bh = strategy_returns(df)
    closes = pd.to_numeric(df["sp_close"], errors="coerce").dropna()
    rng = np.random.default_rng(42)
    rand_pos = pd.Series(rng.integers(0, 2, len(closes)).astype(float), index=closes.index)
    rand = (rand_pos.shift(1) * closes.pct_change()).dropna()
    table = {
        "rag_llm": perf_stats(strat),
        "buy_hold": perf_stats(bh),
        "random": perf_stats(rand),
    }
    print(json.dumps(table, indent=2))
    print(f"\n results in {BACKTEST_DIR}")


if __name__ == "__main__":
    run()
