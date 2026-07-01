from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import pandas as pd

from prototype.macro import data_sources as _ds


@dataclass
class IndicatorState:
    key: str
    name: str
    as_of: date
    latest: float
    unit: str
    regime: str
    percentile: float
    chg_3m: Optional[float]
    chg_6m: Optional[float]
    chg_12m: Optional[float]
    trend: str
    regime_duration_months: float
    n_obs: int
    lookback_years: float
    query_text: str = ""
    report_text: str = ""
    extras: dict = field(default_factory=dict)


def _value_asof(series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    sub = series[series.index <= target]
    return float(sub.iloc[-1]) if len(sub) else None


def _percentile(values: pd.Series, x: float) -> float:
    if len(values) == 0:
        return float("nan")
    return 100.0 * float((values <= x).sum()) / float(len(values))


def _regime_duration_months(series: pd.Series, is_in_regime) -> float:
    count = 0
    for v in reversed(series.tolist()):
        if is_in_regime(v):
            count += 1
        else:
            break
    return round(count / 21.0, 1)  # +-21 trading days per month


def _consecutive_months(series: pd.Series, is_in_regime) -> int:
    count = 0
    for v in reversed(series.tolist()):
        if is_in_regime(v):
            count += 1
        else:
            break
    return count

def compute_yield_curve_state(
    series: pd.Series,
    as_of: Optional[date] = None,
    lookback_years: float = 20.0,
) -> IndicatorState:
    series = series.dropna().sort_index()
    asof_ts = pd.Timestamp(as_of) if as_of else series.index[-1]
    series = series[series.index <= asof_ts]
    if series.empty:
        raise ValueError("No yield-curve observations at or before as_of date")

    latest = float(series.iloc[-1])
    latest_date = series.index[-1]

    window = series[series.index >= (asof_ts - pd.DateOffset(years=int(lookback_years)))]
    pct = _percentile(window, latest)

    def _chg(months: int) -> Optional[float]:
        past = _value_asof(series, asof_ts - pd.DateOffset(months=months))
        return None if past is None else round(latest - past, 2)

    chg_3m, chg_6m, chg_12m = _chg(3), _chg(6), _chg(12)

    if chg_3m is None or abs(chg_3m) < 0.05:
        trend = "broadly flat"
    elif chg_3m > 0:
        trend = "rising"
    else:
        trend = "falling"

    if latest < 0:
        regime = "inverted"
        duration = _regime_duration_months(series, lambda v: v < 0)
    elif latest < 0.5:
        regime = "flat"
        duration = _regime_duration_months(series, lambda v: 0 <= v < 0.5)
    else:
        regime = "normal (upward sloping)"
        duration = _regime_duration_months(series, lambda v: v >= 0.5)

    state = IndicatorState(
        key="yield_curve",
        name="US Treasury yield curve (10y minus 2y)",
        as_of=latest_date.date(),
        latest=round(latest, 2),
        unit="percentage points",
        regime=regime,
        percentile=round(pct, 1),
        chg_3m=chg_3m,
        chg_6m=chg_6m,
        chg_12m=chg_12m,
        trend=trend,
        regime_duration_months=duration,
        n_obs=int(len(series)),
        lookback_years=lookback_years,
    )
    state.query_text = _yield_curve_query_text(state)
    state.report_text = _yield_curve_report_text(state)
    return state


def _yield_curve_query_text(s: IndicatorState) -> str:
    if s.regime == "inverted":
        shape = ("The US Treasury yield curve (10-year minus 2-year) is inverted: "
                 "short-term interest rates are higher than long-term rates.")
    elif s.regime == "flat":
        shape = ("The US Treasury yield curve (10-year minus 2-year) is flat: "
                 "long-term and short-term rates are nearly equal.")
    else:
        shape = ("The US Treasury yield curve (10-year minus 2-year) is upward sloping "
                 "(normal): long-term rates are higher than short-term rates.")
    slope = {"rising": "the spread has been rising (curve steepening)",
             "falling": "the spread has been falling (curve flattening)",
             "broadly flat": "the spread has been broadly stable"}[s.trend]
    return f"{shape} The 10y-2y spread is {s.latest} percentage points and {slope} over recent months."


def _yield_curve_report_text(s: IndicatorState) -> str:
    lines = [
        f"Indicator: {s.name}",
        f"As of: {s.as_of}",
        f"Current 10y-2y spread: {s.latest} {s.unit}  (regime: {s.regime})",
        f"Historical percentile (last {int(s.lookback_years)}y): {s.percentile:.0f}th",
        f"Change: 3m {s.chg_3m:+} pp | 6m {s.chg_6m:+} pp | 12m {s.chg_12m:+} pp"
        if None not in (s.chg_3m, s.chg_6m, s.chg_12m)
        else f"Change 3m: {s.chg_3m} pp",
        f"Recent trend: {s.trend}",
        f"Time in current regime: ~{s.regime_duration_months} months",
    ]
    return "\n".join(lines)

def _vix_band(v: float) -> str:
    if v < 15:
        return "low (calm)"
    if v < 20:
        return "normal"
    if v < 30:
        return "elevated"
    return "high (stress)"


def compute_vix_state(
    series: pd.Series,
    as_of: Optional[date] = None,
    lookback_years: float = 10.0,
) -> IndicatorState:
    series = series.dropna().sort_index()
    asof_ts = pd.Timestamp(as_of) if as_of else series.index[-1]
    series = series[series.index <= asof_ts]
    if series.empty:
        raise ValueError("No VIX observations at or before as_of date")

    latest = float(series.iloc[-1])
    latest_date = series.index[-1]

    window = series[series.index >= (asof_ts - pd.DateOffset(years=int(lookback_years)))]
    pct = _percentile(window, latest)

    def _chg(months: int) -> Optional[float]:
        past = _value_asof(series, asof_ts - pd.DateOffset(months=months))
        return None if past is None else round(latest - past, 1)

    chg_1m, chg_3m, chg_6m, chg_12m = _chg(1), _chg(3), _chg(6), _chg(12)

    if chg_1m is None or abs(chg_1m) < 1.0:
        trend = "broadly flat"
    elif chg_1m > 0:
        trend = "rising"
    else:
        trend = "falling"

    regime = _vix_band(latest)
    duration = _regime_duration_months(series, lambda v: _vix_band(v) == regime)

    state = IndicatorState(
        key="vix",
        name="US equity market volatility (VIX index)",
        as_of=latest_date.date(),
        latest=round(latest, 1),
        unit="index points",
        regime=regime,
        percentile=round(pct, 1),
        chg_3m=chg_3m,
        chg_6m=chg_6m,
        chg_12m=chg_12m,
        trend=trend,
        regime_duration_months=duration,
        n_obs=int(len(series)),
        lookback_years=lookback_years,
        extras={"chg_1m": chg_1m},
    )
    state.query_text = _vix_query_text(state)
    state.report_text = _vix_report_text(state)
    return state


def _vix_query_text(s: IndicatorState) -> str:
    band = {"low (calm)": "low (the market is calm)",
            "normal": "around normal levels",
            "elevated": "elevated",
            "high (stress)": "high, indicating market stress"}[s.regime]
    move = {"rising": "volatility has been rising recently",
            "falling": "volatility has been falling recently",
            "broadly flat": "volatility has been broadly stable recently"}[s.trend]
    return (f"US equity market volatility, measured by the VIX index, is {band} at "
            f"{s.latest} index points; {move}.")


def _vix_report_text(s: IndicatorState) -> str:
    chg_1m = s.extras.get("chg_1m")
    lines = [
        f"Indicator: {s.name}",
        f"As of: {s.as_of}",
        f"Current VIX: {s.latest} {s.unit}  (regime: {s.regime})",
        f"Historical percentile (last {int(s.lookback_years)}y): {s.percentile:.0f}th",
        f"Change: 1m {chg_1m:+} | 3m {s.chg_3m:+} | 6m {s.chg_6m:+} | 12m {s.chg_12m:+} (points)"
        if None not in (chg_1m, s.chg_3m, s.chg_6m, s.chg_12m)
        else f"Change 1m: {chg_1m} points",
        f"Recent trend (1m): {s.trend}",
        f"Time in current regime: ~{s.regime_duration_months} months",
    ]
    return "\n".join(lines)


def compute_real_rates_state(
    series: pd.Series,
    as_of: Optional[date] = None,
    lookback_years: float = 20.0,
) -> IndicatorState:

    series = series.dropna().sort_index()
    asof_ts = pd.Timestamp(as_of) if as_of else series.index[-1]
    series = series[series.index <= asof_ts]
    if series.empty:
        raise ValueError("No real-rate observations")

    latest = float(series.iloc[-1])
    latest_date = series.index[-1]

    window = series[series.index >= (asof_ts - pd.DateOffset(years=int(lookback_years)))]
    pct = _percentile(window, latest)
    p33, p66 = float(window.quantile(0.33)), float(window.quantile(0.66))

    def band(v: float) -> str:
        if v < 0:
            return "negative (very low)"
        if v < p33:
            return "low relative to history"
        if v < p66:
            return "around historical average"
        return "high relative to history"

    def _chg(months: int) -> Optional[float]:
        past = _value_asof(series, asof_ts - pd.DateOffset(months=months))
        return None if past is None else round(latest - past, 2)

    chg_3m, chg_6m, chg_12m = _chg(3), _chg(6), _chg(12)

    if chg_3m is None or abs(chg_3m) < 0.05:
        trend = "broadly flat"
    elif chg_3m > 0:
        trend = "rising"
    else:
        trend = "falling"

    regime = band(latest)
    duration = _regime_duration_months(series, lambda v: band(v) == regime)

    state = IndicatorState(
        key="real_rates",
        name="US 10-year real interest rate (10y TIPS yield)",
        as_of=latest_date.date(),
        latest=round(latest, 2),
        unit="percent",
        regime=regime,
        percentile=round(pct, 1),
        chg_3m=chg_3m,
        chg_6m=chg_6m,
        chg_12m=chg_12m,
        trend=trend,
        regime_duration_months=duration,
        n_obs=int(len(series)),
        lookback_years=lookback_years,
    )
    state.query_text = _real_rates_query_text(state)
    state.report_text = _real_rates_report_text(state)
    return state


def _real_rates_query_text(s: IndicatorState) -> str:
    band = {
        "negative (very low)": "negative",
        "low relative to history": "low relative to history",
        "around historical average": "around their historical average",
        "high relative to history": "high relative to history",
    }[s.regime]
    move = {"rising": "real rates have been rising recently",
            "falling": "real rates have been falling recently",
            "broadly flat": "real rates have been broadly stable recently"}[s.trend]
    return (f"US 10-year real interest rates (the 10-year TIPS yield) are {band} at "
            f"{s.latest}%; {move}.")


def _real_rates_report_text(s: IndicatorState) -> str:
    lines = [
        f"Indicator: {s.name}",
        f"As of: {s.as_of}",
        f"Current 10y real rate: {s.latest}%  (regime: {s.regime})",
        f"Historical percentile (last {int(s.lookback_years)}y): {s.percentile:.0f}th",
        f"Change: 3m {s.chg_3m:+} pp | 6m {s.chg_6m:+} pp | 12m {s.chg_12m:+} pp"
        if None not in (s.chg_3m, s.chg_6m, s.chg_12m)
        else f"Change 3m: {s.chg_3m} pp",
        f"Recent trend: {s.trend}",
        f"Time in current regime: ~{s.regime_duration_months} months",
    ]
    return "\n".join(lines)

def compute_pmi_state(
    series: pd.Series,
    as_of: Optional[date] = None,
    name: str = "US ISM Manufacturing PMI",
    kind: str = "manufacturing",
    lookback_years: float = 15.0,
) -> IndicatorState:

    series = series.dropna().sort_index()
    asof_ts = pd.Timestamp(as_of) if as_of else series.index[-1]
    series = series[series.index <= asof_ts]
    if series.empty:
        raise ValueError("No PMI observations at or before as_of date")

    latest = float(series.iloc[-1])
    latest_date = series.index[-1]

    window = series[series.index >= (asof_ts - pd.DateOffset(years=int(lookback_years)))]
    pct = _percentile(window, latest)

    def _chg(months: int) -> Optional[float]:
        past = _value_asof(series, asof_ts - pd.DateOffset(months=months))
        return None if past is None else round(latest - past, 1)

    chg_3m, chg_6m, chg_12m = _chg(3), _chg(6), _chg(12)

    if chg_3m is None or abs(chg_3m) < 0.5:
        trend = "broadly flat"
    elif chg_3m > 0:
        trend = "rising"
    else:
        trend = "falling"

    if latest >= 55:
        regime = "strong expansion"
    elif latest >= 50:
        regime = "expansion"
    elif latest >= 45:
        regime = "contraction"
    else:
        regime = "deep contraction"

    expanding = latest >= 50
    duration = _consecutive_months(series, lambda v: (v >= 50) == expanding)

    if latest >= 58:
        position = "near historic peak levels (~60)"
    elif latest <= 45:
        position = "near historic trough levels (40-45)"
    else:
        position = ""

    state = IndicatorState(
        key="pmi_mfg" if kind == "manufacturing" else "pmi_svc",
        name=name,
        as_of=latest_date.date(),
        latest=round(latest, 1),
        unit="index",
        regime=regime,
        percentile=round(pct, 1),
        chg_3m=chg_3m,
        chg_6m=chg_6m,
        chg_12m=chg_12m,
        trend=trend,
        regime_duration_months=float(duration),
        n_obs=int(len(series)),
        lookback_years=lookback_years,
        extras={"kind": kind, "sector": kind, "expanding": expanding, "position": position},
    )
    state.query_text = _pmi_query_text(state)
    state.report_text = _pmi_report_text(state)
    return state


def _pmi_query_text(s: IndicatorState) -> str:
    sector = s.extras.get("sector", "manufacturing")
    side = "above 50" if s.extras.get("expanding") else "below 50"
    economic = "expansion" if s.extras.get("expanding") else "contraction"
    move = {"rising": "the index has been rising",
            "falling": "the index has been falling",
            "broadly flat": "the index has been broadly stable"}[s.trend]
    pos = f" The reading is {s.extras['position']}." if s.extras.get("position") else ""
    return (f"The US ISM {sector.capitalize()} PMI is at {s.latest}, {side}, signaling "
            f"economic {economic} in the {sector} sector (a PMI above 50 indicates expansion, "
            f"below 50 contraction); {move} over recent months.{pos}")


def _pmi_report_text(s: IndicatorState) -> str:
    side = "above 50 (expansion)" if s.extras.get("expanding") else "below 50 (contraction)"
    lines = [
        f"Indicator: {s.name}",
        f"As of: {s.as_of}",
        f"Current PMI: {s.latest} {s.unit}  ({side}; regime: {s.regime})",
        f"Historical percentile (last {int(s.lookback_years)}y): {s.percentile:.0f}th",
        f"Change: 3m {s.chg_3m:+} | 6m {s.chg_6m:+} | 12m {s.chg_12m:+} (points)"
        if None not in (s.chg_3m, s.chg_6m, s.chg_12m)
        else f"Change 3m: {s.chg_3m} points",
        f"Recent trend: {s.trend}",
        f"Months on current side of 50: ~{int(s.regime_duration_months)}",
    ]
    if s.extras.get("position"):
        lines.append(f"Note: {s.extras['position']}")
    return "\n".join(lines)

@dataclass
class IndicatorSpec:
    name: str
    fetch: Callable[[], pd.Series]
    compute: Callable[[pd.Series, Optional[date]], IndicatorState]


REGISTRY: dict[str, IndicatorSpec] = {
    "yield_curve": IndicatorSpec(
        name="FRED:T10Y2Y (yield curve 10y-2y)",
        fetch=lambda: _ds.fetch_fred_series("T10Y2Y"),
        compute=compute_yield_curve_state,
    ),
    "vix": IndicatorSpec(
        name="FRED:VIXCLS (VIX)",
        fetch=lambda: _ds.fetch_fred_series("VIXCLS"),
        compute=compute_vix_state,
    ),
    "real_rates": IndicatorSpec(
        name="FRED:DFII10 (10y real rate)",
        fetch=lambda: _ds.fetch_fred_series("DFII10"),
        compute=compute_real_rates_state,
    ),
    # pmi is not used later- lack of data
    "pmi_mfg": IndicatorSpec(
        name="DBnomics:ISM/pmi (Manufacturing PMI)",
        fetch=lambda: _ds.fetch_dbnomics_series("ISM", "pmi", "pm", min_valid=20, max_valid=80),
        compute=lambda s, a=None: compute_pmi_state(
            s, a, name="US ISM Manufacturing PMI", kind="manufacturing"),
    ),
    "pmi_svc": IndicatorSpec(
        name="DBnomics:ISM/nm-pmi (Services PMI)",
        fetch=lambda: _ds.fetch_dbnomics_series("ISM", "nm-pmi", "pm", min_valid=20, max_valid=80),
        compute=lambda s, a=None: compute_pmi_state(
            s, a, name="US ISM Services PMI", kind="services"),
    ),
}
