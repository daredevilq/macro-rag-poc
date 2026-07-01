from __future__ import annotations

import json
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field
from prototype import config, llm_client
from prototype.macro.indicators import REGISTRY, IndicatorState


class MacroVerdict(BaseModel):
    regime: str = Field(description="Economic regime implied: expansion | slowdown | contraction | recovery | unclear")
    equities_outlook: str = Field(description="bullish | bearish | neutral")
    bonds_outlook: str = Field(description="bullish | bearish | neutral")
    confidence: str = Field(description="low | medium | high")
    signal_agreement: str = Field(default="", description="do the indicators agree, conflict, or mixed?")
    key_drivers: list[str] = Field(default_factory=list)
    cited_rule_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


SYSTEM = (
    "You are a macro analyst. Interpret the CURRENT indicator states using ONLY the "
    "expert rules provided as evidence. Do not invent facts or use outside knowledge. "
    "Weigh the indicators jointly: macro is about the confluence of signals, so note "
    "explicitly whether they agree or conflict (`signal_agreement`). Cite the rule_id of "
    "every rule you rely on in `cited_rule_ids`. Output STRICT JSON only."
)


def _render_rules(hits: list[tuple]) -> str:
    lines = []
    for rule, score in hits:
        cond = f"when: {rule.condition} -> {rule.effect}" if (rule.condition or rule.effect) else "(general principle)"
        lines.append(
            f"[{rule.rule_id}] {rule.statement} ({cond}; direction: {rule.direction.value}; "
            f"indicator: {rule.indicator}; conf: {rule.confidence.value}) [sim={score:.2f}]"
        )
    return "\n".join(lines)


def synthesize(states: list[IndicatorState], hits_by_key: dict[str, list[tuple]]) -> tuple[MacroVerdict, str]:
    blocks = []
    for st in states:
        blocks.append(
            f"INDICATOR: {st.name}\nSTATE (point-in-time):\n{st.report_text}\n"
            f"RELEVANT EXPERT RULES:\n{_render_rules(hits_by_key[st.key])}"
        )
    schema_hint = (
        '{"regime": "...", "equities_outlook": "bullish|bearish|neutral", '
        '"bonds_outlook": "bullish|bearish|neutral", "confidence": "low|medium|high", '
        '"signal_agreement": "...", "key_drivers": ["..."], "cited_rule_ids": ["..."], '
        '"reasoning": "..."}'
    )
    user = "\n\n".join(blocks) + f"\n\nReturn JSON exactly in this shape:\n{schema_hint}"
    raw = llm_client.chat(SYSTEM, user, json_mode=True, temperature=0.1, max_tokens=1400)
    data = llm_client.extract_json(raw) or {}
    try:
        verdict = MacroVerdict(**data)
    except Exception as e:
        print(e)
        verdict = MacroVerdict(reasoning=f"[parse-failed] {raw[:500]}")
    return verdict, raw


INDICATORS = ["yield_curve", "vix", "real_rates", "pmi_mfg", "pmi_svc"]
CATEGORY = "macro"


def analyze(as_of: Optional[date] = None):
    from prototype.expert_kb.vector_store import RuleVectorStore

    store = RuleVectorStore()

    states: list[IndicatorState] = []
    hits_by_key: dict[str, list[tuple]] = {}
    for key in INDICATORS:
        spec = REGISTRY[key]
        print(f"\n[{key}] fetching {spec.name} ...")
        try:
            series = spec.fetch()
            state = spec.compute(series, as_of)
        except Exception as e:
            print(e)
            continue
        states.append(state)

        print("STATE: ")
        print(state.report_text)

        hits = store.retrieve(state.query_text, k=config.TOP_K, category=CATEGORY)
        hits_by_key[key] = hits
        print("RETRIEVED RULES: ")
        print(_render_rules(hits))

    if not states:
        raise SystemExit("No indicators produced state")

    print("\n[synthesis] LLM joint analysis over all indicators ...")
    verdict, _ = synthesize(states, hits_by_key)
    print("\nMACRO VERDICT: ")
    print(json.dumps(verdict.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
