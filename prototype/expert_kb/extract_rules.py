from __future__ import annotations

import hashlib
import json
import time

from dotenv import load_dotenv

from prototype import config
from prototype.expert_kb.chunker import chunk_text
from prototype.expert_kb.schema import ExtractedRule, StoredRule
from prototype.llm_client import chat, extract_json

load_dotenv()

TRANSCRIPTS_DIR = config.TRANSCRIPTS_DIR
OUTPUT_PATH = config.RULES_PATH
MAX_WORDS = 2000

SYSTEM_INSTRUCTION = (
    "You are a knowledge engineer building a structured knowledge base from a "
    "transcript of an expert trading/investing course. Capture ALL reusable knowledge, "
    "at BOTH levels:\n"
    "  (a) MARKET rules: heuristics, causal/statistical relationships, trade actions.\n"
    "  (b) GENERAL rules: methodology, process, principles, mindset and key definitions "
    "(e.g. 'fundamentals generate ideas, technicals only time them', 'a bear market is a "
    "20% fall from the high', 'generate your own trade ideas, never copy-trade'). Much of "
    "this course is process/mindset - DO extract it, do not skip it.\n"
    "Ignore only filler, marketing, course logistics and anecdotes with no transferable insight.\n\n"
    "STRICT RULES:\n"
    "1. ATOMIC: one idea per rule, self-contained (understandable without surrounding text).\n"
    "2. NO DUPLICATES: if the speaker repeats the same idea, output it ONCE (the clearest form).\n"
    "3. rule_type drives direction:\n"
    "   - causal: a condition causes a directional market effect -> set direction bullish/bearish.\n"
    "   - action: what the trader should DO (buy/sell/hedge) -> direction = the resulting stance.\n"
    "   - relationship: a statistical link / leading indicator (e.g. 'X leads GDP', "
    "'positive correlation') -> direction = 'n/a' (a positive correlation is NOT 'bullish').\n"
    "   - definition: a term/measure definition -> direction = 'n/a'.\n"
    "   - methodology: process / framework / principle / mindset -> direction = 'n/a'.\n"
    "   For definition/methodology, condition and effect are usually empty ('').\n"
    "4. category: macro = top-down economy (gdp, rates, inflation, cycle, liquidity); "
    "micro = a single COMPANY's fundamentals ONLY; technical = price action/timing/price "
    "correlations; risk = sizing/hedging/drawdowns; framework = methodology/process/mindset/"
    "principles/definitions of the trading approach. "
    "A market-wide 'profit-taking / overextension' scenario is technical or macro, NOT micro.\n"
    "5. indicator: ONE short lowercase topic. For market rules use a canonical signal: gdp, "
    "interest rates, yield curve, inflation, monetary policy, liquidity, credit spreads, bond "
    "market, earnings, valuation, momentum, volatility, correlation, leading indicators. For "
    "GENERAL rules use a short concept: 'trade idea definition', 'risk management', "
    "'discipline', 'time horizon', 'bull/bear market', 'fundamentals vs technicals'. Reuse the "
    "SAME spelling across rules. Use '' only if truly none.\n"
    "6. region: 'us', 'eurozone', 'china', 'global' or '' if not geography-specific.\n"
    "7. time_horizon: copy any explicit horizon (e.g. '6 months', 'quarterly', '20-60 days').\n"
    "8. quote: a SHORT verbatim fragment (<=240 chars) actually present in the chunk.\n"
    "If a chunk contains no transferable knowledge, return an empty rules list."
)

SCHEMA_HINT = (
    'Return ONLY a JSON object of the form {"rules": [RULE, ...]} where each RULE has '
    "fields: "
    "statement (string), condition (string, '' if none), effect (string, '' if none), "
    'rule_type (one of "causal","relationship","action","definition","methodology"), '
    'direction (one of "bullish","bearish","neutral","n/a"; "n/a" for definitions/methodology), '
    "indicator (short lowercase topic: market signal OR general concept, '' if none), "
    'category (one of "macro","micro","technical","risk","framework","other"), '
    "asset_class (string, 'general' if not specific), "
    'region (one of "us","eurozone","china","global" or ""), '
    "time_horizon (string, '' if none), "
    'confidence (one of "low","medium","high"), '
    "quote (short verbatim quote). "
    'Return {"rules": []} if nothing is transferable.'
)

FEWSHOT = (
    "EXAMPLE (note how rule_type sets direction, and general rules are captured too):\n"
    '{"rules": [\n'
    '  {"statement": "The S&P 500 leads US GDP by about six months via positive correlation.",'
    ' "condition": "comparing equity index returns to GDP at a six-month lag",'
    ' "effect": "the index direction anticipates GDP direction", "rule_type": "relationship",'
    ' "direction": "n/a", "indicator": "leading indicators", "category": "macro",'
    ' "asset_class": "equities", "region": "us", "time_horizon": "6 months",'
    ' "confidence": "high", "quote": "S&P 500 leading US GDP with a six month time lag"},\n'
    '  {"statement": "When you predict GDP will fall, reduce equity exposure or go net short.",'
    ' "condition": "GDP is predicted to fall", "effect": "sell the index / net short stocks",'
    ' "rule_type": "action", "direction": "bearish", "indicator": "gdp", "category": "macro",'
    ' "asset_class": "equities", "region": "global", "time_horizon": "quarterly",'
    ' "confidence": "medium", "quote": "if we can predict that GDP will fall... We\'re selling the stock market"},\n'
    '  {"statement": "Trade ideas are generated from fundamentals; technical analysis is used only to time them.",'
    ' "condition": "", "effect": "", "rule_type": "methodology", "direction": "n/a",'
    ' "indicator": "fundamentals vs technicals", "category": "framework", "asset_class": "general",'
    ' "region": "", "time_horizon": "", "confidence": "high",'
    ' "quote": "Our trade ideas are rooted in fundamentals... we use technical analysis only to time our trades"},\n'
    '  {"statement": "A bear market is defined as a 20% fall in an index from its high.",'
    ' "condition": "", "effect": "", "rule_type": "definition", "direction": "n/a",'
    ' "indicator": "bull/bear market", "category": "framework", "asset_class": "equities",'
    ' "region": "", "time_horizon": "", "confidence": "high",'
    ' "quote": "A bear market is defined by a 20% fall in an index from its high"}\n'
    "]}"
)

PROMPT = (
    "{schema}\n\n{fewshot}\n\n"
    "Now extract the expert rules from the following transcript chunk.\n\n"
    "TRANSCRIPT CHUNK \n{chunk}"
)


def rule_id_for(source: str, chunk_index: int, statement: str) -> str:
    h = hashlib.sha1(f"{source}|{chunk_index}|{statement}".encode("utf-8")).hexdigest()
    return h[:12]


def dedup_key(rule: ExtractedRule) -> str:
    norm = " ".join(rule.statement.lower().split())
    norm = "".join(ch for ch in norm if ch.isalnum() or ch == " ")
    return f"{rule.indicator}|{norm}"


def extract_chunk(chunk: str) -> list[ExtractedRule]:
    raw = chat(
        system=SYSTEM_INSTRUCTION,
        user=PROMPT.format(schema=SCHEMA_HINT, fewshot=FEWSHOT, chunk=chunk),
        json_mode=True,
        temperature=0.1,
        max_tokens=4096,
    )

    data = extract_json(raw)
    items = data.get("rules", []) if isinstance(data, dict) else (data or [])
    rules: list[ExtractedRule] = []
    for item in items:
        try:
            rules.append(ExtractedRule(**item))
        except Exception as e:
            print(e)
            continue
    return rules


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"error no .txt files in {TRANSCRIPTS_DIR}")
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    seen: set[str] = set()
    total_rules = 0
    total_dups = 0
    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for path in files:
            chunks = chunk_text(path.read_text(encoding="utf-8"), max_words=MAX_WORDS)
            print(f"\n {path.name} ({len(chunks)} chunks)")
            for ci, chunk in enumerate(chunks):
                kept = 0
                for r in extract_chunk(chunk):
                    key = dedup_key(r)
                    if key in seen:
                        total_dups += 1
                        continue
                    seen.add(key)
                    stored = StoredRule(
                        **r.model_dump(),
                        rule_id=rule_id_for(path.name, ci, r.statement),
                        source_video=path.name,
                        chunk_index=ci,
                    )
                    out_f.write(json.dumps(stored.model_dump(), ensure_ascii=False) + "\n")
                    kept += 1
                out_f.flush()
                total_rules += kept
                print(f"  [chunk {ci}] +{kept} rules (total {total_rules})")
                time.sleep(0.5)

    print(f"\n DOEN! Wrote {total_rules} rules to {OUTPUT_PATH}, dropped {total_dups} duplicates")


if __name__ == "__main__":
    main()
