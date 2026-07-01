import os
from pathlib import Path

PROTOTYPE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROTOTYPE_DIR.parent

RESULTS_DIR = REPO_DIR / "results"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
RULES_PATH = RESULTS_DIR / "rules" / "rules.jsonl"
LLM_CACHE_DIR = RESULTS_DIR / "llm_cache"
BACKTEST_DIR = RESULTS_DIR / "backtest"

FORGE_BASE_URL = "https://llmlab.plgrid.pl/api/v1"
FORGE_API_KEY_ENV = "PLGRID_API_KEY"
FORGE_MODEL = os.environ.get("FORGE_MODEL", "google/gemma-4-31B")
KB_LANCEDB_URI = str(RESULTS_DIR / "lancedb")
KB_TABLE = "rules"

KB_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
KB_EMBED_BATCH = 32
KB_QUERY_INSTRUCTION = (
    "Given a current market situation or indicator state, retrieve the expert "
    "trading and macroeconomic rules that apply and explain its implication."
)

TOP_K = 6
