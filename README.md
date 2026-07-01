# Expert-knowledge macro RAG

A proof of concept for the thesis: turn an expert investing course into a
structured rule base and use it, via retrieval-augmented generation, to read the
current macroeconomic regime and predict the direction of the US stock market

## Setup

Copy `prototype/.env.example` to `.env` and set `PLGRID_API_KEY`.

## Run

On the cluster use the SLURM wrappers in `slurm/`

## Layout

```
rag-poc/
  requirements.txt              local deps
  requirements-athena.txt       cluster deps
  transcription/
    transcribe.py               faster-whisper: videos/*.mp4 -> results/transcripts/
    requirements.txt            transcription-only deps
  slurm/                        Athena job scripts (one per pipeline stage)
  prototype/
    config.py                   all settings and output paths
    llm_client.py               PLGrid Forge chat client (+ JSON parsing helper)
    expert_kb/
      chunker.py                split a transcript into word-bounded chunks
      schema.py                 Pydantic schema of one atomic expert rule
      extract_rules.py          LLM extraction: transcripts -> rules.jsonl
      embeddings.py             sentence-transformers wrapper (Qwen3-Embedding)
      vector_store.py           LanceDB build + semantic retrieval
    macro/
      data_sources.py           FRED / DBnomics (not used finally) / S&P fetch
      indicators.py             deterministic data-to-text state encoding
      analyze.py                state -> retrieve rules -> LLM MacroVerdict
    eval/
      backtest.py               walk-forward backtest vs the S&P 500
  results/                      all generated artifacts (single output tree)
    transcripts/                transcript .txt files (extraction input)
    rules/rules.jsonl           extracted knowledge base 
    lancedb/                    vector index 
    llm_cache/                  cached quarterly LLM verdicts (backtest)
    backtest/                   backtest result CSVs
```
