from __future__ import annotations

import json
import lancedb
from pathlib import Path
from typing import Optional
from prototype import config
from prototype.expert_kb.schema import StoredRule


DEFAULT_RULES_PATH = config.RULES_PATH


def rule_to_passage(r: StoredRule) -> str:
    parts = [r.statement.strip()]
    if r.condition:
        parts.append(f"Applies when: {r.condition}.")
    if r.effect:
        parts.append(f"Implication: {r.effect}.")
    if r.direction.value != "n/a":
        parts.append(f"Direction: {r.direction.value} for {r.asset_class or 'markets'}.")
    if r.indicator:
        parts.append(f"Topic: {r.indicator}.")
    if r.region:
        parts.append(f"Region: {r.region}.")
    if r.time_horizon:
        parts.append(f"Horizon: {r.time_horizon}.")
    return " ".join(parts)


def load_rules(rules_path: Path) -> list[StoredRule]:
    rules: list[StoredRule] = []
    for line in rules_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rules.append(StoredRule(**json.loads(line)))
    if not rules:
        raise ValueError(f"no rules in {rules_path}")
    return rules


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class RuleVectorStore:
    def __init__(self, uri: Optional[str] = None, table: Optional[str] = None, embedder=None):
        self.uri = uri or config.KB_LANCEDB_URI
        self.table_name = table or config.KB_TABLE
        db = lancedb.connect(self.uri)
        self._table = db.open_table(self.table_name)
        self._embedder = embedder

    @classmethod
    def build(cls, rules_path: Path = DEFAULT_RULES_PATH, uri: Optional[str] = None,
              table: Optional[str] = None, embedder=None) -> "RuleVectorStore":
        uri = uri or config.KB_LANCEDB_URI
        table = table or config.KB_TABLE
        embedder = embedder or Embedder()

        rules = load_rules(rules_path)
        passages = [rule_to_passage(r) for r in rules]
        print(f"[build] embedding {len(rules)} rules with {embedder.model_name} (dim={embedder.dim})")
        vectors = embedder.encode_documents(passages)

        rows = [
            {
                "vector": vec.tolist(),
                "rule_id": r.rule_id,
                "passage": passage,
                "statement": r.statement,
                "indicator": r.indicator,
                "category": r.category.value,
                "direction": r.direction.value,
                "rule_type": r.rule_type.value,
                "region": r.region,
                "asset_class": r.asset_class,
                "source_video": r.source_video,
                "rule_json": r.model_dump_json(),
            }
            for r, passage, vec in zip(rules, passages, vectors)
        ]

        db = lancedb.connect(uri)
        db.create_table(table, data=rows, mode="overwrite")
        print(f"[build] wrote {len(rows)} rules to {uri} / table '{table}'")
        return cls(uri=uri, table=table, embedder=embedder)

    def _ensure_embedder(self):
        if self._embedder is None:
            from prototype.expert_kb.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    def retrieve(self, query: str, k: int = 5,
                 category: Optional[str] = None) -> list[tuple[StoredRule, float]]:
        qvec = self._ensure_embedder().encode_query(query)
        search = self._table.search(qvec).metric("cosine").limit(k)
        if category:
            search = search.where(f"category = {_sql_str(category)}", prefilter=True)

        results = []
        for row in search.to_list():
            rule = StoredRule(**json.loads(row["rule_json"]))
            score = 1.0 - float(row.get("_distance", 0.0))
            results.append((rule, score))
        return results

    def __len__(self) -> int:
        return self._table.count_rows()


if __name__ == "__main__":
    RuleVectorStore.build()
