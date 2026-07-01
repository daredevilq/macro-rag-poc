from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class Category(str, Enum):
    macro = "macro"
    micro = "micro"
    technical = "technical"
    risk = "risk"
    framework = "framework"
    other = "other"


class Direction(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"
    not_applicable = "n/a"


class RuleType(str, Enum):
    causal = "causal"
    relationship = "relationship"
    action = "action"
    definition = "definition"
    methodology = "methodology"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


def _norm_lower(value: str) -> str:
    return " ".join(value.strip().lower().split())


class ExtractedRule(BaseModel):
    statement: str = Field(
        description="The insight as ONE clear, self-contained sentence."
    )
    condition: str = Field(
        default="",
        description="The trigger / market situation in which the rule applies "
        "(e.g. 'long-term interest rates are rising'). Empty if it is a general principle.",
    )
    effect: str = Field(
        default="",
        description="The expected consequence when the condition holds "
        "(e.g. 'high-multiple growth/tech stocks tend to underperform').",
    )
    rule_type: RuleType = Field(
        default=RuleType.causal,
        description="causal | relationship | action | definition | methodology.",
    )
    direction: Direction = Field(
        default=Direction.not_applicable,
        description="Net effect on the affected asset WHEN the condition holds. "
        "Use 'n/a' (the default) for definitions, methodology, principles and pure "
        "statistical relationships (a positive correlation is NOT 'bullish').",
    )
    indicator: str = Field(
        default="",
        description="Short lowercase topic. A market signal (e.g. 'gdp', 'yield curve') "
        "OR a general concept for process/mindset rules (e.g. 'trade idea definition', "
        "'risk management', 'discipline', 'bull/bear market'). '' only if truly none.",
    )
    category: Category
    asset_class: str = Field(
        default="general",
        description="What it affects (e.g. 'equities', 'tech sector', 'usd', 'bonds'). "
        "Use 'general' if not specific.",
    )
    region: str = Field(
        default="",
        description="Geography it applies to, lowercase: 'us', 'eurozone', 'china', "
        "'global', or '' if not specific.",
    )
    time_horizon: str = Field(
        default="",
        description="Time scale the rule operates on, e.g. '6 months', 'quarterly', "
        "'20-60 days', 'long term'. Empty if not specified.",
    )
    confidence: Confidence = Field(
        default=Confidence.medium,
        description="How strongly the source asserts this rule.",
    )
    quote: str = Field(
        description="A short verbatim quote from the text supporting this rule (<=240 chars)."
    )

    @field_validator("indicator", "asset_class", "region")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _norm_lower(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _enforce_direction_semantics(self) -> "ExtractedRule":
        if self.rule_type in (RuleType.definition, RuleType.methodology):
            object.__setattr__(self, "direction", Direction.not_applicable)
        return self


class StoredRule(ExtractedRule):
    rule_id: str
    source_video: str
    chunk_index: int
