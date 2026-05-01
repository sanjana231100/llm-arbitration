from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
import uuid


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ConfidenceLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class CriticVerdict(str, Enum):
    pass_ = "pass"
    fail = "fail"


class Flag(BaseModel):
    quote: str = Field(description="Exact quote from the original text that has the issue")
    issue: str = Field(description="Clear description of what is wrong")
    severity: Severity = Field(description="How serious this issue is")
    evidence: str = Field(description="Reasoning or evidence supporting this flag")


class CriticReport(BaseModel):
    critic_name: str = Field(description="Name of the critic agent")
    score: int = Field(ge=1, le=10, description="Quality score from 1 to 10")
    verdict: CriticVerdict = Field(description="Overall pass or fail verdict")
    flags: list[Flag] = Field(default_factory=list, description="List of specific issues found")
    reasoning: str = Field(description="Overall reasoning for the score and verdict")


class Disagreement(BaseModel):
    critic_a: str
    critic_b: str
    score_a: int
    score_b: int
    delta: int
    description: str


class ConfirmedIssue(BaseModel):
    quote: str
    issue: str
    severity: Severity
    evidence: str
    flagged_by: list[str]


class DismissedFlag(BaseModel):
    quote: str
    issue: str
    flagged_by: str
    dismissal_reason: str


class ArbitrationVerdict(BaseModel):
    overall_score: int = Field(ge=1, le=10)
    confidence: ConfidenceLevel
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    summary: str = Field(description="One paragraph assessment of the full output")


class ArbitrateRequest(BaseModel):
    llm_output: str = Field(min_length=10, description="The LLM output to arbitrate")
    original_prompt: Optional[str] = Field(default=None, description="The prompt that generated this output")


class ArbitrateResponse(BaseModel):
    arbitration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    verdict: ArbitrationVerdict
    critic_reports: list[CriticReport]
    disagreements: list[Disagreement]


class BatchArbitrateRequest(BaseModel):
    items: list[ArbitrateRequest] = Field(min_length=1, max_length=20)


class BatchArbitrateResponse(BaseModel):
    results: list[ArbitrateResponse]
