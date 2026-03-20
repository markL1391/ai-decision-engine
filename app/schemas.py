from typing import List, Optional
from pydantic import BaseModel, Field

class AssessmentMeta(BaseModel):
    domain: str = "generic"
    notes: Optional[str] = None
    target_level: int = Field(default=2,ge=1)

class MetricInput(BaseModel):
    name: str
    value: float
    unit: str

class AssessmentAnalyzeRequest(BaseModel):
    assessment: AssessmentMeta
    metrics: List[MetricInput]

class HealthResponse(BaseModel):
    status: str

class AssessmentAnalyzeResponse(BaseModel):
    message: str
    assessment_id: int