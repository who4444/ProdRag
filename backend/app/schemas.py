from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    k_images: int = Field(default=5, ge=0, le=20)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    k: int = Field(default=4, ge=1, le=20)
    k_images: int = Field(default=1, ge=0, le=10)
