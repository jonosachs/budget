from pydantic import BaseModel, ConfigDict, field_validator
from decimal import Decimal
from datetime import date
from config import get_taxonomy_children


class Record(BaseModel):
    id: int
    date: date
    amount: Decimal
    type: str
    description: str
    merchant: str | None = None
    account: int | None = None
    category: str | None = None
    parent: str | None = None
    confidence: float | None = None


CATEGORIES = tuple(get_taxonomy_children().keys())
CANONICAL = {val.lower(): val for val in CATEGORIES}


class RecordDtoIn(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    description: str
    merchant: str | None = None
    category: str | None = None
    confidence: float | None = None


class RecordDtoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    description: str
    merchant: str | None = None
    category: str | None = None
    confidence: float | None = None

    @field_validator("category")
    @classmethod
    def valid_category(cls, cat: str | None) -> str | None:
        if cat is None:
            return None
        canonical = CANONICAL.get(cat.lower())
        if canonical is None:
            print(f"❌ Unknown category {cat!r}")
        return canonical


class ReclassRecordDtos(BaseModel):
    dtos: list[RecordDtoOut]
    assumptions: str | None = None


class Analysis(BaseModel):
    total_spend: Decimal
    avg_spend_per_month: Decimal
    per_category_spend: dict[str, Decimal]
    per_category_perc: dict[str, float]
    assumptions: str | None = None
