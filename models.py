from pydantic import BaseModel, ConfigDict
from decimal import Decimal
from datetime import date
from typing import Literal
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


CATEGORIES = tuple(get_taxonomy_children().keys())


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
    category: Literal[CATEGORIES] | None = None  # type: ignore[valid-type]
    confidence: float | None = None


class ReclassRecordDtos(BaseModel):
    dtos: list[RecordDtoOut]
    assumptions: str | None = None


class Analysis(BaseModel):
    total_spend: Decimal
    avg_spend_per_month: Decimal
    per_category_spend: dict[str, Decimal]
    per_category_perc: dict[str, float]
    assumptions: str | None = None
