from pydantic import BaseModel, ConfigDict
from decimal import Decimal
from datetime import date


class Record(BaseModel):
    id: int
    date: date
    amount: Decimal
    type: str
    description: str
    merchant: str | None = None
    account: int | None = None
    category: str | None = None


class RecordDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    description: str
    merchant: str | None = None
    category: str | None = None
    confidence: float | None = None


class ReclassRecordDtos(BaseModel):
    dtos: list[RecordDto]
    assumptions: str | None = None


class GroupedCategrs(BaseModel):
    groups: dict[str, list[str]]


class Analysis(BaseModel):
    total_spend: Decimal
    avg_spend_per_month: Decimal
    per_category_spend: dict[str, Decimal]
    per_category_perc: dict[str, float]
    assumptions: str | None = None


class Categories(BaseModel):
    categrs: list[str]
