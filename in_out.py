from pathlib import Path
from pydantic import BaseModel
from collections.abc import Sequence
import json


def write_records(records: Sequence[BaseModel], file_path: str | Path):
    path = Path(file_path)
    payload = [r.model_dump(mode="json") for r in records]
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(records)} records to file: {path}")


def read_records[T: BaseModel](model: type[T], file_path: str | Path) -> list[T]:
    path = Path(file_path)
    payload = json.loads(path.read_text())
    records = [model.model_validate(record) for record in payload]
    return records
