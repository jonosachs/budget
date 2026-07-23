from pathlib import Path
from pydantic import BaseModel
from collections.abc import Sequence
import json


def write_records(records: Sequence[BaseModel], file_path: str | Path):
    path = Path(file_path)
    payload = [r.model_dump(mode="json") for r in records]
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(records)} records to file: {path}")
