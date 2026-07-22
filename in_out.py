from pathlib import Path
from pydantic import TypeAdapter
from models import Record


def write_records(records: list[Record], file_path: str | Path):
    path = Path(file_path)
    path.write_bytes(TypeAdapter(list[Record]).dump_json(records, indent=2))
    print(f"Wrote to file: {path}")
