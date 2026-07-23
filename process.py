import pandas as pd
from gemini import call_llm
from models import Record, ReclassRecordDtos, RecordDtoIn, RecordDtoOut
from config import BANK_ADAPTORS, TAXONOMY, get_taxonomy_children
from in_out import write_records

CONFIDENCE_THRESHOLD = 0.9
SIMILARITY_THRESHOLD = 0.9
OUTPUT_PATH = "assets/record_dtos.json"
STOPWORDS = {
    "EFTPOS",
    "INTL",
    "TXN",
    "FEE",
    "MC",
    "AUS",
    "AU",
    "PTY",
    "LTD",
    "LIMITED",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Remove empty cols
    df = df.dropna(axis=1, how="all")

    # Sanitise/rename columns to match model
    df = df.rename(columns=BANK_ADAPTORS["nab"]["columns"])

    # Replace Uncategorised category with None
    df["category"] = df["category"].replace({"Uncategorised": None})

    # Make 'nan' python None type
    df = df.astype(object).where(pd.notna(df), None)

    # Parse dates
    df["date"] = pd.to_datetime(
        df["date"], format=BANK_ADAPTORS["nab"]["date_format"], errors="raise"
    )

    print(f"Obtained csv with {len(df)} entries")
    return df


def parse_as_records(df: pd.DataFrame) -> list[Record]:
    # Parse as Transaction model
    df_dict = df.to_dict(orient="records")
    records = [Record.model_validate({**row, "id": i}) for i, row in enumerate(df_dict)]

    return records


def convert_to_dtos(records: list[Record]) -> list[RecordDtoIn]:
    return [RecordDtoIn.model_validate(r) for r in records]


def categorise_records(records: list[RecordDtoIn]) -> list[RecordDtoOut]:
    categrs = tuple(get_taxonomy_children())

    categorised = categorise_with_llm(records, categrs)
    print(f"✅ Categorised {len(categorised.dtos)} records")
    print(f"Assumptions: {categorised.assumptions}")

    return categorised.dtos


def get_unique(records: list[RecordDtoIn], ids: list[int]) -> list[RecordDtoIn]:
    filtered = [r for r in records if r.id in ids]
    print(f"Filtered {len(filtered)} from total {len(records)} records.")
    return filtered


def get_similarity(records: list[RecordDtoIn]) -> dict[int, list[int]]:
    similarity_map = {}
    seen = []

    for record in records:
        if record.id in seen:
            continue

        record_desc = set(
            word.lower()
            for word in record.description.split(" ")
            if word.isalpha() and word.upper() not in STOPWORDS
        )

        if not record_desc:
            similarity_map[record.id] = []
            continue

        similar_records = []

        for comparison in records:
            if record.id == comparison.id or comparison.id in seen:
                continue

            compar_descr = set(
                word.lower()
                for word in comparison.description.split(" ")
                if word.isalpha() and word.upper() not in STOPWORDS
            )

            similarity = sum(word in compar_descr for word in record_desc) / max(
                len(record_desc), len(compar_descr)
            )

            if similarity < SIMILARITY_THRESHOLD:
                continue

            similar_records.append(comparison.id)
            seen.append(comparison.id)

        similarity_map[record.id] = similar_records

    return similarity_map


def categorise_with_llm(
    records: list[RecordDtoIn], categrs: tuple
) -> ReclassRecordDtos:
    prompt = f"""
    Allocate each of these transactions to one of the allowed categories.
    Provide confidence for each (0-1).
    Any records that already have categories are auto-generated and unvetted and should only be used as a fall-back guide to assist in placing in one of the allowed categories if other info is opaque.

    Allowed categories:
    {categrs}
    Orphaned transactions:
    {records}
    """

    categorised = call_llm(prompt, ReclassRecordDtos)
    return categorised


def merge(
    categorsd_dtos: list[RecordDtoOut],
    records: list[Record],
    similarity_map: dict[int, list[int]],
) -> list[Record]:
    records_by_id = {r.id: r for r in records}
    low_confidence: list[int] = []

    # Clear categories to avoid original labels persisting
    for r in records:
        r.category = None

    for cdto in categorsd_dtos:
        target_ids = [cdto.id, *similarity_map.get(cdto.id, [])]
        if (
            cdto.category is None
            or cdto.confidence is None
            or cdto.confidence < CONFIDENCE_THRESHOLD
        ):
            low_confidence.extend(target_ids)
            continue

        for rid in target_ids:
            if r := records_by_id.get(rid):
                r.category = cdto.category

    if low_confidence:
        print(
            f"The following couldn't be allocated due to low confidence:\n{low_confidence}"
        )

    return records


def generate_groups(records: list[Record]) -> dict[str, list[Record]]:
    children = get_taxonomy_children()
    output: dict[str, list[Record]] = {parent: [] for parent in TAXONOMY.keys()}
    for r in records:
        # If the record category is a child, add it to the parent category
        if r.category and (parent := children.get(r.category)):
            output[parent].append(r)

    return output


def run_pipeline(df: pd.DataFrame) -> list[Record]:
    # Clean DataFrame to remove nan and sanitise cols
    df_clean = clean(df)

    # Parse as Record model to assign ids
    records = parse_as_records(df_clean)

    # Parse as DTOs to remove confidential cols e.g. account, amount
    dtos = convert_to_dtos(records)

    # Group similar records to optimise llm evaluation step
    similarity_map = get_similarity(dtos)
    unique_dtos = get_unique(dtos, sorted(similarity_map.keys()))

    # Categorise records based on given taxonomy
    dtos_categrsd = categorise_records(unique_dtos)

    write_records(dtos_categrsd, OUTPUT_PATH)

    # Merge back into Records
    merged = merge(dtos_categrsd, records, similarity_map)

    return merged
