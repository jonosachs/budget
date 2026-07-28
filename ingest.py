import json
import pandas as pd
from gemini import call_llm
from models import Record, ReclassRecordDtos, RecordDtoIn, RecordDtoOut
from config import (
    BANK_ADAPTORS,
    CONFIDENCE_THRESHOLD,
    EXCLUDED,
    TAXONOMY,
    UNCATEGORISED,
    get_taxonomy_children,
)
from in_out import read_records, write_records
from collections import Counter
import math

SIMILARITY_THRESHOLD = 0.9
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
BATCH_SIZE = 150


def normalise(df: pd.DataFrame, bank: str) -> pd.DataFrame:
    bank_adaptor = BANK_ADAPTORS.get(bank)
    if not bank_adaptor:
        raise ValueError(f"Unknown bank {bank!r}")

    # Remove empty cols
    df = df.dropna(axis=1, how="all")

    # Sanitise/rename columns to match model
    df = df.rename(columns=bank_adaptor["columns"])

    # Replace Uncategorised category with None
    df["category"] = df["category"].replace({"Uncategorised": None})

    # Make 'nan' python None type
    df = df.astype(object).where(pd.notna(df), None)

    # Parse dates
    df["date"] = pd.to_datetime(
        df["date"], format=bank_adaptor["date_format"], errors="raise"
    )

    print(f"Obtained csv with {len(df)} entries")
    return df


def parse_as_records(df: pd.DataFrame) -> list[Record]:
    df_dict = df.to_dict(orient="records")
    records = [Record.model_validate({**row, "id": i}) for i, row in enumerate(df_dict)]
    return records


def convert_to_dtos(records: list[Record]) -> list[RecordDtoIn]:
    return [RecordDtoIn.model_validate(r) for r in records]


def categorise_records(
    records: list[RecordDtoIn], taxonomy: dict[str, list[str]] = TAXONOMY
) -> list[RecordDtoOut]:
    categorised = categorise_with_llm(records, taxonomy)
    print(f"✅ Categorised {len(categorised.dtos)} records")
    print(f"Assumptions: {categorised.assumptions}")

    return categorised.dtos


def get_unique(records: list[RecordDtoIn], ids: list[int]) -> list[RecordDtoIn]:
    filtered = [r for r in records if r.id in ids]
    print(f"Filtered {len(filtered)} unique records.")
    return filtered


def get_similarity(records: list[RecordDtoIn]) -> dict[int, list[int]]:
    """Build groups of alike records based on description similarity"""

    similarity_map = {}
    seen = []

    # Pin each record for assessment one at a time
    for record in records:
        # If the record has been assessed already skip it to avoid duplication
        if record.id in seen:
            continue

        # Normalise and filter pinned record description
        record_desc = set(
            word.lower()
            for word in record.description.split(" ")
            if word.isalpha() and word.upper() not in STOPWORDS
        )
        # If the normalised description is empty this record should have
        # it's own category
        if not record_desc:
            similarity_map[record.id] = []
            continue

        # Compare pinned record against other records and keep a list of
        # those that are similar
        similar_records = []
        for comparison in records:
            # Avoid comparing to self or previously seen record
            if record.id == comparison.id or comparison.id in seen:
                continue

            # Normalise description of the compared record
            compar_descr = set(
                word.lower()
                for word in comparison.description.split(" ")
                if word.isalpha() and word.upper() not in STOPWORDS
            )

            # Calculate similarity using largest description length to
            # avoid misleading 100% similarity results
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
    records: list[RecordDtoIn], taxonomy: dict[str, list[str]], batch_size=BATCH_SIZE
) -> ReclassRecordDtos:
    prompt = """
    Allocate each of these transactions to one of the allowed categories.
    Provide confidence for each (0-1).
    Any records that already have categories are auto-generated and unvetted and should only be used as a fall-back guide to assist in placing in one of the allowed categories if other info is opaque."""
    categrs = json.dumps(taxonomy, indent=2)

    # Batch records for categorisation
    assumptions, dtos = [], []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        response = call_llm(
            f"{prompt}\nCategories:\n{categrs}\nRecords:\n{batch}",
            ReclassRecordDtos,
        )
        dtos.extend(response.dtos)
        if response.assumptions:
            assumptions.append(response.assumptions)

        print(
            f"Batch {i // batch_size + 1}/{math.ceil(len(records) / batch_size)} complete"
        )

    # Combine output into a single object
    return ReclassRecordDtos(dtos=dtos, assumptions=" ".join(assumptions))


def merge(
    categorsd_dtos: list[RecordDtoOut],
    records: list[Record],
    similarity_map: dict[int, list[int]],
) -> list[Record]:
    """Merge category data from DTOs back to their Record objects.

    Every record comes out categorised. A weak classification is kept and left
    for the user to confirm rather than thrown away: dropping it would silently
    understate their spend, and a suggestion to correct beats a blank field.
    Two things can go wrong, and they stay distinguishable afterwards:

        no category from the LLM  -> UNCATEGORISED, confidence untouched
        category below threshold  -> the suggestion, with its real confidence

    Both are what the dashboard's review section reads back. CONFIDENCE_THRESHOLD
    is now only a flag for "worth a human look", never a filter.
    """

    # Create id -> Record mapping so we can retrieve a Record object by its id
    records_by_id = {r.id: r for r in records}

    # Get children -> parent category map for assigning record parent categories
    children = get_taxonomy_children()

    # Start everything uncategorised so the bank's own guesses cannot survive,
    # and any record no DTO covers is visibly unclassified rather than stale
    for r in records:
        r.category = UNCATEGORISED
        r.parent = UNCATEGORISED
        r.confidence = None

    unclassified, low_confidence = 0, 0

    # Each record in categorsd_dtos represents a group of alike records
    for cdto in categorsd_dtos:
        # Create list of category member ids
        categ_member_ids = [cdto.id, *similarity_map.get(cdto.id, [])]

        if cdto.category is None:
            # The LLM declined to classify — leave the group as UNCATEGORISED
            unclassified += len(categ_member_ids)
            continue

        if cdto.confidence is None or cdto.confidence < CONFIDENCE_THRESHOLD:
            low_confidence += len(categ_member_ids)

        for categ_id in categ_member_ids:
            # if a Record id matches one of the ids in the category, retrieve
            # the Record and assign the category data from the DTO
            if r := records_by_id.get(categ_id):
                r.category = cdto.category
                r.confidence = cdto.confidence
                r.parent = children.get(cdto.category)

    if unclassified:
        print(f"{unclassified} record(s) left uncategorised")
    if low_confidence:
        print(f"{low_confidence} record(s) categorised below {CONFIDENCE_THRESHOLD} confidence")

    return records


def run_pipeline(df: pd.DataFrame, bank: str) -> list[Record]:
    # Normalise DataFrame to remove nan and sanitise cols
    df_normd = normalise(df, bank)

    # Parse as Record model to assign ids
    records = parse_as_records(df_normd)

    # Parse as DTOs to remove confidential cols e.g. account, amount
    dtos = convert_to_dtos(records)

    # Group similar records to optimise LLM evaluation step
    # (only one representative from each group needs to be assessed)
    similarity_map = get_similarity(dtos)
    unique_dtos = get_unique(dtos, sorted(similarity_map.keys()))

    # Categorise records based on given taxonomy using LLM
    dtos_categrsd = categorise_records(unique_dtos)

    # Merge back into Record objects
    merged = merge(dtos_categrsd, records, similarity_map)

    return merged


def reparent_records(path: str) -> int:
    """Rewrite every record's parent to match the current taxonomy.

    A maintenance pass, not part of run_pipeline. `parent` is denormalised onto
    each record, so moving a category under a different parent in TAXONOMY leaves
    every already-saved record pointing at the old one. Run this after any move.

    Returns the number of records changed. A category no longer in the taxonomy is
    reported and left alone rather than silently orphaned.
    """
    records = read_records(Record, path)
    children = get_taxonomy_children()

    changed, unknown = 0, Counter()
    for r in records:
        if r.category is None:
            continue
        # EXCLUDED sits outside the taxonomy on purpose, and is its own parent
        parent = EXCLUDED if r.category == EXCLUDED else children.get(r.category)
        if parent is None:
            unknown[r.category] += 1
        elif parent != r.parent:
            r.parent = parent
            changed += 1

    for category, n in unknown.most_common():
        print(f"⚠️  {n} record(s) in unknown category {category!r} — left as is")
    if changed:
        write_records(records, path)
    return changed
