import pandas as pd
from gemini import call_llm
from models import Categories, Record, RecordDto, ReclassRecordDtos


BANK_ADAPTORS = {
    "nab": {
        "columns": {
            "Date": "date",
            "Amount": "amount",
            "Account Number": "account",
            "Transaction Type": "type",
            "Transaction Details": "description",
            "Category": "category",
            "Merchant Name": "merchant",
        },
        "date_format": "%d %b %y",
    }
}

# Maps a parent category to the lowercase words identifying its children.
# A category is grouped if ANY space-separated word in it appears in the child set.
# Order matters: where a category matches several parents, the last one wins.
CATEGR_GROUPS = {
    "Transport": {"taxis", "ride", "transport", "public"},
    # 'Car' follows 'Transport' so 'Transport Tolls' groups as a car cost
    "Car": {"petrol", "fuel", "parking", "tolls", "vehicle", "registration"},
    "Dining": {"cafe", "cafes", "coffee", "restaurants", "takeaway", "dining"},
    "Alcohol": {"alcohol", "liquor", "bars", "nightlife"},
    "Health": {
        "medical",
        "pharmacy",
        "optometry",
        "radiology",
        "imaging",
        "health",
        "supplements",
        "gym",
        "fitness",
    },
    "Subscriptions": {
        "subscriptions",
        "streaming",
        "media",
        "software",
        "vpn",
        "apps",
        "content",
    },
    "Shopping": {
        "shopping",
        "marketplace",
        "stores",
        "clothing",
        "accessories",
        "homeware",
        "electronics",
        "technology",
        "department",
    },
    "Home": {"home", "improvement", "improvements", "handyman", "cleaning"},
    "Government": {"government", "council", "rates", "emergency"},
    "Transfers": {"transfers", "transfer"},
    "Travel": {"travel", "accommodation"},
    "Gifts": {"gifts", "florist"},
    "Events": {"attractions", "events", "weddings"},
    "Insurance": {"insurance"},
    "Fees": {"fees"},
    "Cash": {"cash"},
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


def convert_to_dtos(records: list[Record]) -> list[RecordDto]:
    return [RecordDto.model_validate(r) for r in records]


def reclassify_records(records: list[RecordDto]) -> ReclassRecordDtos:
    categrs = get_unique_categrs(records)
    print(f"Found {(len(categrs))} unique categories")

    opaque_categrs = find_opaque_categrs(categrs)
    print(f"Found the following opaque categories: {opaque_categrs}")

    opaque_records = get_records_by_categr(records, opaque_categrs.categrs)
    useable_categrs = set(categrs) - set(opaque_categrs.categrs)
    opaque_reclass = reclass_opaque_records(opaque_records, sorted(useable_categrs))
    print(f"Reclassified {len(opaque_reclass.dtos)} opaque records")

    new_categrs = set(get_unique_categrs(opaque_reclass.dtos)) - set(categrs)
    categrs_superset = sorted(useable_categrs | new_categrs)
    print(f"{len(new_categrs)} new categories added.")
    print(f"Full category list: {categrs_superset}")

    orphaned_records = find_orphaned_records(records)
    orphaned_reclass = classify_orphaned(orphaned_records, categrs_superset)
    print(f"Reclassified {len(orphaned_reclass.dtos)} orphaned records")

    reclass = ReclassRecordDtos(
        dtos=orphaned_reclass.dtos + opaque_reclass.dtos,
        assumptions=" ".join(
            [(orphaned_reclass.assumptions or ""), (opaque_reclass.assumptions or "")]
        ),
    )

    print(f"Reclassified total {len(reclass.dtos)} record.")
    print(f"Assumptions: {reclass.assumptions}")

    return reclass


def get_records_by_categr(
    records: list[RecordDto], categrs: list[str]
) -> list[RecordDto]:
    return [r for r in records if r.category in categrs]


def find_opaque_categrs(categrs: list[str]) -> Categories:
    prompt = f"""
    Identify any financial transaction categories in the following list that are too generic/opaque and could be broken into subcategories. e.g. 'Services', 'Transfers'. 
    {categrs}
    """
    return call_llm(prompt, Categories)


# Reclassify records with opaque categories, e.g. 'Services'
def reclass_opaque_records(
    records: list[RecordDto], categrs: list[str]
) -> ReclassRecordDtos:
    prompt = f"""
For the following transactions with generic/opaque categories, re-classify them
    into more informative subcategories.

    Reuse an existing category wherever one fits. Only invent a new category when
    nothing existing applies. Match the existing naming style (sentence case).

    Existing categories:
    {categrs}
    Transactions:
    {records}
    """

    response = call_llm(prompt, ReclassRecordDtos)
    return response


def get_unique_categrs(records: list[RecordDto]) -> list[str]:
    categrs = {t.category for t in records if t.category}
    return sorted(categrs)


def find_orphaned_records(records: list[RecordDto]) -> list[RecordDto]:
    orphaned = []
    for r in records:
        if r.category is None:
            orphaned.append(r)

    print(f"Found {len(orphaned)} uncategorised records")
    return orphaned


def classify_orphaned(
    records: list[RecordDto], categrs: list[str]
) -> ReclassRecordDtos:

    prompt = f"""
    Allocate each of these orphaned transactions to one of the categories in the list provided. 
    Provide confidence for each (0-1).

    Available categories:
    {categrs}
    Orphaned transactions:
    {records}
    """

    categorised = call_llm(prompt, ReclassRecordDtos)
    return categorised


def merge(dtos: ReclassRecordDtos, records: list[Record]) -> list[Record]:
    dto_by_ids = {d.id: d for d in dtos.dtos}
    for r in records:
        if d := dto_by_ids.get(r.id):
            r.category = d.category

    return records


def group_alike_categrs(records: list[Record]) -> list[Record]:
    for parent, children in CATEGR_GROUPS.items():
        for r in records:
            if r.category and any(
                word.lower() in children for word in r.category.split(" ")
            ):
                r.category = parent

    return records


def run_pipeline(df: pd.DataFrame) -> list[Record]:
    # Clean DataFrame to remove nan and sanitise cols
    df_clean = clean(df)

    # Parse as Record model to assign UUIDs
    records = parse_as_records(df_clean)

    # Parse as DTOs to remove confidential cols e.g. account, amount
    dtos = convert_to_dtos(records)

    # Reclassify records with poor categorisation
    dtos_reclass = reclassify_records(dtos)

    # Merge back into Records
    merged = merge(dtos_reclass, records)

    # Group record categories that are too granular
    grouped = group_alike_categrs(merged)

    return grouped
