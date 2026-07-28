"""Regenerate the committed sample data — fake, safe to publish. Two files:

    synthetic_expenses.csv   a raw NAB export, for exercising the pipeline
    synthetic_processed.csv  the same rows post-pipeline, loaded as demo data

    python assets/generate_synthetic.py

The processed file exists so a first-time visitor lands on a populated dashboard
without an LLM call. Its categories come from TAXONOMY_CATEGORY below — the
answer the classifier would reach — not from a real run, so regenerating costs
nothing and stays deterministic.

Mirrors a real NAB export rather than just the adaptor spec, so it exercises the
same paths ingest.py takes on live data: the same nine columns including the
unnamed blank one that normalise() drops, CRLF line endings, "%d %b %y" dates,
debits as negative amounts, ~20% of rows with no merchant resolved, and NAB's
own coarse Category guesses in NAB's casing ("Cafe & coffee", not the project
taxonomy's "Cafe & Coffee") — which is what leaves the LLM step real work to do.

Descriptions carry the trailing reference numbers and card/date prefixes the
bank puts there, since that noise is exactly what get_similarity() has to see
past when it groups alike transactions.

Seeded, so re-running reproduces the file byte for byte. Change the seed or the
merchant tables below to get a different set.
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# This script lives in assets/, so the project root is not on the path when it is
# run directly — config is what keeps the demo categories honest against TAXONOMY
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIDENCE_THRESHOLD, UNCATEGORISED, get_taxonomy_children  # noqa: E402

random.seed(20260728)

ACCOUNT = 512847391
CARD = "V8213"
START, END = date(2025, 7, 1), date(2026, 6, 30)

# merchant, nab_category, txn_type, suburb, (min, max) amount
CARD_MERCHANTS = [
    ("WOOLWORTHS 1234", "Groceries", "EFTPOS DEBIT", "CARNEGIE", (18, 145)),
    ("COLES SUPERMARKET", "Groceries", "EFTPOS DEBIT", "MALVERN EAST", (22, 160)),
    ("ALDI STORES", "Groceries", "EFTPOS DEBIT", "OAKLEIGH", (15, 95)),
    ("IGA XPRESS", "Groceries", "EFTPOS DEBIT", "MURRUMBEENA", (6, 34)),
    # The provisioning/consumption split, both sides of it. Same kind of shop,
    # told apart by basket size — which is the judgement call the classifier has
    # to make, so the demo should contain it.
    ("NIGHT OWL CONVENIENCE", "Groceries", "EFTPOS DEBIT", "ELSTERNWICK", (8, 38)),
    ("KOORNANG RD MILK BAR", "Groceries", "EFTPOS DEBIT", "CARNEGIE", (4, 12)),
    ("COKE VENDING MACHINE", "Uncategorised", "EFTPOS DEBIT", "MELBOURNE", (3, 7)),
    ("THE GRINDER CAFE", "Cafe & coffee", "EFTPOS DEBIT", "PRAHRAN", (4, 22)),
    ("SEVEN SEEDS COFFEE", "Cafe & coffee", "EFTPOS DEBIT", "CARLTON", (5, 19)),
    ("PATRICIA COFFEE", "Cafe & coffee", "EFTPOS DEBIT", "MELBOURNE", (4, 12)),
    ("LUNE CROISSANTERIE", "Cafe & coffee", "EFTPOS DEBIT", "FITZROY", (9, 28)),
    ("UBER *EATS", "Restaurants", "MISCELLANEOUS DEBIT", "SYDNEY", (24, 78)),
    ("MENULOG", "Restaurants", "MISCELLANEOUS DEBIT", "SYDNEY", (28, 65)),
    ("TONKA RESTAURANT", "Restaurants", "EFTPOS DEBIT", "MELBOURNE", (85, 240)),
    ("CHIN CHIN", "Restaurants", "EFTPOS DEBIT", "MELBOURNE", (60, 185)),
    ("DAN MURPHYS", "Alcohol", "EFTPOS DEBIT", "GLEN IRIS", (32, 130)),
    ("BWS LIQUOR", "Alcohol", "EFTPOS DEBIT", "CAULFIELD", (18, 68)),
    ("BP CONNECT", "Fuel", "EFTPOS DEBIT", "CHADSTONE", (55, 118)),
    ("AMPOL FOODARY", "Fuel", "EFTPOS DEBIT", "CLAYTON", (48, 105)),
    ("SHELL COLES EXPRESS", "Fuel", "EFTPOS DEBIT", "HAWTHORN", (52, 112)),
    ("UBER *TRIP", "Transport", "MISCELLANEOUS DEBIT", "SYDNEY", (12, 58)),
    ("DIDI MOBILITY", "Transport", "MISCELLANEOUS DEBIT", "MELBOURNE", (11, 44)),
    ("PTV MYKI TOPUP", "Transport", "EFTPOS DEBIT", "MELBOURNE", (20, 60)),
    ("WILSON PARKING", "Transport", "EFTPOS DEBIT", "MELBOURNE", (8, 42)),
    ("CHEMIST WAREHOUSE", "Medical", "EFTPOS DEBIT", "OAKLEIGH", (12, 88)),
    ("PRICELINE PHARMACY", "Personal care", "EFTPOS DEBIT", "CARNEGIE", (14, 62)),
    ("MALVERN MEDICAL CTR", "Medical", "EFTPOS DEBIT", "MALVERN", (45, 180)),
    ("SMILE DENTAL GROUP", "Medical", "EFTPOS DEBIT", "ARMADALE", (120, 480)),
    ("KMART", "Other shopping", "EFTPOS DEBIT", "CHADSTONE", (16, 120)),
    ("BUNNINGS WAREHOUSE", "Other shopping", "EFTPOS DEBIT", "PORT MELBOURNE", (22, 260)),
    ("OFFICEWORKS", "Other shopping", "EFTPOS DEBIT", "SOUTH MELBOURNE", (18, 145)),
    ("JB HI-FI", "Other shopping", "EFTPOS DEBIT", "CHADSTONE", (35, 620)),
    ("UNIQLO AUSTRALIA", "Clothing & accessories", "EFTPOS DEBIT", "MELBOURNE", (29, 180)),
    ("COTTON ON", "Clothing & accessories", "EFTPOS DEBIT", "CHADSTONE", (25, 95)),
    ("THE ICONIC", "Clothing & accessories", "MISCELLANEOUS DEBIT", "SYDNEY", (45, 240)),
    ("PETSTOCK", "Uncategorised", "EFTPOS DEBIT", "MOORABBIN", (24, 130)),
    ("LORT SMITH VET", "Uncategorised", "EFTPOS DEBIT", "NORTH MELBOURNE", (95, 420)),
    ("READINGS BOOKS", "Uncategorised", "EFTPOS DEBIT", "CARLTON", (18, 75)),
    ("HOYTS CINEMAS", "Uncategorised", "EFTPOS DEBIT", "CHADSTONE", (22, 58)),
    ("BARBER LANE", "Personal care", "EFTPOS DEBIT", "WINDSOR", (35, 55)),
]

# merchant, nab_category, day_of_month, amount — direct debits, same every month
RECURRING = [
    ("NETFLIX.COM", "Subscriptions", 3, 25.99),
    ("SPOTIFY AUSTRALIA", "Subscriptions", 7, 13.99),
    ("PRIME VIDE* PRIMEVIDEO", "Subscriptions", 12, 9.99),
    ("ANTHROPIC CLAUDE.AI", "Subscriptions", 15, 32.50),
    ("ADOBE CREATIVE CLOUD", "Subscriptions", 21, 29.99),
    ("GOODLIFE HEALTH CLUBS", "Gym & fitness", 5, 24.95),
    ("TELSTRA CORPORATION", "Uncategorised", 18, 65.00),
    ("ORIGIN ENERGY", "Uncategorised", 24, 148.40),
    ("AGL SALES", "Uncategorised", 9, 92.15),
    ("MEDIBANK PRIVATE", "Uncategorised", 2, 189.60),
    ("REAL ESTATE TRUST ACCT", "Uncategorised", 1, 2350.00),
]

QUARTERLY = [
    ("SOUTH EAST WATER", "Uncategorised", 14, 118.75, (7, 10, 1, 4)),
    ("RACV INSURANCE", "Uncategorised", 20, 412.30, (8, 11, 2, 5)),
]

FEES = [
    ("ACCOUNT SERVICE FEE", 5.00),
    ("INTERNATIONAL TRANSACTION FEE", 1.85),
    ("ATM OPERATOR FEE", 2.50),
]

# One-offs that give the dashboard some shape: a holiday, a car bill, Christmas.
ONE_OFFS = [
    (date(2025, 9, 12), 1840.00, "QANTAS AIRWAYS", "Accommodation", "MISCELLANEOUS DEBIT", "MASCOT"),
    (date(2025, 9, 14), 2260.55, "BOOKING.COM", "Accommodation", "MISCELLANEOUS DEBIT", "AMSTERDAM"),
    (date(2025, 9, 20), 430.20, "EUROPCAR ITALIA", "Uncategorised", "MISCELLANEOUS DEBIT", "ROMA"),
    (date(2025, 11, 8), 1285.00, "ULTRA TUNE SERVICE", "Uncategorised", "EFTPOS DEBIT", "MOORABBIN"),
    (date(2025, 12, 18), 620.40, "MYER", "Gifts", "EFTPOS DEBIT", "MELBOURNE"),
    (date(2025, 12, 21), 310.75, "DAVID JONES", "Gifts", "EFTPOS DEBIT", "MALVERN"),
    (date(2026, 2, 3), 890.00, "VICROADS REGISTRATION", "Uncategorised", "AUTOMATIC DRAWING", None),
    (date(2026, 4, 17), 1450.00, "MELBOURNE DENTAL SPEC", "Medical", "EFTPOS DEBIT", "EAST MELBOURNE"),
]

# Merchant -> project taxonomy category, the answer the LLM would have reached.
# This is what lets the demo file exist without an API call: the raw export gets
# NAB's coarse guess, the processed one gets the taxonomy category.
TAXONOMY_CATEGORY = {
    "WOOLWORTHS 1234": "Supermarket",
    "COLES SUPERMARKET": "Supermarket",
    "ALDI STORES": "Supermarket",
    "IGA XPRESS": "Convenience Store",
    "NIGHT OWL CONVENIENCE": "Convenience Store",  # stocking up
    "KOORNANG RD MILK BAR": "Snacks & Drinks",  # a drink and a Paddle Pop
    "COKE VENDING MACHINE": "Snacks & Drinks",
    "THE GRINDER CAFE": "Cafe & Coffee",
    "SEVEN SEEDS COFFEE": "Cafe & Coffee",
    "PATRICIA COFFEE": "Cafe & Coffee",
    "LUNE CROISSANTERIE": "Bakery",
    "UBER *EATS": "Food Delivery",
    "MENULOG": "Food Delivery",
    "TONKA RESTAURANT": "Restaurants & Takeaway",
    "CHIN CHIN": "Restaurants & Takeaway",
    "DAN MURPHYS": "Liquor Store",
    "BWS LIQUOR": "Liquor Store",
    "BP CONNECT": "Fuel",
    "AMPOL FOODARY": "Fuel",
    "SHELL COLES EXPRESS": "Fuel",
    "UBER *TRIP": "Taxis & Rideshare",
    "DIDI MOBILITY": "Taxis & Rideshare",
    "PTV MYKI TOPUP": "Public Transport",
    "WILSON PARKING": "Parking",
    "CHEMIST WAREHOUSE": "Medicine",
    "PRICELINE PHARMACY": "Grooming & Cosmetics",
    "MALVERN MEDICAL CTR": "Doctor",
    "SMILE DENTAL GROUP": "Dentist",
    "KMART": "Department & General",
    "BUNNINGS WAREHOUSE": "Hardware & DIY",
    "OFFICEWORKS": "Office Supplies",
    "JB HI-FI": "Electronics",
    "UNIQLO AUSTRALIA": "Clothing",
    "COTTON ON": "Clothing",
    "THE ICONIC": "Clothing",
    "PETSTOCK": "Pet Supplies",
    "LORT SMITH VET": "Vet",
    "READINGS BOOKS": "Books",
    "HOYTS CINEMAS": "Cinema",
    "BARBER LANE": "Haircuts",
    "NETFLIX.COM": "Streaming",
    "SPOTIFY AUSTRALIA": "Music",
    "PRIME VIDE* PRIMEVIDEO": "Streaming",
    "ANTHROPIC CLAUDE.AI": "AI Tools",
    "ADOBE CREATIVE CLOUD": "Software & Apps",
    "GOODLIFE HEALTH CLUBS": "Gym",
    "TELSTRA CORPORATION": "Mobile Phone",
    "ORIGIN ENERGY": "Electricity",
    "AGL SALES": "Gas",
    "MEDIBANK PRIVATE": "Health Insurance",
    "REAL ESTATE TRUST ACCT": "Rent",
    "SOUTH EAST WATER": "Water",
    "RACV INSURANCE": "Car Insurance",
    "QANTAS AIRWAYS": "Flights & Fares",
    "BOOKING.COM": "Accommodation",
    "EUROPCAR ITALIA": "Car Hire",
    "ULTRA TUNE SERVICE": "Servicing",
    "MYER": "Presents",
    "DAVID JONES": "Presents",
    "VICROADS REGISTRATION": "Registration",
    "MELBOURNE DENTAL SPEC": "Dentist",
    "ACCOUNT SERVICE FEE": "Account Keeping Fees",
    "INTERNATIONAL TRANSACTION FEE": "Foreign Transaction Fees",
    "ATM OPERATOR FEE": "ATM Fees",
}

# Transfers to your own accounts are internal; to other people, external
INTERNAL_PAYEES = {"SAVINGS ACCT", "J SACHS OFFSET"}

rows = []


def ref() -> str:
    """The trailing bank reference number — noise the pipeline has to look past."""
    return str(random.randint(10**10, 10**11 - 1))


def add(
    when: date,
    amount: float,
    details: str,
    category: str,
    txn_type: str,
    merchant: str | None,
    taxonomy: str,
):
    # NAB posts weekend card transactions on the next business day
    processed = when + timedelta(days=random.choice([0, 0, 0, 1, 1, 2]))
    rows.append(
        {
            "Date": when.strftime("%d %b %y"),
            "Amount": -round(amount, 2),
            "Account Number": ACCOUNT,
            "": None,  # the blank column normalise() drops
            "Transaction Type": txn_type,
            "Transaction Details": details,
            "Category": category,
            "Merchant Name": merchant,
            "Processed On": processed.strftime("%d %b %y"),
            # Stripped before the raw export is written — it is the answer, and a
            # raw bank file has no business carrying it
            "_taxonomy": taxonomy,
        }
    )


def title(name: str) -> str:
    """NAB's cleaned-up merchant name, e.g. 'WOOLWORTHS 1234' -> 'Woolworths'."""
    return " ".join(w.capitalize() for w in name.split() if not w.isdigit() and "*" not in w)


# --- everyday card spend, denser on weekends
day = START
while day <= END:
    n = random.choices([0, 1, 2, 3, 4], weights=[12, 30, 30, 18, 10])[0]
    if day.weekday() >= 5:
        n += random.choice([0, 1, 1, 2])
    for _ in range(n):
        name, category, txn_type, suburb, (lo, hi) = random.choice(CARD_MERCHANTS)
        amount = random.uniform(lo, hi)
        details = f"{CARD} {day.strftime('%d/%m')} {name} {suburb} {ref()}"
        # ~20% of rows have no merchant resolved, matching the real export
        merchant = None if random.random() < 0.2 else title(name)
        add(day, amount, details, category, txn_type, merchant, TAXONOMY_CATEGORY[name])
    day += timedelta(days=1)

# --- monthly direct debits, same merchant every month so grouping has work to do
day = START
while day <= END:
    for name, category, dom, amount in RECURRING:
        when = day.replace(day=min(dom, 28))
        if START <= when <= END:
            jitter = amount * random.uniform(-0.02, 0.02) if amount > 100 else 0
            add(when, amount + jitter, f"{name} {ref()}", category, "AUTOMATIC DRAWING",
                title(name), TAXONOMY_CATEGORY[name])
    for name, category, dom, amount, months in QUARTERLY:
        when = day.replace(day=min(dom, 28))
        if day.month in months and START <= when <= END:
            add(when, amount * random.uniform(0.9, 1.15), f"{name} {ref()}", category,
                "AUTOMATIC DRAWING", title(name), TAXONOMY_CATEGORY[name])
    day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)

# --- ATM withdrawals and transfers out
for _ in range(14):
    when = START + timedelta(days=random.randint(0, (END - START).days))
    amount = random.choice([50, 100, 100, 150, 200, 300])
    add(when, amount, f"ATM WDL {random.choice(['CARNEGIE', 'PRAHRAN', 'CBD MELBOURNE'])} {ref()}",
        "Uncategorised", "ATM DEBIT", None, "Cash Withdrawal")

for _ in range(22):
    when = START + timedelta(days=random.randint(0, (END - START).days))
    payee = random.choice(["SAVINGS ACCT", "J SACHS OFFSET", "RENT SHARE", "K MITCHELL"])
    add(when, random.choice([200, 250, 400, 500, 750, 1000]),
        f"Internet Transfer To {payee} {ref()}", "Transfers out", "TRANSFER DEBIT", None,
        "Internal Transfers" if payee in INTERNAL_PAYEES else "External Transfers")

# --- bank fees
day = START
while day <= END:
    name, amount = random.choice(FEES)
    add(day.replace(day=min(28, day.day)), amount, name, "Fees", "FEES", None,
        TAXONOMY_CATEGORY[name])
    day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)

for when, amount, name, category, txn_type, suburb in ONE_OFFS:
    details = f"{CARD} {when.strftime('%d/%m')} {name} {suburb} {ref()}" if suburb else f"{name} {ref()}"
    add(when, amount, details, category, txn_type, title(name), TAXONOMY_CATEGORY[name])

df = pd.DataFrame(rows)
# Newest first, the order NAB exports in
df = df.sort_values("Date", key=lambda s: pd.to_datetime(s, format="%d %b %y"), ascending=False)
df = df.reset_index(drop=True)

children = get_taxonomy_children()  # child category -> parent
unknown = sorted(set(df["_taxonomy"]) - set(children))
if unknown:
    raise SystemExit(f"Categories missing from TAXONOMY: {unknown}")

# --- the raw export: what a bank hands you, with the answer stripped out
raw_out = Path(__file__).with_name("synthetic_expenses.csv")
df.drop(columns=["_taxonomy"]).to_csv(raw_out, index=False, lineterminator="\r\n")  # CRLF, as NAB
print(f"Wrote {len(df)} rows to {raw_out}")

# --- the processed export: the same rows post-pipeline, in the shape analyse.py
# exports, so the app can load it as demo data with no LLM call.
#
# Confidence mirrors how the real classifier behaves rather than being uniform
# noise: mostly dead certain, sometimes hedging, occasionally unable to answer.
# Without the last two the demo would never show the review section.
categories, parents, confidences = [], [], []
for taxonomy_category in df["_taxonomy"]:
    roll = random.random()
    if roll < 0.008:  # returned no category at all
        categories.append(UNCATEGORISED)
        parents.append(UNCATEGORISED)
        confidences.append(None)
        continue

    categories.append(taxonomy_category)
    parents.append(children[taxonomy_category])
    if roll < 0.03:  # answered, but hedged — lands in the review queue
        confidences.append(round(random.uniform(0.45, CONFIDENCE_THRESHOLD - 0.01), 2))
    elif random.random() < 0.82:  # the model's habitual over-confidence
        confidences.append(1.0)
    else:
        confidences.append(round(random.uniform(CONFIDENCE_THRESHOLD + 0.01, 0.99), 2))

processed = pd.DataFrame(
    {
        "id": range(len(df)),  # positional, exactly as parse_as_records assigns
        "date": pd.to_datetime(df["Date"], format="%d %b %y").dt.strftime("%Y-%m-%d"),
        "amount": df["Amount"],
        "type": df["Transaction Type"],
        "description": df["Transaction Details"],
        "merchant": df["Merchant Name"],
        "account": df["Account Number"],
        "category": categories,
        "parent": parents,
        "confidence": confidences,
    }
)
processed_out = Path(__file__).with_name("synthetic_processed.csv")
processed.to_csv(processed_out, index=False)
print(f"Wrote {len(processed)} rows to {processed_out}")

print(f"\nTotal: ${-df['Amount'].sum():,.2f}  |  {df['Merchant Name'].isna().sum()} rows without a merchant")
print(f"{processed['category'].nunique()} categories across {processed['parent'].nunique()} parents")
unsure = processed["confidence"] < CONFIDENCE_THRESHOLD
print(
    f"needs review: {int((processed['category'] == UNCATEGORISED).sum())} uncategorised "
    f"+ {int(unsure.sum())} below {CONFIDENCE_THRESHOLD} confidence"
)
print(df["Transaction Type"].value_counts().to_string())
