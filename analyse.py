from in_out import read_records, write_records
from models import Record
from config import get_taxonomy_children
from typing import cast
import pandas as pd
import streamlit as st
import altair as alt

RECORDS_PATH = "assets/records.json"
CATEGORIES = sorted(get_taxonomy_children().keys())


@st.cache_data
def get_records() -> pd.DataFrame:
    """Row-level transactions with float amount and a 'YYYY-MM' month string."""
    records = read_records(Record, RECORDS_PATH)
    df = pd.DataFrame([r.model_dump() for r in records])
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df.dropna(subset=["category"])


def save_category_edits(edited: pd.DataFrame) -> int:
    """Persist category corrections back to the source file, keyed by id.

    Returns the number of records changed. Manual edits set confidence to 1.0
    to mark them as human-vetted rather than an LLM guess.
    """
    records = read_records(Record, RECORDS_PATH)
    new_category = dict(zip(edited["id"], edited["category"]))
    children = get_taxonomy_children()  # child -> parent

    changed = 0
    for r in records:
        category = new_category.get(r.id)
        if category is not None and category != r.category:
            r.category = category
            r.parent = children.get(category)
            r.confidence = 1.0
            changed += 1

    if changed:
        write_records(records, RECORDS_PATH)
    return changed


df = get_records()

monthly = df.groupby(["category", "month"]).agg(spend=("amount", "sum")).reset_index()
monthly["spend"] = monthly["spend"].abs()

categories = st.multiselect(
    "Choose categories",
    sorted(monthly["category"].unique()),
    ["Maintenance", "Accommodation"],
)

if not categories:
    st.error("Please select at least one category.")
else:
    data = monthly[monthly["category"].isin(categories)]

    # Point selection bound to the two fields we drill down on
    pick = alt.selection_point(fields=["category", "month"], name="pick")

    base = alt.Chart(data).encode(
        x=alt.X("month:O", title="Month"),
        y=alt.Y("spend:Q", title="Amount ($AUD)", stack=None),
        color=alt.Color("category:N", title="Category"),
    )
    area = base.mark_area(opacity=0.25)
    line = base.mark_line(strokeWidth=2)
    # Markers on the line at each month are the click targets and carry the
    # selection; the clicked one is emphasised, the rest sit quietly on the line
    markers = base.mark_point(filled=True).encode(
        size=alt.condition(pick, alt.value(140), alt.value(45)),
        tooltip=["category", "month", "spend"],
    ).add_params(pick)
    chart = area + line + markers

    event = st.altair_chart(chart, on_select="rerun", width="stretch")

    points = event["selection"].get("pick", [])
    if points:
        picks = pd.DataFrame(points)[["category", "month"]]
        detail = df.merge(picks, on=["category", "month"]).sort_values(by="date")
        st.subheader("Component transactions")

        edited = st.data_editor(
            detail[
                [
                    "id",
                    "date",
                    "amount",
                    "merchant",
                    "description",
                    "category",
                    "parent",
                    "confidence",
                ]
            ],
            column_config={
                "id": None,  # hidden, but kept so we can map edits back
                "category": st.column_config.SelectboxColumn(
                    "category", options=CATEGORIES, required=True
                ),
            },
            disabled=["date", "amount", "merchant", "description", "confidence"],
            hide_index=True,
            key="detail_editor",
        )

        if st.button("Save corrections"):
            changed = save_category_edits(cast(pd.DataFrame, edited))
            if changed:
                get_records.clear()
                st.success(f"Updated {changed} record(s).")
                st.rerun()
            else:
                st.info("No changes to save.")
    else:
        st.caption("Click a bar to see the transactions that make up its total.")
