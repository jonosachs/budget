"""Streamlit dashboard for monthly spend analysis.

Three entry points:

    streamlit run analyse.py          # runs main()
    from analyse import launch        # start the server from ordinary Python
    from analyse import render        # embed the page in another Streamlit app

`render()` and the `render_*` sections need an active Streamlit runtime — they
draw into the current container and start nothing. `launch()` is what actually
opens a browser.

Records live in `st.session_state` and are never written to disk. The app is
multi-tenant by construction: each viewer uploads their own CSV, sees only their
own data, and leaves nothing behind. Anything user-supplied must therefore stay
out of `@st.cache_data`, whose cache is shared across every session.

A cold session opens on committed sample data, so a first-time visitor gets a
working dashboard to explore rather than an empty uploader; a notice says so, and
uploading anything replaces it. The uploader lives in the sidebar beside the
download from the start. It takes
either a raw bank export (which runs the pipeline) or a processed export from
this app (which loads straight back in) — see `is_processed`. Because a refresh
ends the session and its data with it, that download is the only way a viewer
can keep a result without paying for a second pipeline run.

Pipeline progress is streamed to the page by capturing its stdout, so ingest.py
needs no knowledge of Streamlit — see `StreamlitWriter`.

The interactive sections are cross-filtered through one shared `Selection` (see
the class docstring): a click on the heatmap or the trend narrows everything
else, and each section ignores only the dimension it is itself the source of.
The top-10 tables sit outside that — they are a fixed reference, not a control.

Importing this module draws nothing and starts nothing; the one thing it does do
is read `.env`, which has to happen before LOCAL_RECORDS. `st.set_page_config`
lives in `main()`
alone, since Streamlit allows it only once per page and only before any other
`st` call — an embedding app owns that decision, not this module.
"""

from in_out import read_records
from models import Record, RecordDtoIn
from config import (
    get_taxonomy_children,
    BANK_ADAPTORS,
    CONFIDENCE_THRESHOLD,
    EXCLUDED,
    UNCATEGORISED,
)
from dataclasses import dataclass, replace
from dotenv import load_dotenv
from pathlib import Path
from streamlit.delta_generator import DeltaGenerator
from typing import cast, ClassVar, Collection
import contextlib
import ingest
import io
import os
import pandas as pd
import streamlit as st
import altair as alt
import subprocess
import sys

# Derived from the model rather than listed, so adding a field to Record cannot
# quietly drop it from the export or from the round-trip check.
RECORD_FIELDS = tuple(Record.model_fields)
# A processed export is recognised by carrying every field Record demands. Raw
# bank exports use the bank's own capitalised headers, so they never collide.
EXPORT_COLUMNS = frozenset(
    name for name, field in Record.model_fields.items() if field.is_required()
)

# Must run before LOCAL_RECORDS is read below. gemini.py also calls this, but not
# until an LLM request is made — far too late for a constant resolved at import.
load_dotenv()

# Committed sample data, pre-categorised so a first visit lands on a working
# dashboard with no LLM call. Synthetic — see assets/generate_synthetic.py.
DEMO_RECORDS = "assets/synthetic_processed.csv"

# Set locally (in .env) to seed sessions from a file instead of the demo.
# Deliberately absent in deployment: user data lives in the session and nowhere else.
LOCAL_RECORDS = os.environ.get("LOCAL_RECORDS")

CATEGORIES = sorted(get_taxonomy_children().keys())
EDITABLE_CATEGORIES = CATEGORIES + [EXCLUDED]

# Sequential single-hue ramp, light -> dark. Reversed on a dark page so the
# lightest end always means "most spend" and near-zero recedes into the surface.
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
# Fixed hue order for series identity: assigned in order, never cycled or resorted,
# so a category keeps its colour as the selection changes.
CATEGORICAL = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Session state keys. Named so callers can seed or clear them before rendering.
SELECTION_KEY = "selection"  # the shared Selection, single source of truth
EPOCH_KEY = "selection_epoch"  # bumped on clear, see widget_key()
SEED_KEY = "explore_seed"  # last value pushed into the category picker
SAVES_KEY = "saves"  # bumped on every committed edit, see category_editor()
DATA_KEY = "records"  # the session's working frame — see load_frame()
IS_DEMO_KEY = "is_demo"  # whether that frame is the bundled sample data
ACCEPT_CLICK_KEY = "accept_click"  # {row, label} of the Accept button just clicked

# A ButtonColumn takes the cell value as its label. Width is pixels: the button
# is ~72px, so this leaves a little breathing room either side. It is a floor,
# not a ceiling — a stretched table adds its share of any surplus on top.
ACCEPT_LABEL = ":material/check: Accept"
ACCEPT_WIDTH = 110
DEFAULT_CATEGORIES = ()


# ------------------------------------------------------------------ selection


@dataclass(frozen=True)
class Selection:
    """What the user has clicked, shared by every section of the page.

    One immutable value in session state is the whole cross-filter model. Each
    section reads it to decide what to draw and writes back only the dimension it
    owns, so a click in any section reaches all the others. Frozen because change
    detection is just `!=` — see push().
    """

    parents: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    months: tuple[str, ...] = ()

    # Field -> the DataFrame column it constrains. Drives mask(), so adding a
    # dimension here is all it takes for every section to respect it.
    COLUMNS: ClassVar[dict[str, str]] = {
        "parents": "parent",
        "categories": "category",
        "months": "month",
    }

    def __bool__(self) -> bool:
        return any(getattr(self, field) for field in self.COLUMNS)

    def mask(self, df: pd.DataFrame, ignore: Collection[str] = ()) -> pd.Series:
        """True where a row matches every active dimension bar the `ignore`d ones.

        A section ignores whatever it is the source of: filtering a chart by the
        clicks it produces would leave nothing to click next.
        """
        keep = pd.Series(True, index=df.index)
        for field, column in self.COLUMNS.items():
            values = getattr(self, field)
            if values and field not in ignore:
                keep &= df[column].isin(values)
        return keep

    def filter(self, df: pd.DataFrame, ignore: Collection[str] = ()) -> pd.DataFrame:
        """The rows mask() keeps."""
        return cast(pd.DataFrame, df[self.mask(df, ignore)])

    def describe(self) -> str:
        """Readable summary of what's active, for the selection bar."""
        return " · ".join(
            f"**{field.capitalize()}:** {', '.join(self.display(field))}"
            for field in self.COLUMNS
            if getattr(self, field)
        )

    def display(self, field: str) -> list[str]:
        """Selected values of one dimension, months turned back into 'Jul 25'."""
        values = getattr(self, field)
        if field == "months":
            return [pd.Period(m, freq="M").strftime("%b %y") for m in values]
        return list(values)


def get_selection() -> Selection:
    """The page's current selection, empty on first render."""
    return cast(Selection, st.session_state.setdefault(SELECTION_KEY, Selection()))


def clear_selection() -> None:
    """Reset the selection and every widget holding a selection of its own."""
    st.session_state[SELECTION_KEY] = Selection()
    st.session_state[EPOCH_KEY] = st.session_state.get(EPOCH_KEY, 0) + 1


def widget_key(name: str, *scope: object) -> str:
    """Widget key that changes when the selection is cleared or `scope` changes.

    Streamlit remembers a widget's selection against its key, and a dataframe
    remembers it as *row positions*. Both mean a widget whose contents changed
    must get a new key, or position 3 silently starts pointing at a different
    row. Passing the filters that produced the contents as `scope` does that.
    """
    epoch = st.session_state.get(EPOCH_KEY, 0)
    return ":".join([name, str(epoch), *map(str, scope)])


def push(source: str, raw: object, **dimensions: tuple[str, ...]) -> None:
    """Write a section's click into the shared selection, then rerun.

    Only a genuine change in this source's own event rewrites the shared state,
    so two sections that both set `categories` take turns rather than fighting
    on every rerun. The rerun is what makes the page interoperable: sections
    already drawn above this one get redrawn against the new selection.
    """
    state_key = widget_key(f"source:{source}")
    if st.session_state.get(state_key) == raw:
        return
    st.session_state[state_key] = raw

    current = get_selection()
    updated = replace(current, **dimensions)
    if updated != current:
        st.session_state[SELECTION_KEY] = updated
        st.rerun()


# ---------------------------------------------------------------- data access


def build_frame(records: list[Record]) -> pd.DataFrame:
    """Row-level transactions with a 'YYYY-MM' month and positive 'spend'.

    spend is -amount, so debits are positive and refunds net against them.

    The single place a frame is built, so an ingested CSV and a re-uploaded
    processed one cannot drift into slightly different shapes.
    """
    df = pd.DataFrame([r.model_dump() for r in records])
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_label"] = df["date"].dt.strftime("%b %y")
    df["year"] = df["date"].dt.year
    df["spend"] = -df["amount"]
    # Nothing is dropped. A record the classifier was unsure about is still the
    # user's money, so it counts towards every total and surfaces in the review
    # section instead — see needs_review().
    df["category"] = df["category"].fillna(UNCATEGORISED)
    df["parent"] = df["parent"].fillna(UNCATEGORISED)
    return df.reset_index(drop=True)


def needs_review(df: pd.DataFrame) -> pd.DataFrame:
    """Rows worth a human look: unclassified, or classified without conviction.

    Ordered by spend so the expensive uncertainty is dealt with first — the tail
    of a long list is rarely worth anyone's time, and a $4 coffee filed under the
    wrong category changes nothing.

    A null confidence is what "nobody has looked at this" means: merge() leaves it
    null when the classifier returned nothing, and every human action — an edit or
    an accept — writes 1.0. That is what lets an Uncategorised row be dismissed;
    without the isna() guard, confirming one could never clear it from the queue.
    """
    unvetted = df["confidence"].isna()
    unsure = df["confidence"].notna() & (df["confidence"] < CONFIDENCE_THRESHOLD)
    flagged = df[((df["category"] == UNCATEGORISED) & unvetted) | unsure]
    return cast(pd.DataFrame, flagged).sort_values("spend", ascending=False)


def group_similar(flagged: pd.DataFrame) -> dict[int, list[int]]:
    """Collapse alike rows onto one representative: {rep id: every id it stands for}.

    The same grouping the pipeline uses to avoid asking the LLM twice about the
    same merchant, applied to the same end for the person doing the review —
    three near-identical dinners are one decision, not three.

    Recomputed rather than carried through from ingest, because the queue also
    has to work for a CSV re-uploaded into a fresh session, where no map from the
    original run exists. It is cheap: this runs over the review subset, not the
    whole frame.

    Grouping is scoped to the flagged rows alone. Reaching further would let
    confirming one row silently rewrite a record already settled.
    """
    if flagged.empty:
        return {}

    # Sorted by spend on the way in, and get_similarity keeps the first record it
    # sees as the representative — so each group is headed by its largest.
    rows = flagged.astype(object).where(pd.notna(flagged), None).to_dict("records")
    dtos = [RecordDtoIn.model_validate(row) for row in rows]
    return {rep: [rep, *members] for rep, members in ingest.get_similarity(dtos).items()}


def load_frame() -> pd.DataFrame | None:
    """The session's working frame, or None until something has been uploaded.

    Records live in session state and nowhere else: one viewer's data is never
    written to disk and never reachable from another session. Note this rules
    out @st.cache_data for anything user-supplied — that cache is keyed on
    arguments and shared process-wide, so it would hand one viewer's
    transactions to the next.

    A cold session starts on the demo data so there is something to explore
    before uploading anything. LOCAL_RECORDS overrides that with a file, so
    local development doesn't mean re-uploading after every restart; it is unset
    in deployment. A seed that will not load warns and falls back to the demo
    rather than taking the page down — it is a convenience, not a dependency.
    """
    if DATA_KEY not in st.session_state:
        if LOCAL_RECORDS:
            try:
                store_frame(build_frame(read_seed(LOCAL_RECORDS)), demo=False)
            except (OSError, ValueError) as error:
                # Missing file, wrong format, or rows that fail validation. Also
                # catches pydantic's ValidationError, which subclasses ValueError.
                st.warning(f"Could not load LOCAL_RECORDS {LOCAL_RECORDS!r} — {error}")

        if DATA_KEY not in st.session_state and Path(DEMO_RECORDS).exists():
            store_frame(build_frame(to_records(pd.read_csv(DEMO_RECORDS))), demo=True)
    return st.session_state.get(DATA_KEY)


def read_seed(path: str) -> list[Record]:
    """Records from a seed file — a processed CSV export, or a records.json.

    CSV is checked first because it is what the app's own download button
    produces: the obvious file to point LOCAL_RECORDS at is one you exported
    from here, and that should just work without converting it back to JSON.
    """
    if Path(path).suffix.lower() == ".csv":
        return to_records(pd.read_csv(path))
    return read_records(Record, path)


def store_frame(df: pd.DataFrame, demo: bool = False) -> None:
    """Make `df` the session's data. `demo` drives the sample-data notice only."""
    st.session_state[DATA_KEY] = df
    st.session_state[IS_DEMO_KEY] = demo


def apply_category_edits(
    edited: pd.DataFrame, groups: dict[int, list[int]] | None = None
) -> int:
    """Fold category corrections into the session frame, keyed by id.

    Returns the number of rows changed. Manual edits set confidence to 1.0 to
    mark them as human-vetted rather than an LLM guess.

    With `groups`, an edited row stands for every alike row grouped behind it and
    the correction lands on all of them — see group_similar(). Without it each
    row speaks only for itself, which is what the other tables want: those list
    individual records, not representatives.
    """
    df = load_frame()
    if df is None:
        return 0

    children = get_taxonomy_children()  # child -> parent
    index_of = pd.Series(df.index, index=df["id"])

    changed = 0
    for record_id, category in zip(edited["id"], edited["category"]):
        # EXCLUDED has no taxonomy parent, so it is its own — without this
        # children.get() returns None and the row loses its parent entirely
        parent = EXCLUDED if category == EXCLUDED else children.get(category)
        for target in (groups or {}).get(record_id, [record_id]):
            i = index_of.get(target)
            if i is None or df.at[i, "category"] == category:
                continue
            df.at[i, "category"] = category
            df.at[i, "parent"] = parent
            df.at[i, "confidence"] = 1.0
            changed += 1

    return changed


def confirm_categories(record_ids: Collection[int]) -> int:
    """Mark rows as human-vetted, adopting the first id's category across the rest.

    The counterpart to apply_category_edits: that one records "this is wrong, it
    is X", this one records "this is right". Both write confidence 1.0, the
    single marker for "a person has ruled on this row", so both take the row out
    of the review queue.

    `record_ids` is a similarity group led by its representative — the row the
    user actually saw and agreed with. The rest adopt its category, because
    grouping is a claim that these are the same transaction: a table showing one
    category and "covers 3" has promised that all three end up there. Leaving a
    member on its own stale category would quietly break that promise.

    A single id is just the degenerate group, and settles only itself.
    """
    df = load_frame()
    if df is None or not (ids := list(record_ids)):
        return 0

    index_of = pd.Series(df.index, index=df["id"])
    rep = index_of.get(ids[0])
    if rep is None:
        return 0
    category, parent = df.at[rep, "category"], df.at[rep, "parent"]

    changed = 0
    for target in ids:
        i = index_of.get(target)
        # NaN != 1.0 is True, so never-vetted rows are picked up here as intended
        if i is None or (df.at[i, "category"] == category and df.at[i, "confidence"] == 1.0):
            continue
        df.at[i, "category"] = category
        df.at[i, "parent"] = parent
        df.at[i, "confidence"] = 1.0
        changed += 1

    return changed


def accept_clicked(row_groups: tuple[tuple[int, ...], ...]) -> None:
    """on_click handler for the review table's Accept column.

    A ButtonColumn reports the *position* of the row clicked, not its contents,
    so what each displayed row stands for is captured when the table is drawn and
    passed in — reconstructing it here would risk indexing a list that has since
    been reordered. Each entry is a whole similarity group, so one click settles
    every record behind that row.

    Runs as a callback rather than inline because Streamlit reruns of its own
    accord once a callback returns; doing it here means no st.rerun(), and no
    chance of the click being read twice.
    """
    click = st.session_state.get(ACCEPT_CLICK_KEY)
    if click is None:
        return

    row = click["row"]
    if 0 <= row < len(row_groups) and confirm_categories(row_groups[row]):
        st.session_state[SAVES_KEY] = st.session_state.get(SAVES_KEY, 0) + 1


# ------------------------------------------------------------------- ingest


class StreamlitWriter(io.TextIOBase):
    """A stdout stand-in that renders each completed line into a container.

    The pipeline already narrates itself with print(), so capturing stdout is
    what makes it live on the page — ingest.py stays a plain module with no
    Streamlit import and no second progress protocol to keep in sync.

    Buffering to the newline is required, not cosmetic: print("a", "b") issues
    four separate write() calls ("a", " ", "b", "\\n"), so passing each one
    straight through would scatter one message across four rendered lines.
    """

    def __init__(self, container: DeltaGenerator) -> None:
        self.container = container
        self.buffer = ""

    def write(self, s: str) -> int:
        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.container.write(line)
        return len(s)


def is_processed(df: pd.DataFrame) -> bool:
    """Has this CSV already been through the pipeline?

    Detected by shape rather than by asking the user, since getting the answer
    wrong is expensive in one direction: re-running the LLM over an already
    categorised file costs money and a quota slot for no benefit.
    """
    return EXPORT_COLUMNS.issubset(df.columns)


def to_records(df: pd.DataFrame) -> list[Record]:
    """Rebuild Records from a processed export, reusing the model for parsing.

    Dates and amounts come back as strings from CSV; Record's own validation is
    what turns them into date and Decimal, so a round-tripped file lands in
    exactly the state an ingest would have produced.
    """
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    return [Record.model_validate(row) for row in rows]


def to_export_csv(df: pd.DataFrame) -> bytes:
    """Serialise the session frame back to a re-uploadable CSV.

    Only the Record fields go out — month, spend and friends are derived on
    load, so exporting them would just be stale duplication.
    """
    columns = [c for c in RECORD_FIELDS if c in df.columns]
    return df[columns].to_csv(index=False).encode()


def run_ingest(file, bank: str) -> list[Record] | None:
    """Run an uploaded CSV through the pipeline, streaming its log to the page.

    Returns the records, or None if the run failed. Everything that can
    realistically fail here is external — a missing API key, a CSV that isn't
    from `bank` — so failure is reported in place rather than raised, leaving
    whatever the session already had untouched.
    """
    with st.status("Running ingest pipeline…", expanded=True) as status:
        try:
            with contextlib.redirect_stdout(StreamlitWriter(status)):
                records = ingest.run_pipeline(pd.read_csv(file), bank)
        except Exception as error:
            status.update(label="Ingest failed", state="error")
            st.exception(error)
            return None

        status.update(label=f"Ingested {len(records)} records", state="complete")
    return records


def render_upload(replacing: bool) -> None:
    """Upload a CSV — either a raw bank export or a processed one from here.

    `replacing` only changes the warning: loading anything discards the
    session's current frame, manual category corrections included.
    """
    file = st.file_uploader("CSV file", type="csv")
    bank = st.selectbox("Bank", sorted(BANK_ADAPTORS))
    if replacing:
        st.caption("Loading a file replaces the current data, including your edits.")

    if file is None:
        st.caption(
            "A raw bank export runs the categorisation pipeline. A processed "
            "export downloaded from here loads straight back in, free."
        )
        return

    # Peek at the shape before committing to the expensive path. read_csv
    # consumes the buffer, so rewind for whichever branch reads it next.
    peeked = pd.read_csv(file)
    file.seek(0)

    if is_processed(peeked):
        st.success("Processed export detected — no pipeline run needed.")
        if st.button("Load data", type="primary"):
            store_frame(build_frame(to_records(peeked)), demo=False)
            st.rerun()
        return

    st.info(f"Raw bank export detected — {len(peeked)} rows to categorise.")
    if st.button("Run ingest", type="primary"):
        if records := run_ingest(file, bank):
            store_frame(build_frame(records), demo=False)
            st.rerun()


def render_demo_notice() -> None:
    """Say plainly that this is not the viewer's data.

    Sits above the dashboard rather than in the sidebar: someone who mistakes
    sample spending for their own has been actively misled, so it has to be
    somewhere they cannot miss it.
    """
    st.info(
        "Showing sample data so you can explore the dashboard. Upload your own "
        "bank CSV from the sidebar to analyse it — nothing you upload is stored.",
        icon=":material/science:",
    )


def render_download(df: pd.DataFrame) -> None:
    """Export the current frame, edits included.

    This is the only durability the app offers: records live in session state,
    so a browser refresh loses them and re-ingesting a raw export costs another
    LLM run. Downloading and re-uploading is the way back in for free.
    """
    st.download_button(
        "Download processed CSV",
        data=to_export_csv(df),
        file_name="budget_processed.csv",
        mime="text/csv",
        width="stretch",
    )


# ------------------------------------------------------------------- helpers


def month_axis(df: pd.DataFrame) -> alt.X:
    """X encoding of 'Jul 25'-style labels, ordered by the sortable 'YYYY-MM'."""
    order = df.sort_values("month")["month_label"].unique().tolist()
    return alt.X("month_label:O", sort=order, title=None, axis=alt.Axis(labelAngle=0))


def theme_palette() -> tuple[bool, str, list[str]]:
    """(is_dark, page surface, spend ramp) for the viewer's current theme."""
    dark = st.context.theme.type == "dark"
    surface = "#0e1117" if dark else "#ffffff"  # Streamlit's own page backgrounds
    return dark, surface, list(reversed(BLUE_RAMP)) if dark else BLUE_RAMP


def metric_grid(items: list[tuple[str, str]], per_row: int = 4) -> None:
    """Lay metrics out on a fixed grid so a short row doesn't stretch the page.

    st.columns(len(items)) makes two metrics span half the screen each; a fixed
    per_row keeps every card the same width no matter how many there are.
    """
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, items[start : start + per_row]):
            with col.container(border=True):
                st.metric(label, value)


# ------------------------------------------------------------------ sections


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Global filters. Returns the filtered frame every section then works from."""
    excluded = df[df["parent"] == EXCLUDED]

    with st.sidebar:
        st.header("Filters")
        if st.checkbox("Exclude transfers", value=True):
            df = cast(pd.DataFrame, df[df["parent"] != "Transfers"])

        # Excluded rows are hidden by default but must stay reachable — otherwise
        # excluding a record makes it invisible and impossible to put back.
        show_excluded = st.checkbox(
            f"Show excluded ({len(excluded)})",
            value=False,
            help="Excluded transactions are left out of every total.",
            disabled=excluded.empty,
        )
        if not show_excluded:
            df = cast(pd.DataFrame, df[df["parent"] != EXCLUDED])
        if not excluded.empty:
            st.caption(f"${excluded['spend'].sum():,.0f} excluded from totals")

    return df


def render_selection_bar(selection: Selection) -> None:
    """What's currently selected, and the one way out of it.

    Cross-filtering is only safe if it is obviously reversible — without this the
    page can sit filtered down to near-nothing with no visible cause.
    """
    if not selection:
        st.caption(
            "Click a heatmap row, a monthly total or a trend point to filter the page."
        )
        return

    message, button = st.columns([5, 1], vertical_alignment="center")
    message.info(f"Filtered by — {selection.describe()}", icon=":material/filter_alt:")
    button.button("Clear selection", on_click=clear_selection, width="stretch")


def render_summary(df: pd.DataFrame) -> None:
    """Headline totals plus one metric per year of data."""
    n_months = df["month"].nunique()
    total_spend = df["spend"].sum()
    avg_monthly = total_spend / n_months if n_months else 0.0
    yearly = df.groupby("year", as_index=False).agg(spend=("spend", "sum"))

    metric_grid(
        [
            ("Total spend", f"${total_spend:,.0f}"),
            ("Avg monthly spend", f"${avg_monthly:,.0f}"),
            ("Months of data", f"{n_months}"),
            *(
                (f"{row.year} spend", f"${row.spend:,.0f}")
                for row in yearly.itertuples()
            ),
        ]
    )


def category_editor(
    frame: pd.DataFrame,
    key: str,
    columns: dict,
    confirmable: bool = False,
    groups: dict[int, list[int]] | None = None,
) -> None:
    """Transaction table whose only editable column is Category.

    Everything else is a fact the CSV owns, so `disabled` is derived rather than
    listed — a column added to `frame` cannot accidentally become editable. `id`
    rides along hidden because it is what matches an edit back to its record.

    `confirmable` adds an Accept button, for the review queue: a suggestion that
    is already right needs a way to be agreed with, otherwise the only way to
    clear it would be to change the category to something else and back.

    Edits commit as they are made. A data_editor reruns the script on every cell
    change, so the edit is already in hand; applying it there is what lets Parent
    follow Category immediately, which it cannot do while a change sits pending
    in the widget. There is nothing to batch — the frame is in memory, so a write
    costs nothing, and a Save button would only be a second click before the
    parent could catch up.

    The key carries a change counter: a data_editor holds pending edits against
    row *positions*, and committing one can move a row out of the table under it
    (correcting a category drops it from a category-filtered drilldown, or off
    the review queue). Bumping the counter builds a fresh widget, which both
    discards the now-stale diff and stops it being re-applied to whichever rows
    shuffled into those positions.
    """
    editable = {"category"}
    accept_config: dict = {}
    if confirmable:
        # Transient, never part of the session frame — it is a control, not data.
        # assign() appends, which puts it last, immediately after category,
        # parent and confidence: you accept on the strength of those three, so
        # the button belongs beside them rather than before you have read them.
        frame = frame.assign(accept=ACCEPT_LABEL)
        # Left out of `disabled`: a ButtonColumn is read-only regardless, and
        # disabling it risks the frontend swallowing the click too.
        editable.add("accept")
        accept_config["accept"] = st.column_config.ButtonColumn(
            "",  # no header — a column of buttons needs no explaining
            width=ACCEPT_WIDTH,
            type="primary",  # green, via primaryColor in .streamlit/config.toml
            help="Confirm this category as correct",
            key=ACCEPT_CLICK_KEY,
            on_click=accept_clicked,
            # What each displayed row stands for, since the click reports only a
            # position: its own id, plus any alike rows grouped behind it
            args=(
                tuple(
                    tuple((groups or {}).get(i, [i])) for i in cast(list, frame["id"])
                ),
            ),
        )

    edited = st.data_editor(
        frame,
        column_config={
            "id": None,  # hidden, but kept so we can map edits back
            # Pixels, not "small" (75px): that truncated "Restaurants & Takeaway"
            # and "Uncategorised". Stretch tables papered over it by handing out
            # surplus width; a content-sized one shows exactly what you ask for.
            "category": st.column_config.SelectboxColumn(
                "Category", options=EDITABLE_CATEGORIES, required=True, width=190
            ),
            "parent": st.column_config.TextColumn("Parent", width=130),
            **accept_config,
            **columns,
        },
        disabled=[c for c in frame.columns if c not in editable],
        hide_index=True,
        # Every table stretches, so they all line up down the page. The cost is
        # that Streamlit hands any leftover width out evenly to every column,
        # explicit pixel widths included — a button column cannot be pinned
        # narrow here. Sizing the data columns generously is the lever: the less
        # surplus there is, the less of it lands behind the button.
        width="stretch",
        key=f"{key}:{st.session_state.get(SAVES_KEY, 0)}",
    )

    if apply_category_edits(cast(pd.DataFrame, edited), groups):
        st.session_state[SAVES_KEY] = st.session_state.get(SAVES_KEY, 0) + 1
        # Rerun so Parent, the totals and the review queue all reflect the edit.
        # This terminates: the new widget has no pending diff, so the next run
        # finds nothing to apply.
        st.rerun()


def render_top_tables(df: pd.DataFrame, top_n: int = 10) -> None:
    """Biggest single transactions beside the biggest categories.

    Built from the unfiltered frame and deliberately inert as a *filter*: these
    are a fixed reference for the year's largest spends, not a control, and
    everything that drives the page is a click on the heatmap or the trend.

    The records table is still editable, though — a mis-categorised transaction is
    most likely to be noticed here, where the biggest spends are, so it is worth
    being able to fix it without hunting the record down in the drilldown first.
    """
    records_col, categories_col = st.columns([3, 2])

    with records_col:
        st.subheader(f"Top {top_n} expense records")
        category_editor(
            cast(
                pd.DataFrame,
                df.nlargest(top_n, "spend")[
                    ["id", "date", "spend", "merchant", "description", "category", "parent"]
                ].reset_index(drop=True),  # editor holds edits against row positions
            ),
            key="top_records_editor",
            columns={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "spend": st.column_config.NumberColumn("Spend", format="dollar"),
                "merchant": st.column_config.TextColumn("Merchant", width="medium"),
                "description": st.column_config.TextColumn(
                    "Description", width="medium"
                ),
            },
        )

    with categories_col:
        st.subheader(f"Top {top_n} expense categories")
        st.dataframe(
            df.groupby(["category", "parent"], as_index=False)
            .agg(spend=("spend", "sum"))
            .nlargest(top_n, "spend"),
            column_config={
                "category": st.column_config.TextColumn("Category"),
                "parent": st.column_config.TextColumn("Parent"),
                "spend": st.column_config.NumberColumn("Spend", format="dollar"),
            },
            hide_index=True,
        )


def parent_heatmap(
    parent_monthly: pd.DataFrame,
    monthly_totals: pd.DataFrame,
) -> alt.VConcatChart:
    """Parent x month spend heatmap under a monthly-totals bar, sharing an x-scale.

    Both frames carry a boolean `selected` column. Selection is drawn in red text,
    never by touching the cell's fill or opacity: on a heatmap colour *is* the data
    channel, and dimming a cell makes it read as less spend. The label was the one
    channel still free to carry it.

    `selected` is precomputed rather than expressed as a Vega `condition` because a
    chart's own selection state lives client-side: Python can read it but cannot
    set it, so a click made in *another* section could never reach a condition.

    Pure chart builder: no Streamlit calls beyond reading the active theme, so it
    can be reused anywhere an Altair chart is accepted.
    """
    dark, surface, ramp = theme_palette()

    # Heaviest parents on top; ~19 of them is far past the ~8 a colour legend can
    # carry, so identity lives on the y-axis and colour encodes magnitude instead.
    parent_order = (
        parent_monthly.groupby("parent")["spend"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Cell values are heavily skewed (one month is 3x the median), so a linear ramp
    # would flatten everything but the outlier; sqrt keeps the mid-range readable.
    colour = alt.Color(
        "spend:Q",
        title="Spend ($AUD)",
        scale=alt.Scale(range=ramp, type="sqrt"),
        legend=alt.Legend(format="$,.0f", gradientLength=180),
    )

    # The value in the cell does the work a colour legend can't: exact magnitude.
    # The flip point tracks the *colour* ramp, not the value: under a sqrt scale a
    # cell reaches ~60% of the ramp at 36% of the max, so that's where ink inverts.
    # Both reds are picked against the cell they sit on, exactly as the greys are.
    bright_cut = parent_monthly["spend"].max() * 0.36
    strong = alt.datum.spend > bright_cut
    picked = alt.datum.selected
    # Literal values, never a second colour field: the labels share a layer with the
    # cells, and Vega-Lite unifies a channel's scale across a layer — a `label_colour`
    # field here collides with the spend ramp and takes the whole chart down with it.
    label_ink = (
        alt.when(picked & strong)
        .then(alt.value("#a4161a" if dark else "#ffb3b3"))
        .when(picked)
        .then(alt.value("#ff6b6b" if dark else "#c1121f"))
        .when(strong)
        .then(alt.value(surface if dark else "#ffffff"))
        .otherwise(alt.value("#c3c2b7" if dark else "#52514e"))
    )

    parent_pick = alt.selection_point(fields=["parent"], name="parent_pick")
    heat_base = alt.Chart(parent_monthly).encode(
        x=month_axis(parent_monthly),
        y=alt.Y("parent:N", sort=parent_order, title=None),
    )
    # 2px surface-coloured stroke keeps cells from fusing into blocks
    cells = heat_base.mark_rect(stroke=surface, strokeWidth=2).encode(
        color=colour,
        tooltip=[
            alt.Tooltip("parent:N", title="Parent"),
            alt.Tooltip("month_label:N", title="Month"),
            alt.Tooltip("spend:Q", title="Spend", format="$,.0f"),
        ],
    )
    labels = heat_base.mark_text(fontSize=11).encode(
        text=alt.Text("spend:Q", format="$,.0f"),
        color=label_ink,
    )
    heatmap = (cells + labels).add_params(parent_pick).properties(height=520)

    # The bar's colour is decoration, not an encoding, so unlike a cell it is free
    # to turn red for a selected month.
    month_pick = alt.selection_point(fields=["month"], name="month_pick")
    selected_bar = "#ff6b6b" if dark else "#c1121f"
    totals = (
        alt.Chart(monthly_totals)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=month_axis(monthly_totals),
            y=alt.Y(
                "spend:Q", title="Total", axis=alt.Axis(format="$,.0f", tickCount=3)
            ),
            color=alt.when(picked)
            .then(alt.value(selected_bar))
            .otherwise(alt.value(ramp[-1])),
            tooltip=[
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip("spend:Q", title="Total spend", format="$,.0f"),
            ],
        )
        .add_params(month_pick)
        .properties(height=90)
    )

    return (
        alt.vconcat(totals, heatmap, spacing=8, bounds="flush")
        .resolve_scale(x="shared")
        # Set explicitly so Streamlit leaves it alone: it only injects an autosize
        # when the spec has none, and its heuristic downgrades any vconcat holding a
        # layer to "pad". Under pad, Vega sizes the *plot* to the container and then
        # adds the axis, labels and legend outside it, so the chart overhangs the
        # page. fit-x sizes the whole thing, decorations included, to the width it
        # is given. (See _has_nested_composition in streamlit/elements/vega_charts.)
        .properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    )


def render_parent_heatmap(df: pd.DataFrame, selection: Selection) -> None:
    """Draw the heatmap and push the parents and months the user clicked.

    This is the page's overview, so unlike every other section it is never
    filtered: the grid always shows all parents against all months at their true
    totals. The selection only changes which cells are lit. Source of `parents`
    (a heatmap row) and `months` (a totals bar), both of which would be
    unclickable — and so impossible to undo — if selecting them removed them.
    """
    st.subheader("Spend by parent category per month")
    # Built from the unfiltered frame, always: the grid is the overview, so every
    # parent and month stays in place and every cell keeps its true total. Holding
    # the data still also holds the colour scale still, which is what makes two
    # cells comparable at a glance. Selection changes the lighting, nothing else.
    parent_monthly = df.groupby(["parent", "month", "month_label"], as_index=False).agg(
        spend=("spend", "sum")
    )
    monthly_totals = df.groupby(["month", "month_label"], as_index=False).agg(
        spend=("spend", "sum")
    )

    # Marking is a row test and a column test, never a per-cell one. Parents and
    # categories pick the rows, months pick the columns, and a cell is red where
    # both agree — so a parent lights its whole row and a month its whole column.
    #
    # Testing cells individually (does *this* parent-month survive the filter?) is
    # what left gaps mid-row: a category only spends in some months, so the months
    # it skipped went unmarked and the row read as three separate selections.
    #
    # An empty selection marks nothing rather than everything: red means "you
    # picked this".
    rows = parent_monthly["parent"]
    columns = parent_monthly["month"]
    if selection.parents or selection.categories:
        # filter() keeps parents and categories as one "and", so a category outside
        # the chosen parent correctly lights nothing.
        rows = rows.isin(selection.filter(df, ignore={"months"})["parent"].unique())
    else:
        rows = pd.Series(bool(selection), index=parent_monthly.index)
    if selection.months:
        columns = columns.isin(selection.months)
    else:
        columns = pd.Series(True, index=parent_monthly.index)
    parent_monthly["selected"] = rows & columns

    # The bar row is the month picker, so it marks the months themselves — not every
    # month the selection happens to touch, which for any parent is all of them.
    monthly_totals["selected"] = monthly_totals["month"].isin(selection.months)

    event = st.altair_chart(
        parent_heatmap(parent_monthly, monthly_totals),
        key=widget_key("heatmap"),
        on_select="rerun",
        width="stretch",
    )
    st.caption(
        "Click a row to filter by that parent, or a bar to filter by that month "
        "(shift-click for more)."
    )

    def picked(param: str, field: str) -> tuple[str, ...]:
        return tuple(sorted({p[field] for p in event["selection"].get(param, [])}))

    parents = picked("parent_pick", "parent")
    months = picked("month_pick", "month")
    push("heatmap", (parents, months), parents=parents, months=months)


def category_trend(data: pd.DataFrame) -> alt.LayerChart:
    """Monthly line/area per category, with clickable per-month markers."""
    # Point selection bound to the two fields we drill down on
    pick = alt.selection_point(fields=["category", "month"], name="pick")

    base = alt.Chart(data).encode(
        x=month_axis(data),
        y=alt.Y("spend:Q", title="Amount ($AUD)", stack=None),
        color=alt.Color(
            "category:N", title="Category", scale=alt.Scale(range=CATEGORICAL)
        ),
    )
    area = base.mark_area(opacity=0.25)
    line = base.mark_line(strokeWidth=2)
    # Markers on the line at each month are the click targets and carry the
    # selection; the clicked one is emphasised, the rest sit quietly on the line
    markers = (
        base.mark_point(filled=True)
        .encode(
            size=alt.condition(pick, alt.value(140), alt.value(45)),
            tooltip=["category", "month", "spend"],
        )
        .add_params(pick)
    )
    # alt.layer() is typed as also possibly returning a FacetChart; it can't here
    return cast(alt.LayerChart, alt.layer(area, line, markers))


def render_drilldown(df: pd.DataFrame, selection: Selection) -> None:
    """Editable list of every transaction in the current selection.

    Filtered by all three dimensions, so whichever section the click came from
    this is always "the rows behind the numbers above".
    """
    detail = selection.filter(df).sort_values(by="date")
    st.subheader(f"Component transactions ({len(detail)})")
    if detail.empty:
        st.info("No transactions match the current selection.")
        return

    category_editor(
        cast(
            pd.DataFrame,
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
            ].reset_index(drop=True),
        ),
        # Scoped to the selection as well as the save count: a different selection
        # is a different set of rows in the same positions.
        key=widget_key(
            "detail_editor", selection.parents, selection.categories, selection.months
        ),
        columns={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "merchant": st.column_config.TextColumn("Merchant", width="medium"),
            "description": st.column_config.TextColumn("Description", width="medium"),
            "confidence": st.column_config.NumberColumn(
                "Confidence", format="%.2f", width="small"
            ),
        },
    )


def render_review(df: pd.DataFrame) -> None:
    """The classifier's own doubts, as a to-do list.

    Deliberately outside the cross-filtering: it is a queue to work through, not
    a view of the current selection, and it takes the unfiltered frame so a stray
    heatmap click cannot hide rows still needing attention. A correction sets
    confidence to 1.0, so each row leaves the queue as it is dealt with and the
    section disappears once it is empty.

    Alike rows are collapsed onto one representative, so three near-identical
    dinners are a single decision rather than the same judgement three times. The
    count is of decisions; the dollar total is of every record behind them.
    """
    flagged = needs_review(df)
    if flagged.empty:
        return

    groups = group_similar(flagged)
    reps = cast(pd.DataFrame, flagged[flagged["id"].isin(list(groups))]).assign(
        covers=lambda f: [len(groups[i]) for i in f["id"]]
    )
    total = flagged["spend"].sum()
    # The key is what keeps this open while you work through it. Without one,
    # Streamlit derives the expander's identity from its parameters — and this
    # label carries a count that drops with every correction, so each edit would
    # look like a brand new expander and snap back to collapsed. `expanded` is
    # only the initial state; the key persists whatever the user last chose.
    grouped = len(reps) != len(flagged)
    with st.expander(
        f"⚠︎  {len(reps)} to review — ${total:,.0f}"
        + (f"  ({len(flagged)} transactions)" if grouped else ""),
        expanded=False,
        key="review_expander",
    ):
        st.caption(
            f"Either the classifier returned nothing, or it was under "
            f"{CONFIDENCE_THRESHOLD:.0%} confident. Largest first, and already counted "
            "in every total above. Hit **Accept** to confirm a category as it "
            "stands, or pick a different one — either way the row leaves this list."
            + (
                "  Alike transactions are grouped: **Covers** is how many your "
                "decision settles at once."
                if grouped
                else ""
            )
        )
        category_editor(
            cast(
                pd.DataFrame,
                reps[
                    [
                        "id",
                        "date",
                        "amount",
                        "merchant",
                        "description",
                        "category",
                        "parent",
                        "confidence",
                        "covers",
                    ]
                ].reset_index(drop=True),
            ),
            key="review_editor",
            # Deliberately roomy. Width left over after these is split evenly
            # across all eight columns, and an eighth of it lands behind the
            # Accept button — so the surplus is what has to be kept small.
            # Description takes the lion's share: it is what identifies a
            # transaction when the merchant is missing, which here it often is.
            columns={
                "date": st.column_config.DateColumn(
                    "Date", format="DD MMM YYYY", width=110
                ),
                "amount": st.column_config.NumberColumn(
                    "Amount", format="dollar", width=110
                ),
                "merchant": st.column_config.TextColumn("Merchant", width=210),
                "description": st.column_config.TextColumn("Description", width=560),
                "confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.2f", width=120
                ),
                "covers": st.column_config.NumberColumn(
                    "Covers",
                    help="Transactions this decision applies to, including this one",
                    format="%d",
                    width=90,
                ),
            },
            confirmable=True,
            groups=groups,
        )


def explore_seed(
    options: list[str], selection: Selection, current: list[str]
) -> list[str]:
    """What the category picker should show for the current selection.

    Categories clicked elsewhere win; failing that a clicked parent has already
    narrowed `options` to its children, so show them all — capped at the number of
    hues the palette can tell apart. With nothing selected, fall back to defaults.
    """
    if selection.categories:
        wanted = [c for c in selection.categories if c in options]
        # A click on the trend below selects a line that is already drawn. Narrowing
        # to it would pull away the very lines it was being compared against, so
        # only a selection reaching outside the chart refocuses the picker.
        return current if wanted and set(wanted) <= set(current) else wanted
    if selection.parents:
        return options[: len(CATEGORICAL)]
    return [c for c in DEFAULT_CATEGORIES if c in options]


def render_explore(df: pd.DataFrame, selection: Selection) -> None:
    """Category picker and monthly trend, whose points are themselves clickable.

    Source of `categories` and `months` (a point carries both), so it is filtered
    by `parents` only. The picker is a view control rather than a third source:
    if it wrote to the selection the page would open pre-filtered to its defaults.
    """
    st.subheader("Explore categories")
    scoped = selection.filter(df, ignore={"categories", "months"})
    if scoped.empty:
        st.info("No transactions match the current selection.")
        return

    monthly = scoped.groupby(["category", "month", "month_label"], as_index=False).agg(
        spend=("spend", "sum")
    )
    options = sorted(monthly["category"].unique())

    # A keyed widget owns its value once created, so a click made elsewhere has to
    # land in session state *before* the widget exists. Writing only when the seed
    # itself changes is what stops that clobbering the user's manual edits on every
    # rerun.
    key = widget_key("explore")
    seed = explore_seed(options, selection, st.session_state.get(key, []))
    if seed != st.session_state.get(SEED_KEY):
        st.session_state[SEED_KEY] = seed
        st.session_state[key] = seed

    categories = st.multiselect("Choose categories", options, key=key)
    if not categories:
        st.error("Please select at least one category.")
        return

    data = cast(pd.DataFrame, monthly[monthly["category"].isin(categories)])
    # Deliberately not keyed on `categories`: the picker follows a click on this
    # chart, so a key that moved with it would rebuild the chart, drop the very
    # selection that caused the move, and cancel the click.
    event = st.altair_chart(
        category_trend(data),
        key=widget_key("trend"),
        on_select="rerun",
        width="stretch",
    )
    st.caption("Click a point to filter the whole page by that category and month.")

    points = event["selection"].get("pick", [])
    picked_categories = tuple(sorted({p["category"] for p in points}))
    picked_months = tuple(sorted({p["month"] for p in points}))
    push(
        "trend",
        (picked_categories, picked_months),
        categories=picked_categories,
        months=picked_months,
    )


# ------------------------------------------------------------- entry points


def render(df: pd.DataFrame | None = None) -> None:
    """Render the whole page into the current Streamlit container.

    Pass a frame to analyse something other than the session's own — it must
    carry the columns build_frame() adds (month, month_label, year, spend).
    Upload and download are offered only for the session-backed frame, since a
    caller passing its own data owns how that data comes and goes.
    """
    session_backed = df is None
    if df is None:  # not `session_backed`, which a type checker cannot narrow on
        df = load_frame()
        if df is None:  # only reachable if the bundled demo file is missing
            st.info("Upload a CSV to get started.")
            render_upload(replacing=False)
            return
        if st.session_state.get(IS_DEMO_KEY):
            render_demo_notice()

    # Export before filtering: the download is the session's whole dataset, not
    # whatever slice happens to be on screen
    full = df
    df = sidebar_filters(df)
    if session_backed:
        with st.sidebar:
            render_download(full)
            with st.expander("Load a different CSV"):
                render_upload(replacing=True)

    # Read the shared selection once, up front, and hand the same value to every
    # section. Sections don't feed each other in render order any more — they all
    # read this and write back through push(), which reruns the page.
    selection = get_selection()

    render_selection_bar(selection)
    render_summary(selection.filter(df))
    # Unfiltered on purpose — a review queue the current selection could hide is
    # worse than no queue at all
    render_review(full)
    render_top_tables(df)
    render_parent_heatmap(df, selection)
    render_explore(df, selection)
    if selection:
        render_drilldown(df, selection)


def main() -> None:
    """Page entry point, run by Streamlit itself: `streamlit run analyse.py`."""
    st.set_page_config(page_title="Budget analysis", layout="wide")
    st.title("Budget analysis")
    render()


def launch(port: int = 8501, open_browser: bool = True) -> int:
    """Start the dashboard server and open it in a browser. Blocks until stopped.

    Shells out because `streamlit run` has to own the script's execution — a
    Streamlit page cannot be started by importing it. Uses the current
    interpreter (not a bare `streamlit` on PATH) so the active venv is honoured,
    and pins cwd to this file's directory so relative paths resolve wherever the
    caller happens to be.
    """
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                Path(__file__).name,
                "--server.port",
                str(port),
                "--server.headless",
                "false" if open_browser else "true",
            ],
            cwd=Path(__file__).parent,
        ).returncode
    except KeyboardInterrupt:  # Ctrl-C is how you stop a server, not an error
        return 0


if __name__ == "__main__":
    main()
