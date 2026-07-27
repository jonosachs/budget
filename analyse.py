"""Streamlit dashboard for monthly spend analysis.

Three entry points:

    streamlit run analyse.py          # runs main()
    from analyse import launch        # start the server from ordinary Python
    from analyse import render        # embed the page in another Streamlit app

`render()` and the `render_*` sections need an active Streamlit runtime — they
draw into the current container and start nothing. `launch()` is what actually
opens a browser.

Every section is cross-filtered through one shared `Selection` (see the class
docstring): a click anywhere narrows everything else, and each section ignores
only the dimension it is itself the source of.

Importing this module has no side effects. `st.set_page_config` lives in `main()`
alone, since Streamlit allows it only once per page and only before any other
`st` call — an embedding app owns that decision, not this module.
"""

from in_out import read_records, write_records
from models import Record
from config import get_taxonomy_children
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, ClassVar, Collection
import pandas as pd
import streamlit as st
import altair as alt
import subprocess
import sys

RECORDS_PATH = "assets/records.json"

# Manual-only category: the user parks a transaction here to keep it out of every
# total. Deliberately NOT in TAXONOMY — that dict is the LLM's option list, and
# nothing should be auto-excluded. It is its own parent so parent-level views
# (the heatmap, the filter) treat it like any other top-level bucket.
EXCLUDED = "Excluded"
CATEGORIES = sorted(get_taxonomy_children().keys())
EDITABLE_CATEGORIES = CATEGORIES + [EXCLUDED]

# Sequential single-hue ramp, light -> dark. Reversed on a dark page so the
# lightest end always means "most spend" and near-zero recedes into the surface.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
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
DEFAULT_CATEGORIES = ("Maintenance", "Accommodation")


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


@st.cache_data
def get_records(path: str = RECORDS_PATH) -> pd.DataFrame:
    """Row-level transactions with a 'YYYY-MM' month and positive 'spend'.

    spend is -amount, so debits are positive and refunds net against them.
    """
    records = read_records(Record, path)
    df = pd.DataFrame([r.model_dump() for r in records])
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_label"] = df["date"].dt.strftime("%b %y")
    df["year"] = df["date"].dt.year
    df["spend"] = -df["amount"]
    return df.dropna(subset=["category"])


def save_category_edits(edited: pd.DataFrame, path: str = RECORDS_PATH) -> int:
    """Persist category corrections back to the source file, keyed by id.

    Returns the number of records changed. Manual edits set confidence to 1.0
    to mark them as human-vetted rather than an LLM guess.
    """
    records = read_records(Record, path)
    new_category = dict(zip(edited["id"], edited["category"]))
    children = get_taxonomy_children()  # child -> parent

    changed = 0
    for r in records:
        category = new_category.get(r.id)
        if category is not None and category != r.category:
            r.category = category
            # EXCLUDED has no taxonomy parent, so it is its own — without this
            # children.get() returns None and the record loses its parent entirely
            r.parent = EXCLUDED if category == EXCLUDED else children.get(category)
            r.confidence = 1.0
            changed += 1

    if changed:
        write_records(records, path)
    return changed


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
        st.caption("Click any table row, heatmap cell or chart point to filter the page.")
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


def render_top_tables(df: pd.DataFrame, selection: Selection, top_n: int = 10) -> None:
    """Biggest single transactions beside the biggest categories.

    Both tables are built from the unfiltered frame, like the heatmap: a top 10
    that re-ranks under its own filter is a moving target. A selection made
    elsewhere shows up as selected rows instead, wherever it reaches into the
    top 10. Source of the `categories` dimension — a records row contributes its
    own category.
    """
    # Both frames reset the index because a dataframe selection is row *positions*.
    top_records = cast(
        pd.DataFrame,
        df.nlargest(top_n, "spend")[
            ["date", "spend", "merchant", "description", "category", "parent"]
        ].reset_index(drop=True),
    )
    top_categories = cast(
        pd.DataFrame,
        df.groupby(["category", "parent"], as_index=False)
        .agg(spend=("spend", "sum"))
        .nlargest(top_n, "spend")[["category", "parent", "spend"]]
        .reset_index(drop=True),
    )

    def preselected(frame: pd.DataFrame) -> pd.Series:
        """Mask of the rows a selection made elsewhere on the page already implies.

        Category only, deliberately. Seeding on `parent` too would put rows on
        screen that the user cannot switch off: deselecting one says nothing about
        the parent that put it there, so it would just come straight back.
        """
        return frame["category"].isin(selection.categories)

    def positions(mask: pd.Series) -> list[int]:
        """The mask as the row positions Streamlit wants. Index is reset, so
        position and label are the same number."""
        return mask.index[mask].tolist()

    records_seed = preselected(top_records)
    categories_seed = preselected(top_categories)

    # selection_default only applies when a widget is first created, so the key has
    # to move with it — otherwise a selection made elsewhere never reaches the table.
    key = widget_key("top", selection.categories)

    records_col, categories_col = st.columns([3, 2])

    with records_col:
        st.subheader(f"Top {top_n} expense records")
        records_event = st.dataframe(
            top_records,
            key=f"{key}:records",
            selection_default={"selection": {"rows": positions(records_seed)}},
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "spend": st.column_config.NumberColumn("Spend", format="dollar"),
                "merchant": st.column_config.TextColumn("Merchant", width="medium"),
                "description": st.column_config.TextColumn(
                    "Description", width="medium"
                ),
                "category": st.column_config.TextColumn("Category", width="small"),
                "parent": st.column_config.TextColumn("Parent", width="small"),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )
        st.caption("Select rows to filter the page by their categories.")

    with categories_col:
        st.subheader(f"Top {top_n} expense categories")
        categories_event = st.dataframe(
            top_categories,
            key=f"{key}:categories",
            selection_default={"selection": {"rows": positions(categories_seed)}},
            column_config={
                "category": st.column_config.TextColumn("Category"),
                "parent": st.column_config.TextColumn("Parent"),
                "spend": st.column_config.NumberColumn("Spend", format="dollar"),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )
        st.caption("Select rows to filter the page by those categories.")

    def selected(event, frame: pd.DataFrame) -> set[str]:
        # iloc, not loc: Streamlit hands back row *positions*
        rows = event.get("selection", {}).get("rows", [])
        return set(frame["category"].iloc[rows])

    # Each table reports what the user *changed* against what we seeded it with,
    # never the categories it happens to be showing. Unioning the two tables looks
    # equivalent and is not: several records share a category, so a category
    # switched off in one table stays in the other's union and snaps straight back.
    # A delta says "remove this" in a way the other table cannot contradict.
    added: set[str] = set()
    removed: set[str] = set()
    for event, frame, seed in (
        (records_event, top_records, records_seed),
        (categories_event, top_categories, categories_seed),
    ):
        picked = selected(event, frame)
        seeded = set(frame.loc[seed, "category"])
        added |= picked - seeded
        removed |= seeded - picked

    categories = tuple(sorted((set(selection.categories) | added) - removed))
    # Written straight to the selection rather than through push(): the delta is
    # already exact, so comparing against the current value is all the guard needed.
    if categories != selection.categories:
        st.session_state[SELECTION_KEY] = replace(selection, categories=categories)
        st.rerun()


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
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "merchant": st.column_config.TextColumn("Merchant", width="medium"),
            "description": st.column_config.TextColumn("Description", width="medium"),
            "category": st.column_config.SelectboxColumn(
                "Category", options=EDITABLE_CATEGORIES, required=True, width="small"
            ),
            "parent": st.column_config.TextColumn("Parent", width="small"),
            "confidence": st.column_config.NumberColumn(
                "Confidence", format="%.2f", width="small"
            ),
        },
        disabled=["date", "amount", "merchant", "description", "confidence"],
        hide_index=True,
        # Edits are held against row positions too, so a new selection needs a new
        # widget — otherwise a pending edit lands on whatever row now sits there.
        key=widget_key(
            "detail_editor", selection.parents, selection.categories, selection.months
        ),
    )

    if st.button("Save corrections"):
        changed = save_category_edits(cast(pd.DataFrame, edited))
        if changed:
            get_records.clear()
            st.success(f"Updated {changed} record(s).")
            st.rerun()
        else:
            st.info("No changes to save.")


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

    Pass a frame to analyse something other than the records file — it must carry
    the columns get_records() builds (month, month_label, year, spend).
    """
    df = get_records() if df is None else df
    df = sidebar_filters(df)

    # Read the shared selection once, up front, and hand the same value to every
    # section. Sections don't feed each other in render order any more — they all
    # read this and write back through push(), which reruns the page.
    selection = get_selection()

    render_selection_bar(selection)
    render_summary(selection.filter(df))
    render_top_tables(df, selection)
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
    and pins cwd to this file's directory so RECORDS_PATH resolves wherever the
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
