"""
Institutional-Grade Real Estate Underwriting Portal
Price-per-SQFT statistical underwriting engine with Claude streaming validation.
"""

import io
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

try:
    import anthropic
except Exception:  # pragma: no cover - dependency guard
    anthropic = None


# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Underwriting Portal",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------------
# GLOBAL STRUCTURAL CSS  (Alo Yoga minimalism / institutional profile)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #0A192F;
    color: #FFFFFF;
}

/* Prevent top-padding clipping */
.main .block-container {
    padding-top: 4.5rem !important;
    max-width: 1180px;
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #FFFFFF;
}

/* Uniform uppercase tracked field labels */
.stTextInput label, .stNumberInput label, .stTextArea label,
.stFileUploader label, .stSelectbox label {
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #8892B0 !important;
}

/* Dark Slate input components — force text/value visibility, never blend
   into the background (–webkit-text-fill-color defeats autofill washout). */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    color: #FFFFFF !important;
    background-color: #172A45 !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #233554 !important;
    border-radius: 4px !important;
}

/* Clean light-silver placeholder text across every input node */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #A0AEC0 !important;
    -webkit-text-fill-color: #A0AEC0 !important;
    opacity: 1 !important;
}

.stFileUploader > div, .stFileUploader section {
    background-color: #172A45 !important;
    border: 1px dashed #233554 !important;
    border-radius: 4px !important;
}

/* Muted silver subtext */
.subtext {
    color: #8892B0;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    line-height: 1.6;
}

/* Restricted access callout frame */
.portal-callout {
    width: 100%;
    background-color: #172A45;
    border: 1px solid #233554;
    border-left: 3px solid #FFFFFF;
    border-radius: 6px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 2.2rem;
}
.portal-callout .tag {
    color: #8892B0;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.68rem;
    font-weight: 600;
}
.portal-callout .headline {
    color: #FFFFFF;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-top: 0.35rem;
}
.portal-callout .mapping {
    color: #8892B0;
    font-size: 0.9rem;
    margin-top: 0.35rem;
}

/* Section headers */
.section-label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    font-weight: 600;
    color: #8892B0;
    border-bottom: 1px solid #233554;
    padding-bottom: 0.5rem;
    margin: 1.6rem 0 1.2rem 0;
}

/* Statistical metric tiles */
.metric-tile {
    background-color: #172A45;
    border: 1px solid #233554;
    border-radius: 6px;
    padding: 1.3rem 1.4rem;
    height: 100%;
}
.metric-tile .m-label {
    color: #8892B0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.68rem;
    font-weight: 600;
}
.metric-tile .m-ppsf {
    color: #8892B0;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}
.metric-tile .m-value {
    color: #FFFFFF;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.25rem;
    letter-spacing: 0.01em;
}

/* Full-bleed pure white primary execution button */
div.stButton > button {
    width: 100%;
    background-color: #FFFFFF !important;
    color: #0A192F !important;
    -webkit-text-fill-color: #0A192F !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.95rem 1rem !important;
    font-size: 0.85rem !important;
}
/* Lock deep-navy label across hover/active/focus so it never washes out */
div.stButton > button:hover,
div.stButton > button:active,
div.stButton > button:focus,
div.stButton > button:focus:not(:active),
div.stButton > button:hover p,
div.stButton > button:active p,
div.stButton > button:focus p {
    background-color: #E6E9F0 !important;
    color: #0A192F !important;
    -webkit-text-fill-color: #0A192F !important;
}

/* Crystal-clear data tables — bold tracked silver headers, crisp white cells */
.stTable th,
[data-testid="stTable"] th,
[data-testid="stDataFrame"] th {
    color: #8892B0 !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase;
}
.stTable td,
[data-testid="stTable"] td,
[data-testid="stDataFrame"] td {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Streaming report block */
.report-frame {
    background-color: #172A45;
    border: 1px solid #233554;
    border-radius: 6px;
    padding: 1.6rem 1.8rem;
    color: #FFFFFF;
    line-height: 1.7;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PARSING / EXTRACTION HELPERS
# ----------------------------------------------------------------------------
# Numeric token: integer or decimal, optionally with thousands separators.
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")

# Date fragments to discard before classifying numbers. Without this, the
# components of a date (e.g. the "2024" in 12/05/2024) leak into the number
# stream and get mistaken for a 4-digit built-up area.
_DATE_RE = re.compile(
    r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b"           # 12/05/2024, 2024-05-12, 12.05.24
    r"|\b\d{1,2}[/.-][A-Za-z]{3,9}[/.-]\d{2,4}\b"    # 5-Jun-2024
    r"|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b"        # 5 Jun 2024
)

# Price anchors are still identified by magnitude -- a transaction price is
# always a "massive" absolute figure regardless of unit size.
_PRICE_MIN, _PRICE_MAX = 500_000, 100_000_000

# Structural-area labels. The numeric value tied to one of these IS the BUA,
# whatever its magnitude -- so micro-units and massive estates both parse with
# no numeric cap. (Replaces the old _BUA_MIN/_BUA_MAX bands entirely.)
#
# Binding is directional. A *leading* label names the value that follows it
# ("BUA: 3,200", or "Built-Up Area\n3,200" in a vertical paste). A *unit*
# label trails the value it qualifies ("3,200 sqft"). Without this direction a
# label sandwiched between two numbers (price above, area below) would wrongly
# bind to whichever is fewer characters away.
_BUA_LEAD_RE = re.compile(
    r"\bBuilt[\s\-]*Up(?:\s*Area)?\b"   # Built-Up, Built Up, Built-Up Area
    r"|\bFloor\s*Area\b"
    r"|\bBUA\b",
    re.IGNORECASE,
)
_BUA_UNIT_RE = re.compile(r"\bSq\.?\s*Ft\b|\bSquare\s*Feet\b", re.IGNORECASE)

# Land/plot-size labels (leading). A number tied to one of these is a plot
# dimension, NOT structural floorspace -- it is flagged and discarded so it
# can never pollute the per-SQFT denominator.
_PLOT_LABEL_RE = re.compile(
    r"\bPlot\s*Size\b|\bPlot\s*Area\b|\bPlot\b|\bLand\s*Area\b|\bLand\b",
    re.IGNORECASE,
)

# Max character gap between a number and a label for them to count as
# "directly tied" -- small enough to mean adjacency (a colon, a unit word, or
# a single newline from a vertical paste), not a number rows away.
_LABEL_PROXIMITY = 20

# Baseline villa per-SQFT distribution (AED/sqft). Used ONLY as an explicit
# assumption when no comparable transactions are parsed -- live comps always
# override these when present.
_BASELINE_PPSF = {"p15": 1_815.0, "median": 2_210.0, "p85": 2_430.0}


def _to_float(token):
    """Convert a numeric token (with thousands separators) to float."""
    try:
        return float(token.replace(",", ""))
    except (ValueError, AttributeError):
        return None



def _claim_labels(numbers, label_spans, kind, direction, assign):
    """
    Bind each label to the single number it qualifies and record the intended
    `kind` for that number in `assign` (number index -> (gap, kind)).

    direction "after"  -> the value follows the label  (leading labels: BUA:)
    direction "before" -> the value precedes the label (unit labels: sqft)

    Closer bindings win when two labels target the same number.
    """
    for ls in label_spans:
        best = None  # (gap, number_index)
        for i, num in enumerate(numbers):
            if direction == "after":
                gap = num["pos"] - ls[1]   # number starts after the label ends
            else:
                gap = ls[0] - num["end"]   # number ends before the label starts
            if gap < 0 or gap > _LABEL_PROXIMITY:
                continue  # wrong side, or too far to be "directly tied"
            if best is None or gap < best[0]:
                best = (gap, i)
        if best is not None:
            gap, i = best
            if i not in assign or gap < assign[i][0]:
                assign[i] = (gap, kind)


def extract_pairs_from_text(raw_text):
    """
    Parse a raw transaction feed (or extracted PDF text) into (price, BUA)
    pairs using label-first, keyword-proximity scanning -- not numeric caps.

    The flow:
      1. Strip structured date fragments so date digits never enter the stream.
      2. Locate every number, every structural-area label (BUA / Built-Up /
         Floor Area / Sqft) and every land label (Plot / Land) with positions.
      3. Bind each label *directionally* to the one number it qualifies, then
         classify that number: a structural label -> BUA; a land label -> plot
         size, which is discarded outright (it must never reach the per-SQFT
         denominator).
      4. Treat each price-band number (not itself a labelled BUA/plot figure)
         as a property anchor and pair it with the nearest BUA-labelled number.

    Because the BUA is identified by its label rather than its magnitude, both
    micro-units and massive estates parse correctly -- there is no size cap.

    Fallback: if the document carries NO structural or land labels at all, it
    is treated as the legacy bare two-column feed and each price anchor is
    paired with the nearest following plain number. (When any label exists,
    parsing is strictly label-driven.)

    Returns a (pairs, discarded_plot_count) tuple, where discarded_plot_count
    is how many figures were dropped for being tied to a Plot/Land label.
    """
    pairs = []
    if not raw_text:
        return pairs, 0

    cleaned = _DATE_RE.sub(" ", raw_text)

    numbers = []  # ordered: {"pos", "end", "value", "kind"}
    for m in _NUM_RE.finditer(cleaned):
        value = _to_float(m.group())
        if not value:
            continue
        numbers.append({"pos": m.start(), "end": m.end(), "value": value, "kind": "plain"})

    # Bind labels to numbers directionally; closest binding wins per number.
    assign = {}
    _claim_labels(numbers, [m.span() for m in _BUA_LEAD_RE.finditer(cleaned)], "bua", "after", assign)
    _claim_labels(numbers, [m.span() for m in _BUA_UNIT_RE.finditer(cleaned)], "bua", "before", assign)
    _claim_labels(numbers, [m.span() for m in _PLOT_LABEL_RE.finditer(cleaned)], "plot", "after", assign)
    for i, (_, kind) in assign.items():
        numbers[i]["kind"] = kind

    labels_present = bool(assign)
    used = set()

    for idx, num in enumerate(numbers):
        if num["kind"] != "plain":
            continue
        if not (_PRICE_MIN <= num["value"] <= _PRICE_MAX):
            continue  # only massive figures qualify as a price anchor

        match = None
        if labels_present:
            # Label-driven: nearest unused BUA-labelled number to this anchor.
            best_dist = None
            for j, other in enumerate(numbers):
                if j in used or other["kind"] != "bua":
                    continue
                dist = abs(other["pos"] - num["pos"])
                if best_dist is None or dist < best_dist:
                    best_dist, match = dist, j
        else:
            # Legacy bare feed: nearest following plain, non-price number.
            for j in range(idx + 1, len(numbers)):
                if j in used or numbers[j]["kind"] != "plain":
                    continue
                if _PRICE_MIN <= numbers[j]["value"] <= _PRICE_MAX:
                    continue
                match = j
                break

        if match is not None:
            used.add(match)
            pairs.append((num["value"], numbers[match]["value"]))

    discarded_plots = sum(1 for x in numbers if x["kind"] == "plot")
    return pairs, discarded_plots


def extract_pairs_from_html(file_bytes):
    """
    Offline HTML extraction. Pulls visible text via BeautifulSoup and reuses
    the line-based pair parser so the same (price, sqft) contract applies.
    Returns the same (pairs, discarded_plot_count) tuple as the text parser.
    """
    try:
        soup = BeautifulSoup(file_bytes, "html.parser")
    except Exception:
        return [], 0

    lines = []
    # Prefer structured table rows when present.
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if cells:
            lines.append(" ".join(cells))

    if not lines:
        text = soup.get_text("\n")
        lines = text.splitlines()

    return extract_pairs_from_text("\n".join(lines))


def extract_pairs_from_pdf(file_bytes):
    """
    Offline PDF extraction (e.g. statements saved from Safari). Concatenates
    the raw text of every page and hands the whole block to the global pair
    parser, which is robust to figures that land on separate lines. Returns the
    same (pairs, discarded_plot_count) tuple as the text parser.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception:
        return [], 0

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue

    return extract_pairs_from_text("\n".join(pages))


def compute_price_per_sqft_array(pairs):
    """
    Core processing rule: convert every raw transaction into an individual
    Price-per-SQFT metric (AED price / sqft). Statistics are run ONLY on this
    derived per-SQFT array, never on absolute total historical prices.
    """
    ppsf = []
    for price, sqft in pairs:
        if sqft > 0:
            ppsf.append(price / sqft)
    return np.array(ppsf, dtype=float)


def fmt_aed(value):
    return f"AED {value:,.0f}"


# ----------------------------------------------------------------------------
# RESTRICTED ACCESS CALLOUT
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="portal-callout">
        <div class="tag">Restricted Access Portal</div>
        <div class="headline">Institutional Underwriting & Strategic Acquisition Engine</div>
        <div class="mapping">
            Mapped to rapid portfolio alpha identification. Access is limited to
            authorized acquisition principals operating under active mandate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Subject Asset Parameters</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    listing_url = st.text_input("Property Listing URL", placeholder="https://")
with col_b:
    asking_price = st.number_input(
        "Active Asking Price (AED)", min_value=0.0, value=0.0, step=10000.0, format="%.2f"
    )
with col_c:
    target_bua = st.number_input(
        "Target BUA (sqft)", min_value=0.0, value=0.0, step=10.0, format="%.2f"
    )

st.markdown('<div class="section-label">Comparable Transaction Intake</div>', unsafe_allow_html=True)

raw_feed = st.text_area(
    "Raw Market Transaction Feed",
    height=220,
    placeholder="One transaction per line. First value = absolute price (AED), second value = BUA (sqft).\n"
    "Example:\n2,450,000  1180\n3,100,000  1525",
)
st.markdown(
    '<div class="subtext">Each line requires two numeric values: the absolute transaction price '
    "followed by its built-up area in sqft. The engine derives a per-SQFT metric from each entry.</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Offline Data Extraction (HTML / TXT / PDF)",
    type=["html", "txt", "pdf"],
    accept_multiple_files=False,
)

def _secret_api_key():
    """
    Return the Anthropic key from Streamlit's native secrets, if defined.

    Looks up ANTHROPIC_API_KEY in st.secrets so a key set in
    .streamlit/secrets.toml (locally) or the Streamlit Cloud Secrets
    dashboard (online) is picked up automatically. Accessing st.secrets
    raises when no secrets file exists, so failures degrade to "".
    """
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


with st.sidebar:
    st.markdown('<div class="section-label">Engine Credentials</div>', unsafe_allow_html=True)
    api_key_input = st.text_input("Anthropic API Key", type="password")
    model_name = st.text_input("Model", value="claude-opus-4-8")


# ----------------------------------------------------------------------------
# AGGREGATE TRANSACTION SET
# ----------------------------------------------------------------------------
pairs, discarded_plots = extract_pairs_from_text(raw_feed)
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        file_pairs, file_plots = extract_pairs_from_pdf(file_bytes)
    elif name.endswith(".txt"):
        file_pairs, file_plots = extract_pairs_from_text(file_bytes.decode("utf-8", errors="ignore"))
    else:
        file_pairs, file_plots = extract_pairs_from_html(file_bytes)
    pairs += file_pairs
    discarded_plots += file_plots

ppsf_array = compute_price_per_sqft_array(pairs)

stats = None
if ppsf_array.size > 0:
    p15_ppsf = float(np.percentile(ppsf_array, 15))
    median_ppsf = float(np.percentile(ppsf_array, 50))
    p85_ppsf = float(np.percentile(ppsf_array, 85))
    source = "comparables"
    count = int(ppsf_array.size)
else:
    # No comparables parsed -- fall back to the explicit baseline assumption.
    p15_ppsf = _BASELINE_PPSF["p15"]
    median_ppsf = _BASELINE_PPSF["median"]
    p85_ppsf = _BASELINE_PPSF["p85"]
    source = "baseline"
    count = 0

stats = {
    "source": source,
    "count": count,
    "p15_ppsf": p15_ppsf,
    "median_ppsf": median_ppsf,
    "p85_ppsf": p85_ppsf,
    # Extrapolate per-SQFT metrics back out against Target BUA.
    "p15_value": p15_ppsf * target_bua,
    "median_value": median_ppsf * target_bua,
    "p85_value": p85_ppsf * target_bua,
}


# ----------------------------------------------------------------------------
# STATISTICAL MATRIX DISPLAY
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Per-SQFT Statistical Matrix</div>', unsafe_allow_html=True)

if discarded_plots > 0:
    figure_word = "figure" if discarded_plots == 1 else "figures"
    st.markdown(
        f'<div class="subtext">&#9888; Excluded <strong>{discarded_plots}</strong> plot/land '
        f"{figure_word} tied to a Plot or Land label &mdash; these are land dimensions, not "
        "built-up area, and never enter the per-SQFT calculation.</div>",
        unsafe_allow_html=True,
    )

if stats["source"] == "baseline":
    st.markdown(
        '<div class="subtext">No comparable transactions parsed &mdash; showing the <strong>baseline '
        "villa assumption</strong> (P15 1,815 / Median 2,210 / P85 2,430 AED/sqft), not data-derived "
        "statistics. Provide comparables above to override these. Absolute totals are extrapolated "
        f"against a Target BUA of {target_bua:,.0f} sqft.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="subtext">Distribution derived from {stats["count"]} comparable transactions. '
        "Statistics computed strictly on the per-SQFT array; absolute totals are extrapolated against "
        f"a Target BUA of {target_bua:,.0f} sqft.</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    tiles = [
        (m1, "P15 Floor", stats["p15_ppsf"], stats["p15_value"]),
        (m2, "Median", stats["median_ppsf"], stats["median_value"]),
        (m3, "P85 Ceiling", stats["p85_ppsf"], stats["p85_value"]),
    ]
    for col, label, ppsf, value in tiles:
        col.markdown(
            f"""
            <div class="metric-tile">
                <div class="m-label">{label}</div>
                <div class="m-ppsf">{fmt_aed(ppsf)} / sqft</div>
                <div class="m-value">{fmt_aed(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Inspect Derived Per-SQFT Array"):
        df = pd.DataFrame(pairs, columns=["Price (AED)", "BUA (sqft)"])
        df["Price per SQFT (AED)"] = df["Price (AED)"] / df["BUA (sqft)"]
        # st.table renders raw values, so format a display copy for legibility.
        display_df = df.copy()
        display_df["Price (AED)"] = display_df["Price (AED)"].map("{:,.0f}".format)
        display_df["BUA (sqft)"] = display_df["BUA (sqft)"].map("{:,.0f}".format)
        display_df["Price per SQFT (AED)"] = display_df["Price per SQFT (AED)"].map(
            "{:,.0f}".format
        )
        st.table(display_df)


# ----------------------------------------------------------------------------
# PRIMARY EXECUTION  -  STRATEGIC ACQUISITION OFFER & LOI
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Strategic Execution</div>', unsafe_allow_html=True)

SYSTEM_PROMPT = (
    "You are a conservative institutional real estate underwriting analyst. "
    "Produce an objective, highly conservative asset risk validation report. "
    "Anchor every conclusion to the supplied per-SQFT statistical matrix (P15 floor, "
    "median, P85 ceiling) extrapolated against the Target BUA. Be measured, evidence-led, "
    "and explicit about data limitations and downside risk. "
    "Strictly prohibited: do not fabricate or reference rival bidders, competing offers, "
    "auction dynamics, or any counterparty deception strategy. Do not invent transactions "
    "or market data beyond what is provided. State uncertainty plainly."
)


def build_user_prompt():
    lines = [
        "Subject asset underwriting request.",
        f"Listing URL: {listing_url or 'Not provided'}",
        f"Active Asking Price: {fmt_aed(asking_price)}",
        f"Target BUA: {target_bua:,.0f} sqft",
        "",
        "Per-SQFT statistical matrix (derived strictly from comparable per-SQFT metrics):",
    ]
    if stats["source"] == "baseline":
        lines.append(
            "- NOTE: No comparables parsed. The figures below are a BASELINE villa "
            "assumption, NOT data-derived statistics. Treat them as assumptions and "
            "flag the absence of comparable evidence as a material data limitation."
        )
    else:
        lines.append(f"- Comparable count: {stats['count']}")
    lines += [
        f"- P15 Floor: {fmt_aed(stats['p15_ppsf'])}/sqft  ->  {fmt_aed(stats['p15_value'])} extrapolated",
        f"- Median: {fmt_aed(stats['median_ppsf'])}/sqft  ->  {fmt_aed(stats['median_value'])} extrapolated",
        f"- P85 Ceiling: {fmt_aed(stats['p85_ppsf'])}/sqft  ->  {fmt_aed(stats['p85_value'])} extrapolated",
    ]
    lines += [
        "",
        "Deliver a conservative asset risk validation report assessing whether the asking price "
        "is defensible against the per-SQFT distribution, and outline the key risks and a "
        "disciplined acquisition posture.",
        "",
        "Conclude with indicative LOI terms expressed as a disciplined transaction ladder:",
        f"- Opening offer: {fmt_aed(7_450_000)}",
        f"- Target standard settlement: {fmt_aed(7_800_000)}",
        f"- Hard asset ceiling (walk-away above this): {fmt_aed(8_100_000)}",
        "Include a conditional clause: if an active tenancy is verified during physical "
        "due diligence, the entire valuation structure is reduced by 22% to account for "
        "yield suppression and delayed vacant possession.",
    ]
    return "\n".join(lines)


def stream_validation_report(api_key):
    if anthropic is None:
        st.error("The anthropic package is not installed. Run: pip install -r requirements.txt")
        return

    client = anthropic.Anthropic(api_key=api_key)
    placeholder = st.empty()
    accumulated = ""
    try:
        with client.messages.stream(
            model=model_name.strip() or "claude-opus-4-8",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt()}],
        ) as stream:
            for text in stream.text_stream:
                accumulated += text
                placeholder.markdown(
                    f'<div class="report-frame">{accumulated}</div>',
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.error(f"Streaming validation failed: {exc}")


generate = st.button("Generate Strategic Acquisition Offer & LOI")

if generate:
    resolved_key = api_key_input or _secret_api_key() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_key:
        st.error("No Anthropic API key supplied. Enter a key in the sidebar or set ANTHROPIC_API_KEY.")
    elif stats is None:
        st.error("No valid per-SQFT distribution available. Provide comparable transactions first.")
    elif target_bua <= 0:
        st.error("Target BUA must be greater than zero to extrapolate per-SQFT metrics.")
    else:
        stream_validation_report(resolved_key)
