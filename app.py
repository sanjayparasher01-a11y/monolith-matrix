"""
Institutional-Grade Real Estate Underwriting Portal
Price-per-SQFT statistical underwriting engine with Claude streaming validation.
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

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

/* Dark Slate input components */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background-color: #172A45 !important;
    color: #FFFFFF !important;
    border: 1px solid #233554 !important;
    border-radius: 4px !important;
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
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.95rem 1rem !important;
    font-size: 0.85rem !important;
}
div.stButton > button:hover {
    background-color: #E6E9F0 !important;
    color: #0A192F !important;
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
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _to_float(token):
    """Convert a numeric token (with thousands separators) to float."""
    try:
        return float(token.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def extract_pairs_from_text(raw_text):
    """
    Parse a raw transaction feed into (price, sqft) pairs.

    Each non-empty line is expected to expose at least two numeric values;
    the first is treated as the absolute transaction price and the second as
    the built-up area in sqft. Lines without two valid numbers are skipped.
    """
    pairs = []
    if not raw_text:
        return pairs

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = _NUM_RE.findall(line)
        nums = [v for v in (_to_float(t) for t in tokens) if v is not None]
        if len(nums) >= 2:
            price, sqft = nums[0], nums[1]
            if price > 0 and sqft > 0:
                pairs.append((price, sqft))
    return pairs


def extract_pairs_from_html(file_bytes):
    """
    Offline HTML extraction. Pulls visible text via BeautifulSoup and reuses
    the line-based pair parser so the same (price, sqft) contract applies.
    """
    try:
        soup = BeautifulSoup(file_bytes, "html.parser")
    except Exception:
        return []

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

uploaded_html = st.file_uploader(
    "Offline HTML Data Extraction", type=["html", "htm"], accept_multiple_files=False
)

with st.sidebar:
    st.markdown('<div class="section-label">Engine Credentials</div>', unsafe_allow_html=True)
    api_key_input = st.text_input("Anthropic API Key", type="password")
    model_name = st.text_input("Model", value="claude-opus-4-8")


# ----------------------------------------------------------------------------
# AGGREGATE TRANSACTION SET
# ----------------------------------------------------------------------------
pairs = extract_pairs_from_text(raw_feed)
if uploaded_html is not None:
    pairs += extract_pairs_from_html(uploaded_html.getvalue())

ppsf_array = compute_price_per_sqft_array(pairs)

stats = None
if ppsf_array.size > 0:
    p15_ppsf = float(np.percentile(ppsf_array, 15))
    median_ppsf = float(np.percentile(ppsf_array, 50))
    p85_ppsf = float(np.percentile(ppsf_array, 85))
    stats = {
        "count": int(ppsf_array.size),
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

if stats is None:
    st.markdown(
        '<div class="subtext">No valid transactions parsed. Provide price and sqft pairs above '
        "to derive the per-SQFT distribution.</div>",
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
        st.dataframe(df, use_container_width=True)


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
    if stats:
        lines += [
            f"- Comparable count: {stats['count']}",
            f"- P15 Floor: {fmt_aed(stats['p15_ppsf'])}/sqft  ->  {fmt_aed(stats['p15_value'])} extrapolated",
            f"- Median: {fmt_aed(stats['median_ppsf'])}/sqft  ->  {fmt_aed(stats['median_value'])} extrapolated",
            f"- P85 Ceiling: {fmt_aed(stats['p85_ppsf'])}/sqft  ->  {fmt_aed(stats['p85_value'])} extrapolated",
        ]
    else:
        lines.append("- No comparable transactions parsed.")
    lines += [
        "",
        "Deliver a conservative asset risk validation report assessing whether the asking price "
        "is defensible against the per-SQFT distribution, and outline the key risks and a "
        "disciplined acquisition posture. Conclude with indicative LOI terms grounded in the "
        "P15 floor and median.",
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
            max_tokens=2000,
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
    resolved_key = api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_key:
        st.error("No Anthropic API key supplied. Enter a key in the sidebar or set ANTHROPIC_API_KEY.")
    elif stats is None:
        st.error("No valid per-SQFT distribution available. Provide comparable transactions first.")
    elif target_bua <= 0:
        st.error("Target BUA must be greater than zero to extrapolate per-SQFT metrics.")
    else:
        stream_validation_report(resolved_key)
