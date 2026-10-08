"""
app.py — Streamlit UI for the Multi-Agent Supply Chain Disruption Pipeline
Light "Maritime Chart Room" Theme (High-Contrast Refined)
"""

import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── env must be loaded before any agent imports ───────────────────────────────
load_dotenv()

# ── project imports ───────────────────────────────────────────────────────────
from agents.impact_agent import load_routes_knowledge_base
from main import run_pipeline, _get_agents

# ── constants ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
EVENTS_FILE = ROOT / "data" / "sample_news_events.json"


# ══════════════════════════════════════════════════════════════════════════════
# HTML Rendering Helper
# ══════════════════════════════════════════════════════════════════════════════

def render_html(html_str: str):
    """
    Renders HTML in Streamlit safely.
    Strips leading/trailing indentation from each line to prevent Markdown-it
    from treating 4-space indented HTML blocks as <pre><code> codeblocks.
    """
    cleaned = "\n".join(line.strip() for line in str(html_str).strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Warming up Maritime Intelligence Engine…")
def get_agents_cached():
    """Warm up all 4 CrewAI agents once and cache them for the session."""
    _get_agents()


@st.cache_data(show_spinner=False)
def load_events() -> list[dict]:
    with open(EVENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_routes() -> list[dict]:
    return load_routes_knowledge_base()


# ══════════════════════════════════════════════════════════════════════════════
# Custom Maritime Chart Room CSS (Light Theme — High Contrast)
# ══════════════════════════════════════════════════════════════════════════════

def inject_chart_room_css():
    render_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

        :root {
            --bg-paper: #EFEAD9;
            --panel-paper: #FAF8F2;
            --ink-navy: #16263B;
            --ink-slate: #2C3E50;
            --ink-muted: #3D4F61;
            --accent-brass: #B8841F;
            --accent-brass-dark: #8C6210;
            --accent-teal: #2C6360;
            --accent-teal-dark: #1B4543;
            --alert-crimson: #B2483A;
            --border-navy: #16263B;
            --border-subtle: rgba(22, 38, 59, 0.25);
            --badge-bg-parchment: #E2DCC9;
        }

        /* Base App Background & Ink Typography */
        .stApp {
            background-color: var(--bg-paper) !important;
            color: var(--ink-navy) !important;
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #E6E0CC !important;
            border-right: 1px solid var(--border-navy) !important;
            color: var(--ink-navy) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            font-family: 'Fraunces', Georgia, serif !important;
            color: var(--ink-navy) !important;
            letter-spacing: -0.01em;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: var(--ink-navy) !important;
        }

        /* Captions across app */
        .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: var(--ink-muted) !important;
            font-weight: 500 !important;
        }

        /* Headers */
        h1, h2, h3, h4, .serif-title {
            font-family: 'Fraunces', Georgia, serif !important;
            color: var(--ink-navy) !important;
            font-weight: 600 !important;
        }

        /* Monospace elements */
        code, pre, .mono-text {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* Code snippets tag replacement */
        code {
            background-color: var(--badge-bg-parchment) !important;
            color: var(--ink-navy) !important;
            padding: 2px 6px !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 0px !important;
            font-size: 0.88em !important;
            font-weight: 500 !important;
        }

        /* Maritime Chart Paper Content Panel */
        .chart-panel {
            background-color: var(--panel-paper);
            border: 1px solid var(--border-navy);
            border-radius: 0px;
            padding: 20px 24px;
            margin-bottom: 20px;
            color: var(--ink-navy);
            box-shadow: 2px 2px 0px rgba(22, 38, 59, 0.08);
        }

        /* Manifest Log Sidebar Entry */
        .manifest-log-card {
            background-color: var(--panel-paper);
            border: 1px solid var(--border-subtle);
            border-left: 4px solid var(--accent-brass);
            padding: 12px 14px;
            margin-top: 10px;
            margin-bottom: 12px;
            font-size: 0.88rem;
            line-height: 1.45;
            color: var(--ink-navy);
        }

        /* Progress Rail (Horizontal Connector) */
        .rail-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            margin: 24px 0 32px 0;
            padding: 0 10px;
        }

        .rail-track {
            position: absolute;
            top: 20px;
            left: 5%;
            right: 5%;
            height: 1px;
            background-color: var(--border-navy);
            z-index: 1;
        }

        .rail-node {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            width: 22%;
        }

        .rail-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: var(--panel-paper);
            border: 1px solid var(--border-navy);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--ink-navy);
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }

        .rail-circle.active {
            background-color: var(--accent-brass);
            color: #FFFFFF;
            border-color: var(--border-navy);
            box-shadow: 0 0 0 3px rgba(184, 132, 31, 0.25);
        }

        .rail-circle.complete {
            background-color: var(--accent-teal);
            color: #FFFFFF;
            border-color: var(--border-navy);
        }

        .rail-circle.skipped {
            background-color: #E2DCC9;
            color: var(--ink-muted);
            border-color: var(--border-navy);
            border-style: dashed;
        }

        .rail-label {
            font-family: 'Fraunces', Georgia, serif;
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--ink-navy);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .rail-status {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            color: var(--ink-muted);
            font-weight: 500;
            margin-top: 2px;
        }

        /* Disruption Manifest Summary Ticket */
        .manifest-ticket {
            background-color: var(--panel-paper);
            border: 1px solid var(--border-navy);
            border-radius: 0px;
            padding: 24px 28px;
            margin-top: 30px;
            position: relative;
            box-shadow: 3px 3px 0px rgba(22, 38, 59, 0.10);
        }

        .manifest-ticket::before {
            content: "DISRUPTION MANIFEST";
            position: absolute;
            top: -12px;
            left: 24px;
            background-color: var(--ink-navy);
            color: #FAF8F2;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            padding: 2px 10px;
            font-weight: 600;
        }

        /* Tags & Badges */
        .badge-tag {
            display: inline-block;
            padding: 3px 9px;
            border: 1px solid var(--border-navy);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-right: 6px;
            margin-bottom: 4px;
            border-radius: 0px;
            background-color: var(--badge-bg-parchment);
            color: var(--ink-navy);
        }

        .badge-severity-high {
            background-color: var(--alert-crimson) !important;
            color: #FFFFFF !important;
            border-color: var(--border-navy) !important;
        }

        .badge-severity-medium {
            background-color: var(--accent-brass) !important;
            color: #FFFFFF !important;
            border-color: var(--border-navy) !important;
        }

        .badge-severity-low {
            background-color: var(--accent-teal) !important;
            color: #FFFFFF !important;
            border-color: var(--border-navy) !important;
        }

        .badge-feasibility-high {
            background-color: #DDEBE7 !important;
            color: var(--accent-teal-dark) !important;
            border-color: var(--accent-teal) !important;
        }

        .badge-feasibility-medium {
            background-color: #F8F1DE !important;
            color: var(--accent-brass-dark) !important;
            border-color: var(--accent-brass) !important;
        }

        .badge-feasibility-low {
            background-color: #F9EAE7 !important;
            color: #8F2B1F !important;
            border-color: var(--alert-crimson) !important;
        }

        /* Recommendation Quote Block */
        .rec-callout {
            background-color: #F5EFE0;
            border-left: 4px solid var(--accent-brass);
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            padding: 16px 20px;
            margin-top: 14px;
            font-size: 0.98rem;
            line-height: 1.6;
            color: var(--ink-navy);
        }

        /* Skip explanation banner */
        .skip-banner {
            background-color: #E6DFCB;
            border: 1px dashed var(--border-navy);
            padding: 14px 18px;
            font-size: 0.9rem;
            color: var(--ink-navy);
            font-family: 'IBM Plex Sans', sans-serif;
            margin-bottom: 12px;
        }

        /* Streamlit Button Restyling */
        div.stButton > button:first-child {
            background-color: var(--accent-brass) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--border-navy) !important;
            border-radius: 0px !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            padding: 10px 24px !important;
            box-shadow: 2px 2px 0px var(--border-navy) !important;
            transition: all 0.15s ease !important;
        }

        div.stButton > button:first-child:hover {
            background-color: var(--accent-brass-dark) !important;
            color: #FFFFFF !important;
            box-shadow: 1px 1px 0px var(--border-navy) !important;
            transform: translate(1px, 1px);
        }

        /* Stage Section Header Bars (High Contrast Deep Navy + Crisp Ivory/White) */
        div[data-testid="stExpander"] {
            background-color: var(--panel-paper) !important;
            border: 1px solid var(--border-navy) !important;
            border-radius: 0px !important;
            margin-bottom: 16px !important;
            box-shadow: 2px 2px 0px rgba(22, 38, 59, 0.08) !important;
            overflow: hidden !important;
        }

        div[data-testid="stExpander"] details {
            border: none !important;
            background-color: transparent !important;
        }

        /* Dark Horizontal Header Bar */
        div[data-testid="stExpander"] details summary,
        div[data-testid="stExpander"] summary {
            background-color: var(--ink-navy) !important; /* #16263B */
            color: #FAF8F2 !important; /* Crisp Ivory */
            padding: 12px 18px !important;
            border-bottom: 1px solid var(--border-navy) !important;
            border-radius: 0px !important;
            cursor: pointer !important;
            opacity: 1 !important;
            list-style: none !important;
            list-style-type: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }

        div[data-testid="stExpander"] summary::-webkit-details-marker,
        div[data-testid="stExpander"] details summary::-webkit-details-marker {
            display: none !important;
        }

        /* Header Bar Hover State */
        div[data-testid="stExpander"] details summary:hover,
        div[data-testid="stExpander"] summary:hover {
            background-color: #20354E !important;
            color: #FFFFFF !important;
        }

        /* ── Expander Header: Title Text ────────────────────────────────────────
           DOM structure confirmed from Streamlit 1.65 JS bundle (index.CKCTizkM.js):
             <summary>                 ← Nu  = f('summary', {target:'e1u8u9eb10'})
               <span>                 ← wu  = f('span',    {target:'e1u8u9eb3'})  flex container
                 <span>...</span>     ← R icon component (size='base') — FIRST child of wu
                 <div>                ← Tu  = f('div',     {target:'e1u8u9eb4'})  title wrapper
                   <div>              ← Ht markdown container
                     <p>Title</p>
                   </div>
                 </div>
               </span>
             </summary>
           Structural selectors used below are stable across Streamlit restarts because
           they are based on element types and data-testid, NOT Emotion hash class names.
           ──────────────────────────────────────────────────────────────────────────── */

        /* Style the <p> that Streamlit's markdown renderer places inside the title div */
        div[data-testid="stExpander"] summary p {
            color: #FAF8F2 !important;
            -webkit-text-fill-color: #FAF8F2 !important;
            font-family: 'Fraunces', Georgia, serif !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.02em !important;
            opacity: 1 !important;
        }

        /* Style the Tu title-wrapper div (first div inside the wu span, inside summary).
           Selector: summary > span > div  →  exact structural path from JS analysis.
           Do NOT use > span > span here — that would hit the icon component span. */
        div[data-testid="stExpander"] summary > span > div {
            color: #FAF8F2 !important;
            -webkit-text-fill-color: #FAF8F2 !important;
            font-family: 'Fraunces', Georgia, serif !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.02em !important;
        }

        /* Style all descendants of the title wrapper div (the Ht markdown container + p) */
        div[data-testid="stExpander"] summary > span > div * {
            color: #FAF8F2 !important;
            -webkit-text-fill-color: #FAF8F2 !important;
            font-family: 'Fraunces', Georgia, serif !important;
            font-weight: 600 !important;
        }

        /* ── Expander Header: Hide Icon ──────────────────────────────────────────
           The R icon component (size='base') is the FIRST child of the wu <span>.
           Selector: summary > span > *:first-child  →  exact structural path.
           We do NOT override font-family here — we keep Material Symbols Rounded
           intact so the icon font can still render if loaded. We just collapse its
           visual space to zero so it's invisible regardless of font load state.
           ──────────────────────────────────────────────────────────────────────── */
        div[data-testid="stExpander"] summary > span > *:first-child {
            font-size: 0 !important;
            width: 0 !important;
            height: 0 !important;
            min-width: 0 !important;
            min-height: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
            opacity: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            flex: 0 0 0 !important;
            display: inline-block !important;
        }

        /* Hide step-expander chevron (different testid, different path) and stray SVGs */
        div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
        div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] *,
        div[data-testid="stExpander"] summary svg {
            display: none !important;
        }

        /* Custom pure Unicode arrow indicator (No external font dependency) */
        div[data-testid="stExpander"] details summary::after,
        div[data-testid="stExpander"] summary::after {
            content: "▸" !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: bold !important;
            color: #FAF8F2 !important;
            -webkit-text-fill-color: #FAF8F2 !important;
            margin-left: auto !important;
            padding-left: 12px !important;
            display: inline-block !important;
            line-height: 1 !important;
            flex-shrink: 0 !important;
        }

        div[data-testid="stExpander"] details[open] summary::after {
            content: "▾" !important;
        }

        /* Expander Body Panel */
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: var(--panel-paper) !important;
            color: var(--ink-navy) !important;
            padding: 20px 22px !important;
        }

        /* Streamlit Metric Container Restyling */
        div[data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--ink-navy) !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricLabel"] {
            font-family: 'IBM Plex Sans', sans-serif !important;
            color: var(--ink-muted) !important;
            text-transform: uppercase !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.06em !important;
            font-weight: 600 !important;
        }

        /* Radio buttons & Selectbox styling */
        div[data-testid="stRadio"] label, div[data-testid="stRadio"] span, div[data-testid="stRadio"] p {
            color: var(--ink-navy) !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: var(--panel-paper) !important;
            color: var(--ink-navy) !important;
            border: 1px solid var(--border-navy) !important;
        }

        div[data-baseweb="select"] span {
            color: var(--ink-navy) !important;
        }

        /* Text Input & Text Area */
        div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
            background-color: var(--panel-paper) !important;
            color: var(--ink-navy) !important;
            border: 1px solid var(--border-navy) !important;
        }

        hr {
            border-color: var(--border-navy) !important;
            opacity: 0.3 !important;
        }
        </style>
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage Renderers (Chart-Paper Panels)
# ══════════════════════════════════════════════════════════════════════════════

def render_progress_rail(result: dict | None):
    """Horizontal 4-stage rail (Monitor → Impact → Routing → Cost) connected by line."""
    if result is None:
        stages = [
            ("1", "Monitor", "Ready", "pending"),
            ("2", "Impact", "Ready", "pending"),
            ("3", "Routing", "Ready", "pending"),
            ("4", "Cost", "Ready", "pending"),
        ]
    else:
        monitor = result.get("monitor", {})
        impact  = result.get("impact", {})
        routing = result.get("routing", {})
        cost    = result.get("cost", {})

        m_ok = bool(monitor.get("is_disruption_relevant"))
        i_ok = bool(impact.get("matched_route"))
        r_ok = not bool(routing.get("skipped"))
        c_ok = not bool(cost.get("skipped"))

        stages = [
            ("1", "Monitor", "Disruption Found" if m_ok else "Not Relevant", "complete" if m_ok else "skipped"),
            ("2", "Impact", "Route Matched" if i_ok else "Skipped", "complete" if i_ok else ("skipped" if impact.get("skipped") else "active")),
            ("3", "Routing", "Route Analyzed" if r_ok else "Skipped", "complete" if r_ok else "skipped"),
            ("4", "Cost", "Assessed" if c_ok else "Skipped", "complete" if c_ok else "skipped"),
        ]

    nodes_html = ""
    for num, label, status_text, state_class in stages:
        icon = "✓" if state_class == "complete" else ("—" if state_class == "skipped" else num)
        nodes_html += f"""
        <div class="rail-node">
            <div class="rail-circle {state_class}">{icon}</div>
            <div class="rail-label">{label}</div>
            <div class="rail-status">{status_text}</div>
        </div>
        """

    render_html(
        f"""
        <div class="rail-container">
            <div class="rail-track"></div>
            {nodes_html}
        </div>
        """
    )


def show_monitor_stage(data: dict):
    if data.get("skipped"):
        render_html(f'<div class="skip-banner"><strong>STAGE SKIPPED:</strong> {data["skipped"]}</div>')
        return
    if data.get("_error"):
        st.error(f"Agent Execution Error: {data['_error']}")
        return

    relevant = data.get("is_disruption_relevant")
    sev = (data.get("severity") or "").lower()

    badge_class = f"badge-severity-{sev}" if sev in ("high", "medium", "low") else "badge-tag"
    status_html = '<span class="badge-tag" style="background-color:#2C6360;color:#FFFFFF;">CONFIRMED DISRUPTION</span>' if relevant else '<span class="badge-tag" style="background-color:#E2DCC9;color:#16263B;">NON-DISRUPTION</span>'

    col1, col2 = st.columns([1, 1])
    with col1:
        render_html(f"<div style='margin-bottom:10px;'><strong>Disruption Status:</strong> {status_html}</div>")
        event_type = data.get('event_type') or '—'
        render_html(f"<div style='margin-bottom:10px;'><strong>Event Classification:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>{event_type}</span></div>")
        location = data.get('location') or '—'
        render_html(f"<div style='margin-bottom:10px;'><strong>Geographic Epicenter:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>{location}</span></div>")
    with col2:
        sev_label = sev.upper() if sev else '—'
        render_html(f"<div style='margin-bottom:10px;'><strong>Disruption Severity:</strong> <span class=\"badge-tag {badge_class}\">{sev_label}</span></div>")
        date_str = data.get('date') or '—'
        render_html(f"<div style='margin-bottom:10px;'><strong>Log Date:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>{date_str}</span></div>")
        conf = data.get("confidence")
        if conf is not None:
            render_html(f"<div style='margin-bottom:10px;'><strong>Monitor Confidence:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>{conf:.0%}</span></div>")


def show_impact_stage(data: dict):
    if data.get("skipped"):
        render_html(f'<div class="skip-banner"><strong>STAGE SKIPPED:</strong> {data["skipped"]}</div>')
        return
    if data.get("_error"):
        st.error(f"Agent Execution Error: {data['_error']}")
        return

    route = data.get("matched_route")
    if not route:
        render_html(
            """
            <div class="skip-banner">
                <strong>NO MATCHED CHOKEPOINT:</strong> Location was not matched to a known trade corridor in the knowledge base (confidence &lt; 0.5).
            </div>
            """
        )
        reason = data.get("reasoning", "")
        if reason:
            render_html(f"<div style='color:#2C3E50; font-size:0.9rem; margin-top:6px;'><strong>Reasoning:</strong> {reason}</div>")
        return

    render_html(
        f"""
        <div style="border-bottom: 1px solid var(--border-subtle); padding-bottom: 8px; margin-bottom: 12px;">
            <span style="font-family:'Fraunces',serif; font-size:1.18rem; font-weight:600; color:var(--accent-teal);">
                ⚓ {route}
            </span>
        </div>
        """
    )

    cats = data.get("affected_product_categories", [])
    if cats:
        render_html("<div style='margin-bottom:6px; font-weight:600; color:#16263B;'>Compromised Commodity Sectors:</div>")
        cat_pills = "".join([f'<span class="badge-tag" style="background:#FAF8F2; color:#16263B; border:1px solid #16263B; margin-right:6px; margin-bottom:6px;">{c}</span>' for c in cats])
        render_html(f"<div style='margin-bottom:12px;'>{cat_pills}</div>")

    conf = data.get("impact_confidence")
    if conf is not None:
        render_html(f"<div style='margin-bottom:10px;'><strong>Impact Correlation Confidence:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>{conf:.0%}</span></div>")

    reasoning = data.get("reasoning", "")
    if reasoning:
        with st.expander("Impact Assessment Analysis", expanded=False):
            render_html(f"<div style='color:#16263B; line-height:1.55;'>{reasoning}</div>")


def show_routing_stage(data: dict):
    if data.get("skipped"):
        render_html(f'<div class="skip-banner"><strong>STAGE SKIPPED:</strong> {data["skipped"]}</div>')
        return
    if data.get("_error"):
        st.error(f"Agent Execution Error: {data['_error']}")
        return

    alts = data.get("alternatives", [])
    rec  = data.get("recommended_alternative", "")

    if not alts:
        rec_display = rec or 'Physical geography restricts maritime rerouting for this corridor.'
        render_html(
            f"""
            <div style="background-color:#F9EAE7; border:1px solid var(--alert-crimson); padding:14px 18px; color:#16263B;">
                <strong style="color:#8F2B1F;">🚧 NO MARITIME BYPASS AVAILABLE</strong>
                <p style="margin-top:6px; font-size:0.92rem; line-height:1.5; color:#16263B;">{rec_display}</p>
            </div>
            """
        )
        return

    if rec:
        render_html(
            f"""
            <div style="border-left:4px solid var(--accent-brass); background-color:#F7F3E6; border-top:1px solid var(--border-subtle); border-right:1px solid var(--border-subtle); border-bottom:1px solid var(--border-subtle); padding:12px 16px; margin-bottom:16px;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; color:var(--accent-brass-dark); font-weight:600;">Top Strategic Recommendation</span>
                <div style="font-size:0.96rem; font-weight:500; margin-top:3px; color:#16263B; line-height:1.5;">{rec}</div>
            </div>
            """
        )

    render_html(f"<div style='margin-bottom:8px; font-weight:600; color:#16263B;'>Identified Maritime Rerouting Corridors ({len(alts)}):</div>")
    for alt in alts:
        feas = (alt.get("feasibility") or "").lower()
        badge_cls = f"badge-feasibility-{feas}" if feas in ("high", "medium", "low") else "badge-tag"
        feas_label = feas.upper()
        alt_name = alt.get('route', '—')
        alt_rat = alt.get('rationale', '')
        render_html(
            f"""
            <div style="border:1px solid var(--border-navy); background:#FAF8F2; padding:14px 18px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'Fraunces',serif; font-size:1.05rem; font-weight:600; color:#16263B;">{alt_name}</span>
                    <span class="badge-tag {badge_cls}">{feas_label} FEASIBILITY</span>
                </div>
                <div style="font-size:0.90rem; color:#16263B; margin-top:8px; line-height:1.55;">
                    {alt_rat}
                </div>
            </div>
            """
        )


def show_cost_stage(data: dict):
    if data.get("skipped"):
        render_html(f'<div class="skip-banner"><strong>STAGE SKIPPED:</strong> {data["skipped"]}</div>')
        return
    if data.get("_error"):
        st.error(f"Agent Execution Error: {data['_error']}")
        return

    cpd     = data.get("disruption_cost_per_day_usd")
    days    = data.get("estimated_delay_days")
    total   = data.get("total_disruption_cost_usd")
    reroute = data.get("rerouting_cost_estimate_usd")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Disruption Cost / Day", f"${cpd:,.0f}" if cpd is not None else "—")
    with c2:
        st.metric("Estimated Voyage Delay", f"{days} Days" if days is not None else "—")
    with c3:
        st.metric("Total Absorption Cost", f"${total:,.0f}" if total is not None else "—")

    if reroute is not None:
        render_html(f"<div style='margin-top:8px;'><strong>One-Time Bunker & Switching Surcharge:</strong> <span class='badge-tag' style='background:#E2DCC9; color:#16263B;'>${reroute:,.0f}</span></div>")
    else:
        render_html("<div style='color:#3D4F61; font-size:0.9rem; margin-top:8px;'>Rerouting Surcharge: <em>N/A (Chokepoint has no sea bypass)</em></div>")

    rec = data.get("recommendation", "")
    if rec:
        render_html(f'<div class="rec-callout"><strong>Trade-Off Appraisal:</strong><br><span style="color:#16263B;">{rec}</span></div>')


def show_manifest_summary_ticket(result: dict):
    """Bordered DISRUPTION MANIFEST summary ticket at the end."""
    monitor = result.get("monitor", {})
    impact  = result.get("impact", {})
    routing = result.get("routing", {})
    cost    = result.get("cost", {})

    if not isinstance(monitor, dict) or not monitor.get("is_disruption_relevant"):
        return

    route = impact.get("matched_route") if isinstance(impact, dict) else None
    if not route:
        return

    total     = cost.get("total_disruption_cost_usd") if isinstance(cost, dict) else None
    reroute   = cost.get("rerouting_cost_estimate_usd") if isinstance(cost, dict) else None
    final_rec = cost.get("recommendation") if isinstance(cost, dict) else None
    sev       = (monitor.get("severity") or "MEDIUM").upper()
    cats      = impact.get("affected_product_categories", []) if isinstance(impact, dict) else []

    savings_html = ""
    if total and reroute:
        net = total - reroute
        if net > 0:
            savings_html = f"""
            <div style="margin-top:10px; font-family:'IBM Plex Mono',monospace; font-size:0.92rem; color:var(--accent-teal); font-weight:600;">
                ▶ NET REROUTING BENEFIT: +${net:,.0f}
            </div>
            """

    total_str = f"${total:,.0f}" if total is not None else "—"
    reroute_str = f"${reroute:,.0f}" if reroute is not None else "N/A (No Bypass)"
    sev_badge_cls = 'badge-severity-high' if sev == 'HIGH' else ('badge-severity-medium' if sev == 'MEDIUM' else 'badge-severity-low')
    rec_text = final_rec or 'Maintain holding positions and monitor canal authority bulletins.'

    render_html(
        f"""
        <div class="manifest-ticket">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--border-navy); padding-bottom:12px; margin-bottom:16px;">
                <div>
                    <h3 style="margin:0; font-family:'Fraunces',serif; font-size:1.45rem; color:#16263B;">{route}</h3>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#3D4F61; font-weight:600;">CORRIDOR SECTOR ASSESSMENT</span>
                </div>
                <div style="text-align:right;">
                    <span class="badge-tag {sev_badge_cls}">{sev} SEVERITY</span>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-bottom:18px;">
                <div style="border-right:1px solid var(--border-subtle); padding-right:10px;">
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.75rem; text-transform:uppercase; color:#3D4F61; font-weight:600;">Total Impact Liability</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:600; color:#16263B;">{total_str}</div>
                </div>
                <div style="border-right:1px solid var(--border-subtle); padding-right:10px;">
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.75rem; text-transform:uppercase; color:#3D4F61; font-weight:600;">Switching Cost</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:600; color:#16263B;">{reroute_str}</div>
                </div>
                <div>
                    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.75rem; text-transform:uppercase; color:#3D4F61; font-weight:600;">Sectors Affected</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:600; color:#16263B;">{len(cats)} Product Lines</div>
                </div>
            </div>

            {savings_html}

            <div style="margin-top:16px; padding-top:14px; border-top:1px solid var(--border-subtle);">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.74rem; text-transform:uppercase; color:var(--accent-brass-dark); font-weight:600; margin-bottom:4px;">Action Directive</div>
                <div style="font-size:0.96rem; line-height:1.6; color:#16263B;">{rec_text}</div>
            </div>
        </div>
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Maritime Chart Room — Supply Chain Disruption Engine",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_chart_room_css()

    # Pre-cache agents and data
    get_agents_cached()
    events      = load_events()
    routes_data = load_routes()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        render_html(
            """
            <div style="margin-bottom: 12px;">
                <h2 style="margin: 0; color: #16263B; font-family: 'Fraunces', serif;">🧭 CHART ROOM</h2>
                <div style="color: #3D4F61; font-size: 0.86rem; font-weight: 500; margin-top: 2px;">
                    Maritime Multi-Agent Supply Chain Intelligence
                </div>
            </div>
            """
        )
        st.divider()

        source = st.radio(
            "DISRUPTION FEED SOURCE",
            options=["📋 Manifest Log (12 Events)", "✏️ Custom Dispatch"],
            horizontal=False,
        )

        selected_event = None

        if source == "📋 Manifest Log (12 Events)":
            headlines = [e.get("headline", f"Event {i+1}") for i, e in enumerate(events)]
            idx = st.selectbox(
                "Select Intelligence Dispatch",
                options=range(len(headlines)),
                format_func=lambda i: f"[{i+1:02d}] {headlines[i][:50]}…"
                    if len(headlines[i]) > 50 else f"[{i+1:02d}] {headlines[i]}",
            )
            selected_event = events[idx]

            # Manifest log sidebar card
            event_raw_snip = selected_event.get('raw_text', '')[:160]
            render_html(
                f"""
                <div class="manifest-log-card">
                    <strong style="color:#16263B;">DISPATCH #{idx+1:02d}</strong><br>
                    <span style="color:#2C3E50;"><strong>Source:</strong> {selected_event.get('source', '—')}</span><br>
                    <span style="color:#2C3E50;"><strong>Date:</strong> {selected_event.get('date', '—')}</span><br><br>
                    <span style="color:#16263B; line-height:1.45;">{event_raw_snip}…</span>
                </div>
                """
            )

        else:
            custom_headline = st.text_input(
                "Dispatch Headline",
                placeholder="e.g., Tugboat Strike Paralyzes Port of Antwerp",
            )
            custom_text = st.text_area(
                "Dispatch Body / Raw Cable",
                height=180,
                placeholder="Paste the full incident report or news dispatch text…",
            )
            if custom_headline and custom_text:
                selected_event = {
                    "headline":  custom_headline,
                    "date":      "CURRENT",
                    "source":    "Ad-Hoc Field Cable",
                    "raw_text":  custom_text,
                }
            else:
                st.caption("Provide both headline and cable text to enable analysis.")

        render_html("<div style='margin-top:16px;'></div>")
        run_btn = st.button(
            "▶ EXECUTE PIPELINE",
            use_container_width=True,
            disabled=(selected_event is None),
        )

    # ── Main Panel ────────────────────────────────────────────────────
    render_html(
        """
        <div style="margin-bottom: 8px;">
            <h1 style="margin: 0; color: #16263B; font-family: 'Fraunces', serif;">MARITIME DISRUPTION ENGINE</h1>
            <div style="color: #3D4F61; font-size: 0.95rem; font-weight: 500; margin-top: 4px;">
                Autonomous Sequential Evaluation: Monitor → Impact → Routing → Cost (CrewAI + Groq)
            </div>
        </div>
        """
    )

    if selected_event:
        render_html(
            f"""
            <div class="chart-panel" style="margin-top:16px; margin-bottom:16px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; color:var(--accent-brass-dark); font-weight:600;">Current Incident Under Evaluation</div>
                <h3 style="margin-top:4px; margin-bottom:6px; font-family:'Fraunces',serif; color:#16263B;">{selected_event.get('headline', '')}</h3>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:#3D4F61; font-weight:500;">
                    ORIGIN SOURCE: {selected_event.get('source', '—')} &nbsp;|&nbsp; DISPATCH DATE: {selected_event.get('date', '—')}
                </span>
            </div>
            """
        )

    # ── Session State Management ──────────────────────────────────────────────
    if "result"     not in st.session_state: st.session_state.result     = None
    if "last_event" not in st.session_state: st.session_state.last_event = None

    if run_btn and selected_event:
        if st.session_state.last_event != selected_event.get("headline"):
            st.session_state.result = None

        with st.spinner("⏳ Transmitting cables through 4-Agent Pipeline… Rate-limit pauses applied…"):
            st.session_state.result     = run_pipeline(selected_event, routes_data)
            st.session_state.last_event = selected_event.get("headline")

    result = st.session_state.result

    # Render horizontal stage-progress rail
    render_progress_rail(result)

    if result is None:
        st.info("⚓ Select an incident dispatch from the manifest on the left and click **EXECUTE PIPELINE** to commence analysis.")
        return

    # ── Sequential Stage Panels ───────────────────────────────────────────────
    with st.expander("🔍 STAGE 1 — DISRUPTION MONITOR", expanded=True):
        show_monitor_stage(result.get("monitor", {}))

    with st.expander("📦 STAGE 2 — TRADE CORRIDOR IMPACT", expanded=True):
        show_impact_stage(result.get("impact", {}))

    with st.expander("🗺️ STAGE 3 — ALTERNATIVE ROUTING STRATEGY", expanded=True):
        show_routing_stage(result.get("routing", {}))

    with st.expander("💸 STAGE 4 — FINANCIAL DISRUPTION & DELAY APPRAISAL", expanded=True):
        show_cost_stage(result.get("cost", {}))

    # Bordered Ticket Summary
    show_manifest_summary_ticket(result)

    # Collapsible Debug JSON
    with st.expander("⚙️ RAW INTELLIGENCE PAYLOAD (JSON)", expanded=False):
        st.json(result)


if __name__ == "__main__":
    main()
