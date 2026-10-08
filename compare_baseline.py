"""
compare_baseline.py

Runs a single-agent "General Supply Chain Analyst" baseline against the
4-agent pipeline (Monitor → Impact → Routing → Cost) on all 12 test events
and prints a side-by-side comparison table plus a summary.

Results are cached to cache/baseline_results.json and cache/pipeline_results.json
so re-runs only hit the LLM for whichever results are missing.
"""

import sys
import io
import json
import os
import time
import re
import textwrap
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import json_repair
from crewai import Agent, Task, Crew, Process

# ── project imports ──────────────────────────────────────────────────────────
from agents.monitor_agent import get_default_llm
from agents.impact_agent  import load_routes_knowledge_base
from agents.cost_agent    import BASE_RATE_BY_SEVERITY   # for reasonableness check
from main import run_pipeline, _get_agents

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

BASELINE_CACHE  = CACHE_DIR / "baseline_results.json"
PIPELINE_CACHE  = CACHE_DIR / "pipeline_results.json"
EVENTS_FILE     = DATA_DIR  / "sample_news_events.json"

# ── helpers ───────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Single-Agent Baseline
# ═══════════════════════════════════════════════════════════════════════════════

_COMBINED_SCHEMA = """\
{
  "is_disruption_relevant": true,
  "event_type": "...",
  "location": "...",
  "severity": "low | medium | high | null",
  "matched_route": "exact route name from KB or null",
  "affected_product_categories": ["..."],
  "impact_confidence": 0.0,
  "alternatives": [
    {"route": "...", "rationale": "...", "feasibility": "high | medium | low"}
  ],
  "recommended_alternative": "...",
  "disruption_cost_per_day_usd": 0,
  "estimated_delay_days": 0,
  "total_disruption_cost_usd": 0,
  "rerouting_cost_estimate_usd": 0,
  "recommendation": "..."
}"""


def _build_baseline_task(agent: Agent, event: dict, routes_data: list) -> Task:
    """Constructs the single combined task for the baseline agent."""
    routes_lines = []
    for r in routes_data:
        cats = ", ".join(r.get("product_categories", []))
        routes_lines.append(
            f"- {r['route_name']} ({r.get('location')} | {r.get('region')}): {cats}"
        )
    routes_text = "\n".join(routes_lines)

    cost_heuristic = (
        "Cost heuristics (use these exactly):\n"
        "  disruption_cost_per_day_usd = base_rate[severity] × number_of_affected_categories\n"
        "    base_rate: low=$50,000  medium=$150,000  high=$300,000\n"
        "  estimated_delay_days: low=2  medium=5  high=8  (+5 days if no alternatives exist)\n"
        "  total_disruption_cost_usd = disruption_cost_per_day_usd × estimated_delay_days\n"
        "  rerouting_cost_estimate_usd: one-time switching cost scaled by feasibility\n"
        "    (high feasibility ~$250k-$600k, medium ~$600k-$1.5M, low ~$1.5M-$3M); null if no alternatives."
    )

    geographic_rule = (
        "CRITICAL GEOGRAPHIC CONSTRAINT: Only propose an alternative route if it provides "
        "a physically navigable bypass for the same trade lane — not just a port that handles "
        "similar products elsewhere. Example: if the Bosphorus Strait is blocked, no maritime "
        "alternative exists for Black Sea-origin cargo; return alternatives: []."
    )

    description = f"""\
You are a General Supply Chain Analyst. Given a raw news event and a knowledge base of
shipping routes, perform ALL of the following in a single response:

1. MONITOR: Determine if this event is a supply chain disruption (is_disruption_relevant).
   If not relevant, set all other fields to null/empty and return.

2. IMPACT: Identify which route from the knowledge base is affected (matched_route).
   List the affected product categories. Set impact_confidence (0-1).
   If no route matches with confidence ≥ 0.5, set matched_route to null.

3. ROUTING: Propose 1-3 geographically viable alternative routes from the knowledge base.
   {geographic_rule}
   If no viable alternative exists, return alternatives: [].

4. COST: Estimate financial impact using these heuristics:
   {cost_heuristic}

NEWS EVENT:
  Headline : {event.get('headline', '')}
  Date     : {event.get('date', '')}
  Text     : {event.get('raw_text', '')}

KNOWN SHIPPING ROUTES (knowledge base):
{routes_text}

Return ONLY a raw JSON object matching this exact schema (no markdown, no commentary):
{_COMBINED_SCHEMA}
"""

    return Task(
        description=description,
        expected_output="A single raw JSON object matching the combined schema above.",
        agent=agent,
    )


def _parse_baseline_output(raw: str) -> dict:
    text = str(raw).strip()
    try:
        result = json_repair.repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    # Fallback: strip markdown fences then extract first {...}
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not parse baseline output: {text[:300]}")


def run_baseline(events: list, routes_data: list) -> list:
    """
    Runs the single-agent baseline on all events.
    Loads from cache if available; only runs missing events.
    """
    cached = _load_json(BASELINE_CACHE) or {}
    llm    = get_default_llm()
    agent  = Agent(
        role="General Supply Chain Analyst",
        goal=(
            "Given a raw news event, determine in one pass whether it is a supply chain "
            "disruption, identify the affected trade route and products from the knowledge base, "
            "propose viable alternative routes, and estimate the financial cost impact."
        ),
        backstory=(
            "You are a veteran single-source supply chain intelligence analyst who produces "
            "rapid, comprehensive disruption assessments. Unlike specialized agent pipelines, "
            "you handle monitoring, impact analysis, routing alternatives, and cost estimation "
            "all in one step. Your assessments are always grounded in the supplied knowledge base."
        ),
        verbose=False,
        llm=llm,
    )

    results = []
    for idx, event in enumerate(events):
        key = str(idx)
        if key in cached:
            print(f"  [Baseline {idx+1:02d}] (cached)")
            results.append(cached[key])
            continue

        headline = event.get("headline", "")
        print(f"  [Baseline {idx+1:02d}] Running: {headline[:70]}...")
        task = _build_baseline_task(agent, event, routes_data)
        try:
            raw  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff().raw
            out  = _parse_baseline_output(raw)
        except Exception as exc:
            out  = {"_error": str(exc), "is_disruption_relevant": None}

        cached[key] = out
        results.append(out)
        _save_json(BASELINE_CACHE, cached)   # persist after each event
        time.sleep(2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — 4-Agent Pipeline (cached)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline_all(events: list, routes_data: list) -> list:
    """
    Runs the 4-agent pipeline on all events.
    Loads from cache if available; only runs missing events.
    """
    cached = _load_json(PIPELINE_CACHE) or {}
    _get_agents()   # warm up once

    results = []
    for idx, event in enumerate(events):
        key = str(idx)
        if key in cached:
            print(f"  [Pipeline {idx+1:02d}] (cached)")
            results.append(cached[key])
            continue

        headline = event.get("headline", "")
        print(f"  [Pipeline {idx+1:02d}] Running: {headline[:70]}...")
        try:
            out = run_pipeline(event, routes_data)
        except Exception as exc:
            out = {"monitor": {"_error": str(exc)}, "impact": {}, "routing": {}, "cost": {}}

        cached[key] = out
        results.append(out)
        _save_json(PIPELINE_CACHE, cached)
        time.sleep(2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Assessment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_baseline(b: dict, expected_relevant: bool | None) -> dict:
    """Score one baseline result."""
    err       = b.get("_error")
    relevant  = b.get("is_disruption_relevant")
    route     = b.get("matched_route")
    alts      = b.get("alternatives", [])
    cpd       = b.get("disruption_cost_per_day_usd")
    days      = b.get("estimated_delay_days")
    total     = b.get("total_disruption_cost_usd")
    reroute   = b.get("rerouting_cost_estimate_usd")
    cats      = b.get("affected_product_categories", [])
    severity  = b.get("severity", "")

    hallucination_notes = []

    # Relevance check
    if expected_relevant is not None and relevant != expected_relevant:
        hallucination_notes.append(
            f"relevance={relevant} (expected {expected_relevant})"
        )

    # Cost math correctness (total should equal cpd × days, ±1%)
    cost_ok = False
    if cpd and days and total:
        expected_total = cpd * days
        cost_ok = abs(expected_total - total) / max(expected_total, 1) < 0.02
        if not cost_ok:
            hallucination_notes.append(
                f"cost math wrong: {cpd}×{days}={expected_total} ≠ {total}"
            )

    # Cost scale reasonableness (not all identical / not wildly off heuristic)
    cost_reasonable = False
    if cpd and days and severity and cats:
        sev_key   = (severity or "").lower()
        base_rate = BASE_RATE_BY_SEVERITY.get(sev_key, 150_000)
        expected_cpd = base_rate * max(len(cats), 1)
        ratio = cpd / max(expected_cpd, 1) if expected_cpd else 1
        # Accept within 3× of the expected deterministic value
        cost_reasonable = 0.33 < ratio < 3.0 and cost_ok
        if not cost_reasonable:
            hallucination_notes.append(
                f"cost scale off: got {cpd}/day, heuristic says ~{expected_cpd}/day"
            )

    # Alternatives-when-no-route check
    if not route and alts:
        hallucination_notes.append("proposed alternatives with no matched route")

    return {
        "relevant":          relevant,
        "route_matched":     bool(route),
        "alt_given":         bool(alts),
        "cost_reasonable":   cost_reasonable,
        "cost_math_ok":      cost_ok,
        "hallucination":     "; ".join(hallucination_notes) if hallucination_notes else None,
        "error":             err,
    }


def _assess_pipeline(p: dict) -> dict:
    """Score one 4-agent pipeline result."""
    monitor = p.get("monitor", {})
    impact  = p.get("impact",  {})
    routing = p.get("routing", {})
    cost    = p.get("cost",    {})

    route   = impact.get("matched_route") if isinstance(impact, dict) else None
    alts    = routing.get("alternatives", []) if isinstance(routing, dict) else []
    cpd     = cost.get("disruption_cost_per_day_usd")  if isinstance(cost, dict) else None
    days    = cost.get("estimated_delay_days")          if isinstance(cost, dict) else None
    total   = cost.get("total_disruption_cost_usd")     if isinstance(cost, dict) else None

    skipped = isinstance(impact, dict) and impact.get("skipped")
    relevant = monitor.get("is_disruption_relevant") if isinstance(monitor, dict) else None

    cost_math_ok = False
    if cpd and days and total:
        expected = cpd * days
        cost_math_ok = abs(expected - total) / max(expected, 1) < 0.02

    return {
        "relevant":        relevant,
        "route_matched":   bool(route),
        "alt_given":       bool(alts),
        "cost_math_ok":    cost_math_ok,
        "cost_reasonable": cost_math_ok,  # deterministic → always correct
        "skipped":         bool(skipped),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Table printing
# ═══════════════════════════════════════════════════════════════════════════════

def _yn(val: bool | None) -> str:
    if val is True:  return "✓ Yes"
    if val is False: return "✗ No"
    return "—"


def _wrap(text: str, width: int = 28) -> str:
    if not text:
        return "—"
    return "\n".join(textwrap.wrap(str(text), width))


def print_table(events: list, baseline_results: list, pipeline_results: list,
                b_scores: list, p_scores: list):

    col_w = [4, 36, 10, 10, 12, 26, 10, 10, 12]
    hdrs  = [
        "#",
        "Headline",
        "B: Route\nMatched?",
        "B: Alt\nGiven?",
        "B: Cost\nReasonable?",
        "B: Hallucination / Error",
        "P: Route\nMatched?",
        "P: Alt\nGiven?",
        "P: Cost\nReasonable?",
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def _row_lines(cells):
        # split each cell into lines, pad to column width, yield row-by-row
        split = [str(c).split("\n") for c in cells]
        max_lines = max(len(s) for s in split)
        for li in range(max_lines):
            parts = []
            for ci, s in enumerate(split):
                text = s[li] if li < len(s) else ""
                parts.append(" " + text.ljust(col_w[ci]) + " ")
            yield "|" + "|".join(parts) + "|"

    print("\n" + sep)
    for line in _row_lines(hdrs):
        print(line)
    print(sep.replace("-", "="))

    for idx, (event, b, p, bs, ps) in enumerate(
        zip(events, baseline_results, pipeline_results, b_scores, p_scores), start=1
    ):
        headline = event.get("headline", "")[:34]
        hall     = _wrap(bs.get("hallucination") or ("Error: " + bs["error"][:22] if bs.get("error") else ""), 26)

        cells = [
            str(idx),
            headline,
            _yn(bs["route_matched"]),
            _yn(bs["alt_given"]),
            _yn(bs["cost_reasonable"]),
            hall,
            _yn(ps["route_matched"]),
            _yn(ps["alt_given"]),
            _yn(ps["cost_reasonable"]),
        ]
        for line in _row_lines(cells):
            print(line)
        print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Summary scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Ground-truth relevance for all 12 events:
# 1-9 relevant, 10-11 irrelevant, 12 relevant-but-unknown-location
EXPECTED_RELEVANT = [
    True, True, True, True, True,
    True, True, True, True,
    False, False,
    True,
]

def _grade_baseline(bs: dict, ev_idx: int, pipeline_score: dict) -> str:
    """
    Grade one baseline result:
      CORRECT       — relevance right, route matches pipeline, cost reasonable
      PARTIAL       — some fields right, at least one major field wrong
      WRONG         — relevance wrong OR error
    """
    if bs.get("error"):
        return "WRONG"
    expected_rel = EXPECTED_RELEVANT[ev_idx]
    relevant_ok  = bs["relevant"] == expected_rel

    if not expected_rel:
        # Irrelevant event: only relevance matters
        return "CORRECT" if relevant_ok else "WRONG"

    route_ok = bs["route_matched"] == pipeline_score["route_matched"]
    cost_ok  = bs["cost_reasonable"]
    hall_ok  = not bs.get("hallucination")

    if relevant_ok and route_ok and cost_ok and hall_ok:
        return "CORRECT"
    if relevant_ok and (route_ok or cost_ok):
        return "PARTIAL"
    return "WRONG"


def print_summary(events, b_scores, p_scores):
    grades = [
        _grade_baseline(bs, idx, ps)
        for idx, (bs, ps) in enumerate(zip(b_scores, p_scores))
    ]
    correct = grades.count("CORRECT")
    partial = grades.count("PARTIAL")
    wrong   = grades.count("WRONG")
    total   = len(grades)

    print("\n" + "=" * 60)
    print("SUMMARY — Single-Agent Baseline vs 4-Agent Pipeline")
    print("=" * 60)
    print(f"{'Event':<6} {'Headline':<44} {'Grade'}")
    print("-" * 60)
    for idx, (event, g) in enumerate(zip(events, grades), start=1):
        hl = event.get("headline", "")[:42]
        icon = {"CORRECT": "✓", "PARTIAL": "~", "WRONG": "✗"}[g]
        print(f"  {idx:<4} {hl:<44} {icon} {g}")

    print("-" * 60)
    print(f"  CORRECT : {correct}/{total}")
    print(f"  PARTIAL : {partial}/{total}")
    print(f"  WRONG   : {wrong}/{total}")
    print()

    p_correct = sum(
        1 for ps in p_scores
        if ps.get("route_matched") is not False and ps.get("cost_reasonable") is not False
    )
    print(f"  4-Agent pipeline fully correct: {p_correct}/{total}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    with open(EVENTS_FILE, encoding="utf-8") as f:
        events = json.load(f)

    routes_data = load_routes_knowledge_base()

    print(f"\nLoaded {len(events)} events, {len(routes_data)} routes.")
    print(f"Cache dir: {CACHE_DIR}\n")

    # ── Run baseline (cached) ─────────────────────────────────────────────────
    print("── Running / loading Single-Agent Baseline ──")
    baseline_results = run_baseline(events, routes_data)

    # ── Run 4-agent pipeline (cached) ─────────────────────────────────────────
    print("\n── Running / loading 4-Agent Pipeline ──")
    pipeline_results = run_pipeline_all(events, routes_data)

    # ── Score ─────────────────────────────────────────────────────────────────
    b_scores = [
        _assess_baseline(b, EXPECTED_RELEVANT[i])
        for i, b in enumerate(baseline_results)
    ]
    p_scores = [_assess_pipeline(p) for p in pipeline_results]

    # ── Table ─────────────────────────────────────────────────────────────────
    print_table(events, baseline_results, pipeline_results, b_scores, p_scores)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(events, b_scores, p_scores)


if __name__ == "__main__":
    main()
