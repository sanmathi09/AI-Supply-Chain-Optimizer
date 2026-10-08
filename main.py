import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from agents.monitor_agent import (
    create_monitor_agent,
    create_monitor_task,
    parse_disruption_output,
)
from agents.impact_agent import (
    create_impact_agent,
    create_impact_task,
    parse_impact_output,
    load_routes_knowledge_base,
)
from agents.routing_agent import (
    create_routing_agent,
    create_routing_task,
    parse_routing_output,
)
from agents.cost_agent import (
    create_cost_agent,
    create_cost_task,
    parse_cost_output,
    compute_deterministic_costs,
)

# ---------------------------------------------------------------------------
# Shared agent instances — created once, reused across all events
# ---------------------------------------------------------------------------
_monitor_agent = None
_impact_agent  = None
_routing_agent = None
_cost_agent    = None


def _get_agents():
    global _monitor_agent, _impact_agent, _routing_agent, _cost_agent
    if _monitor_agent is None:
        _monitor_agent  = create_monitor_agent()
        _impact_agent   = create_impact_agent()
        _routing_agent  = create_routing_agent()
        _cost_agent     = create_cost_agent()
    return _monitor_agent, _impact_agent, _routing_agent, _cost_agent


def _kickoff(agent, task) -> str:
    """Run a single-agent Crew and return raw string output."""
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff().raw


# ---------------------------------------------------------------------------
# Per-event pipeline
# ---------------------------------------------------------------------------
def run_pipeline(event: dict, routes_data: list) -> dict:
    """
    Runs the Monitor → Impact → Routing → Cost pipeline for one news event.

    Conditional skipping:
      - Impact, Routing, Cost are skipped if Monitor marks event as not relevant.
      - Routing and Cost are skipped if Impact finds no matched route.
      - Cost heuristic values (cost/day, delay, total) are computed deterministically
        in Python after Routing, then injected into the Cost task — not left to the LLM.

    Returns a result dict with keys: monitor, impact, routing, cost.
    Each value is either the parsed output dict or {"skipped": "<reason>"}.
    """
    monitor_agent, impact_agent, routing_agent, cost_agent = _get_agents()
    result = {}

    # ------------------------------------------------------------------ #
    # Stage 1: Monitor                                                     #
    # ------------------------------------------------------------------ #
    monitor_task = create_monitor_task(monitor_agent, event)
    try:
        monitor_raw    = _kickoff(monitor_agent, monitor_task)
        monitor_output = parse_disruption_output(monitor_raw)
    except Exception as exc:
        monitor_output = {"is_disruption_relevant": False, "_error": str(exc)}
    result["monitor"] = monitor_output

    if not monitor_output.get("is_disruption_relevant"):
        skip = {"skipped": "Event is not a supply chain disruption"}
        result["impact"] = result["routing"] = result["cost"] = skip
        return result

    # ------------------------------------------------------------------ #
    # Stage 2: Impact                                                      #
    # ------------------------------------------------------------------ #
    impact_task = create_impact_task(impact_agent, monitor_output, routes_data)
    try:
        impact_raw    = _kickoff(impact_agent, impact_task)
        impact_output = parse_impact_output(impact_raw)
    except Exception as exc:
        impact_output = {"matched_route": None, "_error": str(exc)}
    result["impact"] = impact_output

    if not impact_output.get("matched_route"):
        skip = {"skipped": "No route matched by Impact Agent"}
        result["routing"] = result["cost"] = skip
        return result

    # ------------------------------------------------------------------ #
    # Stage 3: Routing                                                     #
    # ------------------------------------------------------------------ #
    routing_task = create_routing_task(routing_agent, impact_output, routes_data)
    try:
        routing_raw    = _kickoff(routing_agent, routing_task)
        routing_output = parse_routing_output(routing_raw)
    except Exception as exc:
        routing_output = {"alternatives": [], "recommended_alternative": f"Error: {exc}"}
    result["routing"] = routing_output

    # ------------------------------------------------------------------ #
    # Stage 4: Cost  (deterministic Python values injected pre-LLM)       #
    # ------------------------------------------------------------------ #
    severity      = monitor_output.get("severity", "medium")
    affected_cats = impact_output.get("affected_product_categories", [])
    has_alts      = bool(routing_output.get("alternatives"))

    cost_per_day, delay_days, total_cost = compute_deterministic_costs(
        severity=severity,
        affected_categories=affected_cats,
        has_alternatives=has_alts,
    )

    cost_task = create_cost_task(
        agent=cost_agent,
        severity=severity,
        affected_categories=affected_cats,
        routing_output=routing_output,
        disruption_cost_per_day_usd=cost_per_day,
        estimated_delay_days=delay_days,
        total_disruption_cost_usd=total_cost,
    )
    try:
        cost_raw    = _kickoff(cost_agent, cost_task)
        cost_output = parse_cost_output(
            cost_raw,
            disruption_cost_per_day_usd=cost_per_day,
            estimated_delay_days=delay_days,
            total_disruption_cost_usd=total_cost,
            has_alternatives=has_alts,
        )
    except Exception as exc:
        cost_output = {"_error": str(exc)}
    result["cost"] = cost_output

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _print_stage(label: str, data: dict):
    if data.get("skipped"):
        print(f"\n  [{label}]: Skipped ({data['skipped']})")
    elif data.get("_error"):
        print(f"\n  [{label} Error]: {data['_error']}")
    else:
        formatted = json.dumps(data, indent=4).replace("\n", "\n  ")
        print(f"\n  [{label}]:\n  {formatted}")


def test_multi_agent_pipeline():
    news_file = Path(__file__).parent / "data" / "sample_news_events.json"
    if not news_file.exists():
        print(f"Error: Could not find {news_file}")
        return

    with open(news_file, "r", encoding="utf-8") as f:
        news_events = json.load(f)

    routes_data = load_routes_knowledge_base()

    print(f"Loaded {len(news_events)} news events from {news_file.name}")
    print(f"Loaded {len(routes_data)} routes from supply_chain_routes.json")
    print("Initializing 4-Agent Pipeline: Monitor → Impact → Routing → Cost...\n")

    # Warm up agents once
    _get_agents()

    for idx, event in enumerate(news_events, start=1):
        headline = event.get("headline", "")
        print(f"\n[{idx}/{len(news_events)}] Headline: {headline}")

        result = run_pipeline(event, routes_data)

        _print_stage("Monitor Agent Output",  result["monitor"])
        _print_stage("Impact Agent Output",   result["impact"])
        _print_stage("Routing Agent Output",  result["routing"])
        _print_stage("Cost Agent Output",     result["cost"])

        print("\n" + "=" * 75)
        time.sleep(2)  # Respect API rate limits between events


if __name__ == "__main__":
    test_multi_agent_pipeline()
