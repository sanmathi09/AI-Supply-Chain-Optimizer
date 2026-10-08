"""
Run only the Bosphorus event (event index 8, the 9th event) twice through the full pipeline
to verify the updated Routing Agent prompt consistently returns no alternatives.
"""
import sys
import io
import json
import time
import os
from dotenv import load_dotenv
from crewai import Crew

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
load_dotenv()

from agents.monitor_agent  import create_monitor_agent,  create_monitor_task,  parse_disruption_output
from agents.impact_agent   import create_impact_agent,   create_impact_task,   parse_impact_output,   load_routes_knowledge_base
from agents.routing_agent  import create_routing_agent,  create_routing_task,  parse_routing_output
from agents.cost_agent     import create_cost_agent,     create_cost_task,     parse_cost_output, compute_deterministic_costs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(DATA_DIR, "sample_news_events.json"), encoding="utf-8") as f:
    all_events = json.load(f)

routes_data = load_routes_knowledge_base()

# Event 9 is index 8 (0-based)
bosphorus_event = all_events[8]
headline = bosphorus_event.get("headline", "")
print(f"\nTarget event: {headline}\n")

monitor_agent  = create_monitor_agent()
impact_agent   = create_impact_agent()
routing_agent  = create_routing_agent()
cost_agent     = create_cost_agent()

for run_num in range(1, 3):
    print(f"\n{'='*70}")
    print(f"RUN {run_num}/2")
    print(f"{'='*70}")

    # --- Monitor ---
    monitor_task = create_monitor_task(monitor_agent, bosphorus_event)
    monitor_res  = Crew(agents=[monitor_agent], tasks=[monitor_task], verbose=False).kickoff()
    monitor_out  = parse_disruption_output(monitor_res.raw)
    print("\n[Monitor Output]:")
    print(json.dumps(monitor_out, indent=2))

    if not monitor_out.get("is_disruption_relevant"):
        print("Skipped downstream (not relevant)")
        continue

    # --- Impact ---
    impact_task = create_impact_task(impact_agent, monitor_out, routes_data)
    impact_res  = Crew(agents=[impact_agent], tasks=[impact_task], verbose=False).kickoff()
    impact_out  = parse_impact_output(impact_res.raw)
    print("\n[Impact Output]:")
    print(json.dumps(impact_out, indent=2))

    if not impact_out.get("matched_route"):
        print("Skipped Routing/Cost (no matched route)")
        continue

    # --- Routing ---
    routing_task = create_routing_task(routing_agent, impact_out, routes_data)
    routing_res  = Crew(agents=[routing_agent], tasks=[routing_task], verbose=False).kickoff()
    routing_out  = parse_routing_output(routing_res.raw)
    print("\n[Routing Output]:")
    print(json.dumps(routing_out, indent=2))

    # --- Cost ---
    severity      = monitor_out.get("severity", "medium")
    affected_cats = impact_out.get("affected_product_categories", [])
    has_alts      = len(routing_out.get("alternatives", [])) > 0

    cost_per_day, delay_days, total_cost = compute_deterministic_costs(severity, affected_cats, has_alts)

    cost_task = create_cost_task(
        cost_agent, severity, affected_cats, routing_out,
        cost_per_day, delay_days, total_cost
    )
    cost_res  = Crew(agents=[cost_agent], tasks=[cost_task], verbose=False).kickoff()
    cost_out  = parse_cost_output(cost_res.raw, cost_per_day, delay_days, total_cost, has_alts)
    print("\n[Cost Output]:")
    print(json.dumps(cost_out, indent=2))

    if run_num < 2:
        print("\nWaiting 4s before run 2...")
        time.sleep(4)

print("\n\nDone.")
