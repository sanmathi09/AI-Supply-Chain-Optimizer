import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

from agents.impact_agent import load_routes_knowledge_base
from main import run_pipeline, _get_agents, _print_stage

_get_agents()
routes = load_routes_knowledge_base()

with open("data/sample_news_events.json", encoding="utf-8") as f:
    events = json.load(f)

for idx in [0, 9]:   # Suez (relevant) + Real Madrid (irrelevant)
    ev = events[idx]
    headline = ev["headline"]
    print(f"\n--- Event {idx+1}: {headline} ---")
    result = run_pipeline(ev, routes)
    _print_stage("Monitor", result["monitor"])
    _print_stage("Impact",  result["impact"])
    _print_stage("Routing", result["routing"])
    _print_stage("Cost",    result["cost"])
    print()
