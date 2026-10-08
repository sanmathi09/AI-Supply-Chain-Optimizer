import os
import json
import re
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import json_repair
from crewai import Agent, Task, LLM
from agents.monitor_agent import get_default_llm


class ImpactAnalysisOutput(BaseModel):
    matched_route: Optional[str] = Field(
        default=None,
        description="Name of the matched trade route, port, or chokepoint from the knowledge base. Null if no confident match."
    )
    affected_product_categories: List[str] = Field(
        default_factory=list,
        description="List of product categories typically transported through this route that are affected."
    )
    impact_confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 indicating how strongly the disruption matches the route."
    )
    reasoning: str = Field(
        description="Explanation of the match rationale, affected product categories, or why no match was found."
    )


def load_routes_knowledge_base(routes_file: Optional[Path] = None) -> List[dict]:
    """
    Loads data/supply_chain_routes.json from disk.
    """
    if routes_file is None:
        routes_file = Path(__file__).resolve().parent.parent / "data" / "supply_chain_routes.json"

    if not routes_file.exists():
        raise FileNotFoundError(f"Supply chain routes file not found at: {routes_file}")

    with open(routes_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_impact_agent(llm: Optional[LLM] = None) -> Agent:
    """
    Creates and returns the Supply Chain Impact Analyst agent.
    """
    if llm is None:
        llm = get_default_llm()

    return Agent(
        role="Supply Chain Impact Analyst",
        goal="Determine which trade routes and product categories are affected by a given disruption event, using the supply chain knowledge base",
        backstory=(
            "You are a meticulous supply chain risk analyst and trade route specialist. You evaluate incoming "
            "maritime disruption alerts, cross-referencing affected locations and event characteristics against "
            "established global shipping corridors, chokepoints, and terminal hubs. By leveraging the supply chain "
            "routes knowledge base, you identify the exact affected route, map out compromised product categories, "
            "and articulate clear, evidence-based reasoning."
        ),
        verbose=False,
        llm=llm
    )


def create_impact_task(
    agent: Agent,
    monitor_output: dict,
    routes_data: Optional[List[dict]] = None
) -> Task:
    """
    Creates a CrewAI task to match a monitor disruption event against the supply chain routes knowledge base.
    """
    if routes_data is None:
        routes_data = load_routes_knowledge_base()

    event_type = monitor_output.get("event_type", "Unknown")
    location = monitor_output.get("location", "Unknown")
    severity = monitor_output.get("severity", "Unknown")

    # Format the routes knowledge base into a concise, token-efficient reference list
    routes_lines = []
    for r in routes_data:
        categories = ", ".join(r.get("product_categories", []))
        routes_lines.append(
            f"- {r.get('route_name')} ({r.get('location')} | {r.get('region')}): {categories}"
        )
    routes_text = "\n".join(routes_lines)

    description = f"""
Disruption Event to Analyze:
- Event Type: {event_type}
- Location: {location}
- Severity: {severity}

Known Shipping Routes, Ports & Product Categories:
{routes_text}

Instructions:
1. Cross-reference the disruption location and event type against the known routes list above.
2. If a route matches with confidence >= 0.5:
   - 'matched_route': Set to the exact route name from the list.
   - 'affected_product_categories': Set to the list of affected product categories from that route.
   - 'impact_confidence': Float between 0.5 and 1.0 reflecting how clear the match is.
   - 'reasoning': Brief explanation of the geographic/operational connection.
3. If no route confidently matches (confidence < 0.5):
   - 'matched_route': null
   - 'affected_product_categories': []
   - 'impact_confidence': Float < 0.5
   - 'reasoning': Brief explanation of why no route matches.
4. Output ONLY a valid, raw JSON object matching the requested schema. No markdown formatting.
"""

    expected_output = """
A strict raw JSON object with keys:
{
  "matched_route": "...",
  "affected_product_categories": ["..."],
  "impact_confidence": 0.95,
  "reasoning": "..."
}
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )


def parse_impact_output(raw_output: str) -> dict:
    """
    Cleans and parses raw agent output into a validated dictionary conforming to ImpactAnalysisOutput.
    """
    text = str(raw_output).strip()

    try:
        data = json_repair.repair_json(text, return_objects=True)
        if isinstance(data, dict):
            return ImpactAnalysisOutput.model_validate(data).model_dump()
    except Exception:
        pass

    # Fallback cleanup
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    data = json.loads(text)
    return ImpactAnalysisOutput.model_validate(data).model_dump()
