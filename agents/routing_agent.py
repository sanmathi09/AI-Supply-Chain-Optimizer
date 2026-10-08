import os
import json
import re
from pathlib import Path
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import json_repair
from crewai import Agent, Task, LLM
from agents.monitor_agent import get_default_llm
from agents.impact_agent import load_routes_knowledge_base


class RouteAlternative(BaseModel):
    route: str = Field(
        description="Name of the alternative trade route or port from the knowledge base"
    )
    rationale: str = Field(
        description="Why this route serves as a viable alternative for the affected product categories"
    )
    feasibility: Literal["high", "medium", "low"] = Field(
        description="Feasibility level of rerouting: 'high', 'medium', or 'low'"
    )


class RoutingAnalysisOutput(BaseModel):
    alternatives: List[RouteAlternative] = Field(
        default_factory=list,
        description="List of 1 to 3 plausible alternative routes, excluding the disrupted route"
    )
    recommended_alternative: str = Field(
        description="Primary recommended alternative route or explanation if no reasonable alternative exists"
    )


def create_routing_agent(llm: Optional[LLM] = None) -> Agent:
    """
    Creates and returns the Alternative Routing Strategist agent.
    """
    if llm is None:
        llm = get_default_llm()

    return Agent(
        role="Alternative Routing Strategist",
        goal="Propose viable alternative shipping routes or suppliers when a primary route is disrupted",
        backstory=(
            "You are a veteran maritime logistics planner and supply chain rerouting specialist. "
            "When critical shipping corridors, canals, or ports face sudden disruptions, you evaluate "
            "global trade geography to propose 1-3 viable, realistic bypass routes or alternative ports "
            "capable of handling the affected cargo categories. You evaluate operational feasibility, "
            "transit trade-offs, and geographic realities, while realistically acknowledging when a chokepoint "
            "has no immediate substitute."
        ),
        verbose=False,
        llm=llm
    )


def create_routing_task(
    agent: Agent,
    impact_output: dict,
    routes_data: Optional[List[dict]] = None
) -> Task:
    """
    Creates a CrewAI task to determine 1-3 alternative routes for a disrupted route.
    """
    if routes_data is None:
        routes_data = load_routes_knowledge_base()

    matched_route = impact_output.get("matched_route", "Unknown")
    affected_categories = impact_output.get("affected_product_categories", [])
    categories_str = ", ".join(affected_categories) if affected_categories else "General Cargo"
    impact_reasoning = impact_output.get("reasoning", "")

    # Format available alternative routes excluding the disrupted route
    routes_lines = []
    for r in routes_data:
        r_name = r.get("route_name")
        categories = ", ".join(r.get("product_categories", []))
        routes_lines.append(
            f"- {r_name} ({r.get('location')} | {r.get('region')}): {categories}"
        )
    routes_text = "\n".join(routes_lines)

    description = f"""
Primary Route Disrupted: {matched_route}
Affected Product Categories: {categories_str}
Context & Impact Reasoning: {impact_reasoning}

Known Global Routes & Ports in Knowledge Base:
{routes_text}

CRITICAL GEOGRAPHIC CONSTRAINT RULE:
Some chokepoints have no sea-route alternative due to fundamental geography. An alternative is only valid
if it is a genuinely viable bypass route for the SPECIFIC cargo flow — not merely a port that handles
similar product categories in a different region.

Example: Black Sea ports (Ukraine, Russia, Georgia, Bulgaria, Romania) can ONLY be accessed via the
Bosphorus and Dardanelles Straits. If the Bosphorus Strait is blocked, NO maritime alternative exists
for Black Sea-origin or Black Sea-bound cargo. Ports like Rotterdam or Antwerp handle similar product
categories (grains, steel, crude) but they are DESTINATION hubs, not bypass routes — rerouting to them
would mean sourcing from entirely different origins, which is a procurement strategy, not a routing alternative.
Do NOT list such ports as 'alternatives' for a Bosphorus disruption.

Apply this same geographic logic to all chokepoints: only suggest alternatives that provide a physically
navigable bypass for vessels that are already on the affected trade lane.

Instructions:
1. Identify 1 to 3 plausible alternative routes or ports from the knowledge base that provide a genuinely
   geographically viable bypass for the stated disrupted route and trade lane.
2. EXCLUDE the disrupted route ({matched_route}) from the alternatives list.
3. For each alternative, provide:
   - 'route': Exact route name from the knowledge base
   - 'rationale': Specific explanation of why this route is a genuinely navigable bypass for {categories_str}
     on the SAME trade lane — not just a port that handles similar goods elsewhere
   - 'feasibility': 'high', 'medium', or 'low' based on transit time, geographic proximity, or capacity
4. Set 'recommended_alternative' to the top recommended choice with a brief recommendation summary.
5. If NO geographically viable alternative exists (e.g., Bosphorus Strait blocked → no Black Sea bypass),
   return an empty 'alternatives' list [] and explain the geographic constraint clearly in 'recommended_alternative'.
6. Return ONLY a valid, raw JSON object matching the required schema. No markdown ticks or commentary.
"""

    expected_output = """
A strict raw JSON object with keys:
{
  "alternatives": [
    {
      "route": "...",
      "rationale": "...",
      "feasibility": "high"
    }
  ],
  "recommended_alternative": "..."
}
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )


def parse_routing_output(raw_output: str) -> dict:
    """
    Cleans and parses raw agent output into a validated dictionary conforming to RoutingAnalysisOutput.
    """
    text = str(raw_output).strip()

    try:
        data = json_repair.repair_json(text, return_objects=True)
        if isinstance(data, dict):
            return RoutingAnalysisOutput.model_validate(data).model_dump()
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
    return RoutingAnalysisOutput.model_validate(data).model_dump()
