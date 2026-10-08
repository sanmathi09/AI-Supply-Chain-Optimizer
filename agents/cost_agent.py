import os
import json
import re
from typing import Optional, List
from pydantic import BaseModel, Field
import json_repair
from crewai import Agent, Task, LLM
from agents.monitor_agent import get_default_llm

# ---------------------------------------------------------------------------
# Deterministic heuristic tables — these are illustrative strategic models,
# not real audited financial data.
# ---------------------------------------------------------------------------
BASE_RATE_BY_SEVERITY = {
    "low":    50_000,   # $/category/day
    "medium": 150_000,  # $/category/day
    "high":   300_000,  # $/category/day
}
BASE_DELAY_BY_SEVERITY = {
    "low":    2,   # days
    "medium": 5,   # days
    "high":   8,   # days
}
NO_ALTERNATIVE_EXTRA_DAYS = 5  # added when alternatives list is empty


def compute_deterministic_costs(
    severity: str,
    affected_categories: List[str],
    has_alternatives: bool,
) -> tuple[int, int, int]:
    """
    Returns (disruption_cost_per_day_usd, estimated_delay_days, total_disruption_cost_usd)
    computed deterministically from the heuristic tables.
    """
    sev_key = severity.lower() if severity and severity.lower() in BASE_RATE_BY_SEVERITY else "medium"
    num_cats = max(len(affected_categories), 1)

    cost_per_day = BASE_RATE_BY_SEVERITY[sev_key] * num_cats
    delay_days = BASE_DELAY_BY_SEVERITY[sev_key]
    if not has_alternatives:
        delay_days += NO_ALTERNATIVE_EXTRA_DAYS
    total = cost_per_day * delay_days

    return cost_per_day, delay_days, total


class CostAnalysisOutput(BaseModel):
    disruption_cost_per_day_usd: int = Field(
        description="Daily disruption cost (deterministic: base_rate_by_severity × num_categories)"
    )
    estimated_delay_days: int = Field(
        description="Estimated delay days (deterministic: base_delay_by_severity [+5 if no alternatives])"
    )
    total_disruption_cost_usd: int = Field(
        description="Total disruption cost = disruption_cost_per_day_usd × estimated_delay_days"
    )
    rerouting_cost_estimate_usd: Optional[int] = Field(
        default=None,
        description="One-time rerouting/bunker cost estimate. Null if no alternative exists."
    )
    recommendation: str = Field(
        description="Plain-English recommendation comparing disruption cost against rerouting cost"
    )


def create_cost_agent(llm: Optional[LLM] = None) -> Agent:
    """
    Creates and returns the Disruption Cost Analyst agent.
    """
    if llm is None:
        llm = get_default_llm()

    return Agent(
        role="Disruption Cost Analyst",
        goal="Estimate the financial impact of a supply chain disruption versus the cost of switching to the recommended alternative route",
        backstory=(
            "You are a specialized supply chain financial analyst and freight cost estimator. "
            "You assess operational disruption costs, daily demurrage, and cargo holding penalties across affected "
            "product categories, and compare them against one-time route switching and bunker surcharges. "
            "You utilize standard freight cost heuristics to deliver clear, actionable trade-off evaluations "
            "and cost-benefit recommendations. (Note: These evaluations are illustrative heuristic models designed "
            "for strategic decision simulation rather than real-time audited financial figures)."
        ),
        verbose=False,
        llm=llm
    )


def create_cost_task(
    agent: Agent,
    severity: str,
    affected_categories: List[str],
    routing_output: dict,
    disruption_cost_per_day_usd: int,
    estimated_delay_days: int,
    total_disruption_cost_usd: int,
) -> Task:
    """
    Creates a CrewAI task for the Cost Agent.
    disruption_cost_per_day_usd, estimated_delay_days, and total_disruption_cost_usd
    are pre-computed deterministically in Python and injected into the prompt.
    The LLM only needs to estimate rerouting_cost_estimate_usd and write recommendation.
    """
    categories_str = ", ".join(affected_categories) if affected_categories else "General Freight"
    num_categories = max(len(affected_categories), 1)

    alternatives = routing_output.get("alternatives", [])
    recommended_alt = routing_output.get("recommended_alternative", "None")
    has_alternatives = len(alternatives) > 0

    if has_alternatives:
        reroute_instruction = (
            f"The pre-computed disruption cost figures are already fixed (see above). "
            f"Your ONLY task is to:\n"
            f"1. Estimate `rerouting_cost_estimate_usd` — a one-time switching/bunker cost "
            f"based on the feasibility of the top recommended alternative:\n"
            f"   * High feasibility: ~$250,000 - $600,000\n"
            f"   * Medium feasibility: ~$600,000 - $1,500,000 (e.g. Cape of Good Hope detour)\n"
            f"   * Low feasibility: ~$1,500,000 - $3,000,000\n"
            f"2. Write `recommendation` — a plain-English sentence comparing "
            f"`total_disruption_cost_usd` (${total_disruption_cost_usd:,}) against your "
            f"`rerouting_cost_estimate_usd` and stating whether rerouting is financially justified."
        )
    else:
        reroute_instruction = (
            f"NO alternative routes exist. You MUST:\n"
            f"1. Set `rerouting_cost_estimate_usd` strictly to null.\n"
            f"2. Write `recommendation` clearly stating no alternative route exists, "
            f"that the supply chain must absorb the full disruption cost "
            f"(${total_disruption_cost_usd:,}), and suggest non-maritime mitigation."
        )

    description = f"""
You are evaluating the financial impact of a supply chain disruption.
The following cost figures have already been calculated deterministically by the system — do NOT change them:

  disruption_cost_per_day_usd = {disruption_cost_per_day_usd:,}
    (= ${BASE_RATE_BY_SEVERITY.get(severity.lower() if severity else 'medium', 150_000):,}/category/day × {num_categories} categories)
  estimated_delay_days        = {estimated_delay_days}
    (= base delay for '{severity}' severity{' + 5 days (no alternative exists)' if not has_alternatives else ''})
  total_disruption_cost_usd   = {total_disruption_cost_usd:,}
    (= {disruption_cost_per_day_usd:,} × {estimated_delay_days} days)

Disruption Context:
  Severity: {severity}
  Affected Product Categories ({num_categories}): {categories_str}
  Alternative Routing Options: {json.dumps(alternatives, indent=2)}
  Recommended Alternative Strategy: {recommended_alt}

Your task:
{reroute_instruction}

Return ONLY a valid, raw JSON object with these EXACT values for the first three fields,
plus your estimates for rerouting_cost_estimate_usd and recommendation:
{{
  "disruption_cost_per_day_usd": {disruption_cost_per_day_usd},
  "estimated_delay_days": {estimated_delay_days},
  "total_disruption_cost_usd": {total_disruption_cost_usd},
  "rerouting_cost_estimate_usd": <your estimate or null>,
  "recommendation": "<your plain-English recommendation>"
}}

No markdown ticks. No surrounding commentary.
"""

    expected_output = f"""
A strict raw JSON object:
{{
  "disruption_cost_per_day_usd": {disruption_cost_per_day_usd},
  "estimated_delay_days": {estimated_delay_days},
  "total_disruption_cost_usd": {total_disruption_cost_usd},
  "rerouting_cost_estimate_usd": 900000,
  "recommendation": "..."
}}
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )


def parse_cost_output(
    raw_output: str,
    disruption_cost_per_day_usd: int,
    estimated_delay_days: int,
    total_disruption_cost_usd: int,
    has_alternatives: bool = True,
) -> dict:
    """
    Parses LLM output, then overwrites the three deterministic fields with
    the Python-computed values so they are always exactly correct.
    """
    text = str(raw_output).strip()

    data = None
    try:
        repaired = json_repair.repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            data = repaired
    except Exception:
        pass

    if data is None:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)

    # Always overwrite with deterministic Python values — LLM must not drift these
    data["disruption_cost_per_day_usd"] = disruption_cost_per_day_usd
    data["estimated_delay_days"] = estimated_delay_days
    data["total_disruption_cost_usd"] = total_disruption_cost_usd

    if not has_alternatives:
        data["rerouting_cost_estimate_usd"] = None

    return CostAnalysisOutput.model_validate(data).model_dump()
