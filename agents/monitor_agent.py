import os
import json
import re
import time
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import litellm
import json_repair
from crewai import Agent, Task, LLM

load_dotenv()

# Patch LiteLLM to ensure compatibility between CrewAI and Groq API.
# 1. CrewAI injects `cache_breakpoint` into system messages which Groq rejects.
# 2. Handles Groq rate limits (TPM/RPM) with automatic backoff and retry.
# 3. Seamlessly fails over to secondary model if daily quota (TPD) is reached.
# 4. Retries on transient empty/None responses or connection resets.
_orig_completion = litellm.completion


def _safe_litellm_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        kwargs["messages"] = [
            {k: v for k, v in m.items() if k != "cache_breakpoint"}
            if isinstance(m, dict)
            else m
            for m in kwargs["messages"]
        ]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = _orig_completion(*args, **kwargs)
            # Check for empty response and retry if necessary
            if hasattr(resp, "choices") and resp.choices:
                content = getattr(resp.choices[0].message, "content", None)
                if content is None or (isinstance(content, str) and not content.strip()):
                    if attempt < max_retries - 1:
                        time.sleep(2.0)
                        continue
            return resp
        except Exception as e:
            err_str = str(e).lower()
            if ("tokens per day" in err_str or "tpd" in err_str) and attempt < max_retries - 1:
                # Failover to alternate high-capacity Groq model
                if "model" in kwargs and "qwen" not in kwargs["model"]:
                    kwargs["model"] = "groq/qwen/qwen3.8-27b"
                time.sleep(1.0)
                continue
            elif ("rate limit" in err_str or "429" in err_str or "tpm" in err_str) and attempt < max_retries - 1:
                wait_time = 6.0 + (attempt * 3.0)
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    try:
                        wait_time = float(match.group(1)) + 1.5
                    except ValueError:
                        pass
                time.sleep(wait_time)
            elif attempt < max_retries - 1 and any(
                k in err_str
                for k in [
                    "invalid response",
                    "empty",
                    "timeout",
                    "internalservererror",
                    "500",
                    "10054",
                    "connection",
                    "remote host",
                    "server error",
                ]
            ):
                time.sleep(3.0)
            else:
                raise e

    return _orig_completion(*args, **kwargs)


litellm.completion = _safe_litellm_completion


class DisruptionEventOutput(BaseModel):
    is_disruption_relevant: bool = Field(
        description="Whether the news event describes an actual or impending supply chain disruption"
    )
    event_type: Optional[str] = Field(
        default=None,
        description="Type of disruption event. Null if not relevant."
    )
    location: Optional[str] = Field(
        default=None,
        description="Port, canal, strait, or region affected. Null if not relevant."
    )
    date: Optional[str] = Field(
        default=None,
        description="Date or timeframe of disruption. Null if not relevant."
    )
    severity: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Assessed severity level ('low', 'medium', 'high'). Null if not relevant."
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score between 0.0 and 1.0."
    )


def get_default_llm() -> LLM:
    """
    Initializes Groq LLM.
    Defaults to 'groq/qwen/qwen3.8-27b' (or GROQ_MODEL in .env).
    """
    model_name = os.getenv("GROQ_MODEL", "groq/qwen/qwen3.8-27b")
    if not model_name.startswith("groq/"):
        model_name = f"groq/{model_name}"

    api_key = os.getenv("GROQ_API_KEY")
    return LLM(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096
    )


def create_monitor_agent(llm: Optional[LLM] = None) -> Agent:
    """
    Creates and returns the Disruption Event Monitor agent.
    """
    if llm is None:
        llm = get_default_llm()

    return Agent(
        role="Disruption Event Monitor",
        goal="Extract structured, verifiable disruption events from raw news text",
        backstory=(
            "You are a specialized maritime and global logistics intelligence analyst with deep expertise "
            "in identifying supply-chain-relevant events hidden within noisy media feeds. You quickly distinguish "
            "actual disruptions (such as port blockages, canal grounding incidents, severe maritime weather, labor "
            "strikes, and maritime conflict) from routine business news, macroeconomic commentary, or irrelevant noise. "
            "When a disruption is verified, you extract factual details with high precision; when the text is irrelevant "
            "to supply chain operations, you decisively mark it as non-disruptive."
        ),
        verbose=False,
        llm=llm
    )


def create_monitor_task(agent: Agent, news_event: dict) -> Task:
    """
    Creates a CrewAI task to analyze a single news event item.
    """
    headline = news_event.get("headline", "")
    raw_text = news_event.get("raw_text", "")
    event_date = news_event.get("date", "")
    source = news_event.get("source", "")

    description = f"""
Analyze this news item for supply chain disruption relevance:

- Headline: {headline}
- Date: {event_date}
- Source: {source}
- Text: {raw_text}

Rules:
1. If this is NOT an actual supply chain disruption (e.g. general market report, unrelated news), output:
   {{"is_disruption_relevant": false, "event_type": null, "location": null, "date": null, "severity": null, "confidence": 0.0}}
2. If this IS a supply chain disruption, extract:
   - event_type: specific category (e.g., "Canal Blockage", "Port Congestion", "Labor Strike", "Extreme Weather", "Maritime Conflict")
   - location: specific port, canal, strait, or maritime region affected
   - date: date or timeframe (YYYY-MM-DD format if available)
   - severity: "low", "medium", or "high"
   - confidence: float between 0.0 and 1.0 reflecting confidence
3. Return ONLY valid, raw JSON matching the schema. Do not enclose in markdown ticks or include conversational commentary.
"""

    expected_output = """
A strict raw JSON object with keys:
{
  "is_disruption_relevant": true,
  "event_type": "...",
  "location": "...",
  "date": "...",
  "severity": "low",
  "confidence": 0.95
}
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )


def parse_disruption_output(raw_output: str) -> dict:
    """
    Cleans and parses raw agent output into a validated dictionary conforming to DisruptionEventOutput.
    """
    text = str(raw_output).strip()

    try:
        data = json_repair.repair_json(text, return_objects=True)
        if isinstance(data, dict):
            return DisruptionEventOutput.model_validate(data).model_dump()
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
    return DisruptionEventOutput.model_validate(data).model_dump()
