"""
Critic node for validating trip plans using LLM-as-Judge system.

This module provides comprehensive validation for user queries and TripPlan objects
using a dedicated Hugging Face model optimized for reasoning/critique. Falls back to
rule-based validation if the LLM service is unavailable.

Validates:
- Query reasonableness and travel feasibility
- Hallucinated/unrealistic destinations
- Duplicate activities in the itinerary
- Empty itinerary days
- Impossible travel routes
"""

from typing import List, Dict, Any
from schemas.trip_schema import TripPlan, DayPlan
from difflib import SequenceMatcher
from travel.state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Critic LLM client
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_CRITIC_MODEL = os.getenv(
    "HUGGINGFACE_CRITIC_MODEL", "Qwen/Qwen2.5-72B-Instruct"
)

critic_client = None
if HUGGINGFACE_API_KEY:
    critic_client = InferenceClient(api_key=HUGGINGFACE_API_KEY)


def _normalize_location(location: str) -> str:
    """Normalize location string for comparison."""
    return location.lower().strip()


def _similarity(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar enough."""
    matcher = SequenceMatcher(None, a.lower(), b.lower())
    return matcher.ratio() >= threshold


async def evaluate_query_reasonableness(query: str) -> Dict[str, Any]:
    """
    Evaluate if a user's travel query is reasonable and feasible.

    Uses an LLM-based judge to assess:
    - Destination realism (does the destination exist on Earth?)
    - Tourism feasibility (is it accessible for tourism?)
    - Travel safety (are there known safety concerns?)

    Args:
        query: The user's travel query/request

    Returns:
        Dictionary with:
        - "reasonable": bool indicating if query is reasonable
        - "reason": string explanation of the assessment
    """
    if not critic_client:
        logger.warning(
            "Critic LLM client not initialized, skipping query validation")
        # Fallback: allow query to proceed
        return {"reasonable": True, "reason": "Query validation skipped (no API key)"}

    query_validation_prompt = """You are a travel feasibility reviewer.

Determine if the following user request represents a reasonable real-world travel plan that is possible to execute.

Consider and assess:
- **Destination Realism**: Does the destination actually exist on Earth? Is it a real, geographically valid location?
- **Tourism Feasibility**: Can tourists realistically visit this destination? Are there commercial flights, hotels, or tourism infrastructure?
- **Travel Safety**: Are there major safety, health, or political issues that would prevent tourism?

Return a JSON object with this exact format:
{
    "reasonable": true or false,
    "reason": "Short explanation of your assessment"
}

If the destination is fictional (e.g., Mars, Atlantis, Narnia) or unrealistic (e.g., "a place that doesn't exist"), set reasonable to false.
If the query is vague but could be a real place, set it to true.
Return ONLY valid JSON, no other text."""

    try:
        completion = critic_client.chat.completions.create(
            model=HUGGINGFACE_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": query_validation_prompt},
                {"role": "user", "content": f"Is this travel request reasonable?\n\nUser request: {query}"}
            ],
            max_tokens=256,
            temperature=0.3
        )

        response_text = completion.choices[0].message.content
        logger.info(f"Query reasonableness response: {response_text}")

        json_response = json.loads(response_text)
        reasonable = json_response.get("reasonable", True)
        reason = json_response.get("reason", "Assessment complete")

        logger.info(
            "Query validation result: reasonable=%s, reason=%s", reasonable, reason)

        return {
            "reasonable": reasonable,
            "reason": reason
        }

    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse query validation response as JSON: {e}")
        # Fallback: allow query if parsing fails
        return {"reasonable": True, "reason": "Query validation skipped (parse error)"}
    except Exception as e:
        logger.warning(f"Query reasonableness check failed: {e}")
        # Fallback: allow query if LLM fails
        return {"reasonable": True, "reason": "Query validation skipped (LLM error)"}


async def evaluate_tripplan_quality(tripplan: TripPlan) -> Dict[str, Any]:
    """
    Use the critic LLM to evaluate a trip plan's quality and feasibility.

    Evaluates:
    - Hallucinated locations (do they actually exist?)
    - Duplicate activities across days
    - Unrealistic travel routes
    - Empty days with no activities

    Args:
        tripplan: The TripPlan object to validate

    Returns:
        Dictionary with:
        - "valid": bool indicating if trip plan is acceptable
        - "issues": list of detected issues (empty if valid)
    """
    if not critic_client:
        logger.warning(
            "Critic LLM client not initialized, falling back to rule-based validation")
        return validate_tripplan(tripplan)

    trip_plan_dict = tripplan.model_dump()
    trip_plan_json = json.dumps(trip_plan_dict, indent=2)

    tripplan_validation_prompt = """You are a critical travel plan validator. Your job is to evaluate TripPlan JSON objects for quality and feasibility.

Evaluate the following criteria:
1. **Hallucinated Locations**: Check if all mentioned locations are real and exist on Earth
2. **Realistic Destinations**: Ensure destinations and activities are suitable for the travel duration
3. **Duplicate Activities**: Identify if the same activity appears multiple times across days
4. **Empty Days**: Check if all days have at least one activity scheduled
5. **Impossible Travel**: Detect geographically impossible or impractical travel routes
6. **Activity Coherence**: Ensure activities make sense for the location and theme

Return a JSON object with this exact format:
{
    "valid": true or false,
    "issues": ["issue 1", "issue 2", ...]
}

If the trip plan is valid and realistic, return {"valid": true, "issues": []}.
If there are any issues, set valid to false and list all detected problems in issues array.
Return ONLY valid JSON, no other text."""

    try:
        completion = critic_client.chat.completions.create(
            model=HUGGINGFACE_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": tripplan_validation_prompt},
                {"role": "user", "content": f"Evaluate this trip plan:\n\n{trip_plan_json}"}
            ],
            max_tokens=512,
            temperature=0.3
        )

        response_text = completion.choices[0].message.content
        logger.info(f"TripPlan quality response: {response_text}")

        json_response = json.loads(response_text)
        valid = json_response.get("valid", False)
        issues = json_response.get("issues", [])

        logger.info(
            "TripPlan validation result: valid=%s, issues=%s", valid, issues)

        return {
            "valid": valid,
            "issues": issues
        }

    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse trip plan validation response as JSON: {e}")
        # Fallback to rule-based validation
        logger.info(
            "Falling back to rule-based validation due to JSON parse error")
        return validate_tripplan(tripplan)
    except Exception as e:
        logger.warning(f"TripPlan quality evaluation failed: {e}")
        # Fallback to rule-based validation
        logger.info("Falling back to rule-based validation due to LLM error")
        return validate_tripplan(tripplan)


def _check_duplicate_activities(itinerary: List[DayPlan]) -> List[str]:
    """
    Check for duplicate activities in the itinerary.

    Args:
        itinerary: List of DayPlan objects

    Returns:
        List of issue strings for duplicate activities
    """
    issues = []
    activity_map: Dict[str, List[int]] = {}

    # Build map of activities to days they appear in
    for day_plan in itinerary:
        for activity in day_plan.activities:
            activity_name = activity.name.lower().strip()
            if activity_name not in activity_map:
                activity_map[activity_name] = []
            activity_map[activity_name].append(day_plan.day)

    # Find duplicates
    for activity_name, days in activity_map.items():
        if len(days) > 1:
            day_list = ", ".join(str(d) for d in days)
            issues.append(
                f"Duplicate activity '{activity_name}' found on days: {day_list}")

    return issues


def _check_empty_days(itinerary: List[DayPlan], total_days: int) -> List[str]:
    """
    Check for empty itinerary days.

    Args:
        itinerary: List of DayPlan objects
        total_days: Total number of days in the trip

    Returns:
        List of issue strings for empty days
    """
    issues = []
    days_with_activities = {day_plan.day for day_plan in itinerary}

    # Check for missing days
    for day in range(1, total_days + 1):
        if day not in days_with_activities:
            issues.append(f"Day {day} has no activities scheduled")
        else:
            # Check if day has empty activities list
            for day_plan in itinerary:
                if day_plan.day == day and len(day_plan.activities) == 0:
                    issues.append(f"Day {day} has an empty activities list")
                    break

    return issues


def _check_suspicious_locations(
    itinerary: List[DayPlan],
    search_results: List[str]
) -> List[str]:
    """
    Check for suspicious locations not in retrieved search results.

    Args:
        itinerary: List of DayPlan objects
        search_results: List of place names from search

    Returns:
        List of issue strings for suspicious locations
    """
    issues = []

    if not search_results:
        # If no search results, we can't validate locations
        return issues

    # Normalize search results
    normalized_results = [_normalize_location(
        result) for result in search_results]

    # Check each activity location
    for day_plan in itinerary:
        for activity in day_plan.activities:
            activity_location = _normalize_location(activity.location)
            activity_name = _normalize_location(activity.name)

            # Check if location or name matches any search result
            found_match = False
            for result in normalized_results:
                if (activity_location == result or
                    activity_name == result or
                    _similarity(activity_location, result) or
                        _similarity(activity_name, result)):
                    found_match = True
                    break

            if not found_match:
                issues.append(
                    f"Suspicious location on Day {day_plan.day}: "
                    f"'{activity.name}' at '{activity.location}' "
                    f"not found in search results"
                )

    return issues


def validate_tripplan(
    trip_plan: TripPlan,
    search_results: List[str] = None
) -> Dict[str, Any]:
    """
    Validates a TripPlan object.

    Checks for:
    1. Duplicate activities in the itinerary
    2. Empty itinerary days
    3. Suspicious locations not in retrieved search results

    Args:
        trip_plan: The TripPlan object to validate
        search_results: Optional list of place names from search results

    Returns:
        Dictionary with:
        - "valid": bool indicating if itinerary passes validation
        - "issues": list of detected problems (empty if valid)
    """
    if search_results is None:
        search_results = []

    all_issues = []

    # Check for duplicate activities
    all_issues.extend(_check_duplicate_activities(trip_plan.itinerary))

    # Check for empty days
    all_issues.extend(_check_empty_days(trip_plan.itinerary, trip_plan.days))

    # Check for suspicious locations
    all_issues.extend(_check_suspicious_locations(
        trip_plan.itinerary, search_results))

    return {
        "valid": len(all_issues) == 0,
        "issues": all_issues
    }


async def evaluate_with_critic_llm(trip_plan_json: str) -> Dict[str, Any]:
    """
    Use the critic LLM to evaluate a trip plan (legacy function).

    Deprecated in favor of evaluate_tripplan_quality().
    Kept for backward compatibility.

    Args:
        trip_plan_json: JSON string representation of the trip plan

    Returns:
        Dictionary with validation results
    """
    if not critic_client:
        logger.warning("Critic LLM client not initialized (missing API key)")
        return {"valid": None, "issues": [], "error": "No critic API key"}

    critic_system_prompt = """You are a critical travel plan validator. Your job is to evaluate TripPlan JSON objects for quality and feasibility.

Evaluate the following criteria:
1. **Hallucinated Locations**: Check if all mentioned locations are real and exist on Earth
2. **Realistic Destinations**: Ensure destinations are suitable for the travel duration (e.g., not flying to 10 countries in 2 days)
3. **Duplicate Activities**: Identify if the same activity appears multiple times across days
4. **Empty Days**: Check if all days have at least one activity scheduled
5. **Impossible Travel**: Detect geographically impossible or impractical travel routes
6. **Activity Coherence**: Ensure activities make sense for the location and theme

Return a JSON object with this exact format:
{
    "valid": true or false,
    "issues": ["issue 1", "issue 2", ...]
}

If the trip plan is valid and realistic, return {"valid": true, "issues": []}.
If there are any issues, set valid to false and list all detected problems in issues array.
Return ONLY valid JSON, no other text."""

    try:
        completion = critic_client.chat.completions.create(
            model=HUGGINGFACE_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": critic_system_prompt},
                {"role": "user", "content": f"Evaluate this trip plan:\n\n{trip_plan_json}"}
            ],
            max_tokens=512,
            temperature=0.3  # Lower temperature for more consistent critique
        )

        response_text = completion.choices[0].message.content
        logger.info(f"Critic LLM response: {response_text}")

        # Parse the JSON response
        json_response = json.loads(response_text)
        valid = json_response.get("valid", False)
        issues = json_response.get("issues", [])

        logger.info(
            f"Critic validation result: valid={valid}, issues={issues}")

        return {
            "valid": valid,
            "issues": issues,
            "error": None
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse critic LLM response as JSON: {e}")
        return {"valid": None, "issues": [], "error": f"JSON parsing error: {str(e)}"}
    except Exception as e:
        logger.warning(f"Critic LLM evaluation failed: {e}")
        return {"valid": None, "issues": [], "error": str(e)}


@traceable
async def critic_node(state: AgentState, config: RunnableConfig):
    """
    Async LangGraph node that validates and formats TripPlan.

    Validates generated TripPlan quality using rule-based checks:
    - No duplicate activities across days
    - All days have activities scheduled
    - No empty activities lists

    If valid, formats the TripPlan for display in the UI.

    Returns state with:
    - critic_reject flag (True if validation fails, False if passes)
    - Formatted trip content in messages if valid
    - Error message in messages if validation fails

    Args:
        state: The AgentState containing messages and structured TripPlan
        config: RunnableConfig for LangGraph execution

    Returns:
        Updated AgentState with validation result and critic_reject flag
    """

    try:
        # Only validate if there's a TripPlan to validate
        tripplan = state.get("tripplan")

        # --- TripPlan normalization ---
        if isinstance(tripplan, dict):
            tripplan = TripPlan(**tripplan)
            state["tripplan"] = tripplan
        elif tripplan is not None and not isinstance(tripplan, TripPlan):
            tripplan = None
            state["tripplan"] = None

        if not tripplan:
            logger.info("No TripPlan in state, skipping critic validation")
            state["critic_reject"] = False
            return state

        logger.info("Validating TripPlan for city: %s", tripplan.city)
        logger.info("Running rule-based TripPlan validation...")
        validation_result = validate_tripplan(tripplan)

        if validation_result.get("valid", False):
            logger.info("TripPlan validation passed")
            state["critic_reject"] = False
        else:
            issues = validation_result.get("issues", [])
            logger.warning(
                "TripPlan validation failed with issues: %s", issues)
            state["critic_reject"] = True
            # Append critic rejection message, do not overwrite messages
            if "messages" not in state or not isinstance(state["messages"], list):
                state["messages"] = []
            state["messages"].append(
                AIMessage(content=f"Issues found in itinerary: {issues}")
            )
        return state
    except Exception as e:
        logger.error(f"Unexpected error in critic_node: {e}", exc_info=True)
        state["critic_reject"] = False
        return state
