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
# Llama-3.3-70B-Instruct: Excellent reasoning and critique capabilities
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_CRITIC_MODEL = os.getenv(
    "HUGGINGFACE_CRITIC_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
)
USE_LLM_CRITIC = os.getenv("USE_LLM_CRITIC", "false").lower() in ("1", "true")

critic_client = None
if HUGGINGFACE_API_KEY and USE_LLM_CRITIC:
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

    query_validation_prompt = """You are a travel feasibility reviewer with expertise in geography and tourism.

Evaluate the following user travel request for feasibility and authenticity.

Assess:
1. **Destination Reality**: Does this destination actually exist on Earth?
2. **Tourism Accessibility**: Can tourists visit? Is infrastructure available?
3. **Safety/Feasibility**: Are there insurmountable barriers (war, natural disaster) currently?

Respond with JSON only:
{
    "reasonable": true|false,
    "reason": "Clear explanation"
}

Rules:
- Fictional places (Mars, Atlantis, etc.) → reasonable: false
- Real but obscure places → reasonable: true
- Vague but plausible → reasonable: true
Return ONLY JSON, no other text."""

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
    """Optionally use an LLM critic to evaluate a trip plan's quality.

    This runs only if USE_LLM_CRITIC is enabled and the plan confidence is low.

    Returns:
        Dictionary with:
        - "valid": bool
        - "issues": list
    """
    # Skip LLM criticism unless explicitly enabled
    if not USE_LLM_CRITIC or not critic_client:
        logger.info("Skipping LLM critic (disabled or not initialized)")
        return {"valid": True, "issues": []}

    # If plan confidence is high, skip LLM critique for cost/performance.
    if getattr(tripplan, "confidence", 1.0) >= 0.85:
        logger.info("TripPlan confidence is high (>=0.85); skipping LLM critic")
        return {"valid": True, "issues": []}

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


def _check_activity_density(itinerary: List[DayPlan]) -> List[str]:
    """
    Check for unrealistic activity density (too many activities per day).
    
    CRITICAL VALIDATION: This catches unrealistic itineraries.
    Typical ranges:
    - Relaxed: 3-4 activities
    - Busy: 5-6 activities  
    - Maximum realistic: 7-8 activities
    - Impossible: >8 activities

    Args:
        itinerary: List of DayPlan objects

    Returns:
        List of issue strings for excessive activity density
    """
    issues = []
    MAX_ACTIVITIES_PER_DAY = 8
    TYPICAL_ACTIVITIES = 5

    for day_plan in itinerary:
        num_activities = len(day_plan.activities)
        if num_activities > MAX_ACTIVITIES_PER_DAY:
            logger.error(
                f"CRITICAL: Activity density violation on Day {day_plan.day}: "
                f"{num_activities} activities (max realistic: {MAX_ACTIVITIES_PER_DAY})"
            )
            issues.append(
                f"Day {day_plan.day}: Your request includes {num_activities} visits. "
                f"This is physically impossible. Travel time between attractions, "
                f"entrance queues, and site exploration make this unrealistic. "
                f"Typical itineraries include {TYPICAL_ACTIVITIES} attractions per day. "
                f"I will generate a more balanced itinerary.")
        elif num_activities > 6:
            logger.warning(
                f"Activity density warning on Day {day_plan.day}: {num_activities} activities "
                f"(beyond typical range of {TYPICAL_ACTIVITIES})"
            )

    return issues


def _check_duplicate_activities(itinerary: List[DayPlan]) -> List[str]:
    """
    Check for duplicate activities and nested attractions.
    
    Detects:
    1. Exact duplicates (same activity on multiple days)
    2. Nested sub-attractions (e.g., "Diwan-i-Khas" inside "Agra Fort"
       should be consolidated as one visit)

    Args:
        itinerary: List of DayPlan objects

    Returns:
        List of issue strings for duplicates and nested attractions
    """
    issues = []
    activity_map: Dict[str, List[int]] = {}
    all_activities: List[tuple] = []  # (name, location, day)

    # Build map of activities to days they appear in
    for day_plan in itinerary:
        for activity in day_plan.activities:
            activity_name = activity.name.lower().strip()
            location = activity.location.lower().strip() if activity.location else ""
            
            if activity_name not in activity_map:
                activity_map[activity_name] = []
            activity_map[activity_name].append(day_plan.day)
            all_activities.append((activity.name, location, day_plan.day))

    # Check for exact duplicates
    for activity_name, days in activity_map.items():
        if len(days) > 1:
            day_list = ", ".join(str(d) for d in days)
            logger.warning(f"Duplicate activity: {activity_name} on days {day_list}")
            issues.append(
                f"Duplicate activity: '{activity_name}' scheduled on Days {day_list}. "
                f"Remove one instance.")

    # Check for nested sub-attractions of same monument
    nested_map = {
        "agra fort": ["diwan-i-khas", "diwan-i-am", "jahangir mahal", "pearl mosque", "musamman burj"],
        "taj mahal": ["taj nature walk", "taj gardens", "mehtab bagh nearby"],
        "forbidden city": ["palace", "pavilion", "courtyard", "gate"],
        "colosseum": ["arena", "underground", "tier"],
    }

    for monument, sub_list in nested_map.items():
        # Find all activities related to this monument
        monument_activities = []
        for act_name, location, day in all_activities:
            if monument in location or monument in act_name.lower():
                monument_activities.append(act_name)

        # Check for multiple sub-attractions from same monument
        sub_found = []
        for sub in sub_list:
            for act in monument_activities:
                if sub in act.lower():
                    sub_found.append(act)

        if len(sub_found) > 1:
            logger.warning(
                f"Nested attractions in {monument}: {sub_found}. "
                f"These should be consolidated into single visit."
            )
            issues.append(
                f"Sub-locations detected: You have {sub_found[0]} and {sub_found[1]} "
                f"scheduled separately. These are both parts of {monument.title()} "
                f"and should be visited as one attraction.")

    return issues


def _check_activity_costs(itinerary: List[DayPlan]) -> List[str]:
    """
    Check for unrealistic or placeholder activity costs.
    
    LLMs often hallucinate costs, returning 0 for everything. This check
    catches:
    - All costs are 0 (placeholder data)
    - Major attractions cost suspiciously little ($1-2 for Taj Mahal)
    - Unrealistic bulk pricing

    Args:
        itinerary: List of DayPlan objects

    Returns:
        List of issue strings for cost problems
    """
    issues = []
    zero_cost_count = 0
    total_activities = 0
    all_costs = []

    for day_plan in itinerary:
        for activity in day_plan.activities:
            total_activities += 1
            cost = getattr(activity, 'estimated_cost', 0) or 0
            all_costs.append(cost)
            
            if cost == 0 or cost is None:
                zero_cost_count += 1

    # If more than 50% have zero cost, it's placeholder data
    if total_activities > 0 and zero_cost_count >= (total_activities * 0.5):
        logger.error(
            f"Cost validation FAILED: {zero_cost_count}/{total_activities} activities have $0 cost"
        )
        issues.append(
            f"Cost validation issue: {zero_cost_count} of {total_activities} activities have $0 estimated cost. "
            f"This appears to be placeholder data. All major attractions should have realistic pricing "
            f"(typically $5-50 per site).")

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
    Validates a TripPlan object using rule-based checks.

    Checks for:
    1. Duplicate activities in the itinerary
    2. Empty itinerary days
    3. Suspicious locations not in retrieved search results
    4. Excessive activity density (max 8 per day)

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

    logger.info(f"Validating TripPlan: {trip_plan.city}, {trip_plan.days} days")

    # Rule 1: CRITICAL — Check activity density (must be first)
    logger.info("[Rule 1/5] Checking activity density...")
    density_issues = _check_activity_density(trip_plan.itinerary)
    all_issues.extend(density_issues)
    if density_issues:
        logger.error(f"❌ Activity density check FAILED with {len(density_issues)} issues")
        return {"valid": False, "issues": all_issues}  # Fail fast on density

    # Rule 2: Check for duplicate and nested activities
    logger.info("[Rule 2/5] Checking for duplicates and nested attractions...")
    dup_issues = _check_duplicate_activities(trip_plan.itinerary)
    all_issues.extend(dup_issues)
    if dup_issues:
        logger.warning(f"⚠️  Duplicate/nested check found {len(dup_issues)} issues")

    # Rule 3: Check activity costs (NEW - important)
    logger.info("[Rule 3/5] Validating activity costs...")
    cost_issues = _check_activity_costs(trip_plan.itinerary)
    all_issues.extend(cost_issues)
    if cost_issues:
        logger.error(f"❌ Cost validation FAILED with {len(cost_issues)} issues")

    # Rule 4: Check for empty days
    logger.info("[Rule 4/5] Checking for empty days...")
    empty_issues = _check_empty_days(trip_plan.itinerary, trip_plan.days)
    all_issues.extend(empty_issues)
    if empty_issues:
        logger.warning(f"⚠️  Empty day check found {len(empty_issues)} issues")

    # Rule 5: Check for suspicious locations
    logger.info("[Rule 5/5] Checking for suspicious locations...")
    loc_issues = _check_suspicious_locations(
        trip_plan.itinerary, search_results)
    all_issues.extend(loc_issues)
    if loc_issues:
        logger.warning(f"⚠️  Location check found {len(loc_issues)} issues")

    return {
        "valid": len(all_issues) == 0,
        "issues": all_issues
    }


def _generate_suggestion_for_rule_issues(issues: List[str], trip_plan: TripPlan) -> str:
    """
    Generate a suggestion on how to fix rule validation issues.

    Args:
        issues: List of validation issues detected
        trip_plan: The TripPlan object

    Returns:
        A suggestion string for fixing the issues
    """
    if not issues:
        return "No issues detected."

    suggestions = []

    # Categorize issues and provide specific suggestions
    for issue in issues:
        if "Duplicate activity" in issue:
            suggestions.append("Remove or consolidate duplicate activities across days.")
        elif "has no activities scheduled" in issue:
            suggestions.append("Add activities to all empty days in your itinerary.")
        elif "contains" in issue and "activities" in issue and "too many" in issue:
            suggestions.append("Reduce the number of activities per day (aim for 3-6 per day for a comfortable itinerary).")
        elif "Suspicious location" in issue or "not found in search results" in issue:
            suggestions.append("Choose activities from the list of popular attractions at your destination.")

    if not suggestions:
        suggestions.append("Please review your itinerary and make the necessary adjustments.")

    return " ".join(suggestions) if suggestions else "Please revise your itinerary to address the issues."


def _generate_suggestion_for_llm_issues(issues: List[str], trip_plan: TripPlan) -> str:
    """
    Generate a suggestion on how to fix LLM critique issues.

    Args:
        issues: List of critique issues detected
        trip_plan: The TripPlan object

    Returns:
        A suggestion string for fixing the issues
    """
    if not issues:
        return "No issues detected."

    suggestions = []

    # Categorize issues and provide specific suggestions
    for issue in issues:
        if "hallucinated" in issue.lower() or "not found" in issue.lower():
            suggestions.append("Replace any fictional or unrealistic locations with verified tourist attractions.")
        elif "feasibil" in issue.lower() or "realistic" in issue.lower():
            suggestions.append("Ensure your itinerary is realistic for the destination and trip duration.")
        elif "flow" in issue.lower() or "logical" in issue.lower():
            suggestions.append("Reorganize activities to create a more logical geographical flow and minimize travel time.")

    if not suggestions:
        suggestions.append("Please review your itinerary for feasibility and realism.")

    return " ".join(suggestions) if suggestions else "Please revise your itinerary to improve its quality and realism."


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
    Critic node implementing Hybrid Validation System.

    Two-stage sequential validation:
    
    Stage 1 — Rule Validator (Deterministic)
    - Empty day detection
    - Duplicate activities
    - Suspicious locations
    - Activity density (max 8/day)
    
    Stage 2 — LLM Critic (Semantic Reasoning)
    - Itinerary realism
    - Travel feasibility
    - Activity coherence
    - Hallucinated attractions

    Only if both stages pass does the plan get accepted.

    If validation fails at any stage, sets critic_reject=True and returns
    explanation message. Planner can then regenerate via decision_node.

    Does NOT format the output (formatter_node does that).

    Args:
        state: The AgentState containing tripplan
        config: RunnableConfig for LangGraph execution

    Returns:
        Updated AgentState with validation result and critic_reject flag
    """

    try:
        # Only validate if there's a TripPlan
        tripplan = state.get("tripplan")

        # TripPlan normalization
        if isinstance(tripplan, dict):
            tripplan = TripPlan(**tripplan)
            state["tripplan"] = tripplan
        elif tripplan is not None and not isinstance(tripplan, TripPlan):
            tripplan = None
            state["tripplan"] = None

        # If no TripPlan, skip all validation
        if not tripplan:
            logger.info("No TripPlan in state, skipping critic validation")
            state["critic_reject"] = False
            return state

        logger.info("=" * 60)
        logger.info("HYBRID CRITIC VALIDATION SYSTEM")
        logger.info("=" * 60)
        logger.info("Validating TripPlan for city: %s", tripplan.city)

        # =========================================================
        # STAGE 1: RULE VALIDATOR (Fast, Deterministic)
        # =========================================================
        logger.info("\n--- STAGE 1: RULE VALIDATOR ---")
        logger.info("Running rule-based validation...")
        
        rule_validation_result = validate_tripplan(tripplan)
        rule_valid = rule_validation_result.get("valid", False)
        rule_issues = rule_validation_result.get("issues", [])

        if not rule_valid:
            # Rule validation FAILED
            logger.warning("❌ RULE VALIDATION FAILED")
            logger.warning("Rule validation issues:")
            for issue in rule_issues:
                logger.warning("  - %s", issue)
            
            # Generate suggestion based on issues
            suggestion = _generate_suggestion_for_rule_issues(rule_issues, tripplan)
            
            # Store structured feedback
            critic_feedback = {
                "valid": False,
                "issues": rule_issues,
                "suggestion": suggestion,
                "stage": "rule_validation"
            }
            state["critic_feedback"] = critic_feedback
            
            logger.info("Critic feedback: %s", critic_feedback)
            
            state["critic_reject"] = True
            state["messages"] = [
                AIMessage(content=f"Issues found in itinerary:\n" + "\n".join([f"• {issue}" for issue in rule_issues]))
            ]
            logger.info("Decision: REJECT (rule violations)")
            return state
        
        logger.info("✓ RULE VALIDATION PASSED")
        logger.info("Rule issues: %s", rule_issues if rule_issues else "None")

        # =========================================================
        # STAGE 2: LLM CRITIC (Semantic Reasoning)
        # =========================================================
        logger.info("\n--- STAGE 2: LLM CRITIC ---")
        logger.info("Running LLM-based semantic critique...")
        
        llm_validation_result = await evaluate_tripplan_quality(tripplan)
        llm_valid = llm_validation_result.get("valid", False)
        llm_issues = llm_validation_result.get("issues", [])

        if not llm_valid:
            # LLM critique FAILED
            logger.warning("❌ LLM CRITIC FAILED")
            logger.warning("LLM critique issues:")
            for issue in llm_issues:
                logger.warning("  - %s", issue)
            
            # Generate suggestion based on issues
            suggestion = _generate_suggestion_for_llm_issues(llm_issues, tripplan)
            
            # Store structured feedback
            critic_feedback = {
                "valid": False,
                "issues": llm_issues,
                "suggestion": suggestion,
                "stage": "llm_critique"
            }
            state["critic_feedback"] = critic_feedback
            
            logger.info("Critic feedback: %s", critic_feedback)
            
            state["critic_reject"] = True
            state["messages"] = [
                AIMessage(content=f"Trip plan has issues:\n" + "\n".join([f"• {issue}" for issue in llm_issues]))
            ]
            logger.info("Decision: REJECT (LLM critique)")
            return state

        logger.info("✓ LLM CRITIC PASSED")
        logger.info("LLM issues: %s", llm_issues if llm_issues else "None")

        # =========================================================
        # VALIDATION COMPLETE: ALL STAGES PASSED
        # =========================================================
        logger.info("\n" + "=" * 60)
        logger.info("✓✓✓ HYBRID VALIDATION PASSED ✓✓✓")
        logger.info("=" * 60)
        logger.info("Decision: ACCEPT (route to formatter)")
        
        # Store success feedback
        critic_feedback = {
            "valid": True,
            "issues": [],
            "suggestion": "No issues detected. Itinerary is ready!",
            "stage": "all_validations_passed"
        }
        state["critic_feedback"] = critic_feedback
        logger.info("Critic feedback: %s", critic_feedback)
        
        state["critic_reject"] = False
        state["messages"] = []  # No messages - formatter will add output
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in critic_node: {e}", exc_info=True)
        state["critic_reject"] = False
        state["messages"] = [AIMessage(content="Internal validation error. Please try again.")]
        return state
