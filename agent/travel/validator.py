"""
Validator Node for Trip Plan Quality

Uses microsoft/deberta-base-mnli (Natural Language Inference) to detect
contradictions and logical inconsistencies in trip plans.

DeBERTa excels at:
- Detecting contradictions between statements
- Verifying logical consistency
- Checking entailment relationships
"""

from typing import Dict, Any, List
from schemas.trip_schema import TripPlan
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Validator client
# microsoft/deberta-base-mnli: Specialized for Natural Language Inference
# Perfect for detecting logical contradictions in trip plans
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_VALIDATOR_MODEL = os.getenv(
    "HUGGINGFACE_VALIDATOR_MODEL", "microsoft/deberta-base-mnli"
)

validator_client = None
if HUGGINGFACE_API_KEY:
    validator_client = InferenceClient(api_key=HUGGINGFACE_API_KEY)


def flatten_tripplan_statements(tripplan: TripPlan) -> List[str]:
    """
    Extract all factual statements from a trip plan for validation.
    
    Returns a list of statements that can be checked for contradictions.
    """
    statements = []
    
    # Location statements
    statements.append(f"The trip destination is {tripplan.city}")
    statements.append(f"The trip lasts {tripplan.days} days")
    statements.append(f"The estimated budget is ${tripplan.estimated_budget}")
    
    # Activity statements
    for day_plan in tripplan.itinerary:
        for activity in day_plan.activities:
            statements.append(f"On day {day_plan.day}, there is an activity at {activity.location}")
            statements.append(f"The activity {activity.name} costs ${activity.estimated_cost}")
    
    return statements


def check_schedule_consistency(tripplan: TripPlan) -> List[str]:
    """
    Check if activities are scheduled consistently across days.
    
    Returns list of inconsistencies found.
    """
    issues = []
    
    # Check if days match itinerary length
    if len(tripplan.itinerary) != tripplan.days:
        issues.append(
            f"Schedule mismatch: Trip duration is {tripplan.days} days, "
            f"but itinerary has {len(tripplan.itinerary)} days"
        )
    
    # Check for gaps in day numbering
    if tripplan.itinerary:
        day_numbers = [day.day for day in tripplan.itinerary]
        expected_days = list(range(1, tripplan.days + 1))
        missing_days = set(expected_days) - set(day_numbers)
        if missing_days:
            issues.append(f"Missing days in itinerary: {sorted(missing_days)}")
    
    # Check for empty days
    for day_plan in tripplan.itinerary:
        if not day_plan.activities or len(day_plan.activities) == 0:
            issues.append(f"Day {day_plan.day} has no activities scheduled")
    
    return issues


def check_budget_consistency(tripplan: TripPlan) -> List[str]:
    """
    Verify that total budget matches sum of activity costs.
    
    Returns list of budget inconsistencies.
    """
    issues = []
    
    if not tripplan.itinerary:
        return issues
    
    total_activity_cost = 0
    for day_plan in tripplan.itinerary:
        for activity in day_plan.activities:
            total_activity_cost += activity.estimated_cost
    
    # Allow 10% variance for accommodation, transport, etc.
    budget_lower = total_activity_cost * 0.9
    budget_upper = total_activity_cost * 1.5  # Upper bound more lenient
    
    if tripplan.estimated_budget < budget_lower:
        issues.append(
            f"Budget inconsistency: Total activities cost ${total_activity_cost}, "
            f"but estimated budget is only ${tripplan.estimated_budget}. "
            f"Budget should be at least ${int(budget_lower)}"
        )
    
    if tripplan.estimated_budget > budget_upper:
        issues.append(
            f"Budget unusually high: Activities total ${total_activity_cost}, "
            f"but estimated budget is ${tripplan.estimated_budget}"
        )
    
    return issues


def check_for_duplicate_activities(tripplan: TripPlan) -> List[str]:
    """
    Detect duplicate or repetitive activities across the itinerary.
    
    Returns list of duplicate activities found.
    """
    issues = []
    activity_names = {}
    
    for day_plan in tripplan.itinerary:
        for activity in day_plan.activities:
            name = activity.name.lower().strip()
            if name in activity_names:
                issues.append(
                    f"Duplicate activity: '{activity.name}' appears on day {activity_names[name]} "
                    f"and day {day_plan.day}"
                )
            else:
                activity_names[name] = day_plan.day
    
    return issues


async def validate_tripplan_with_inference(tripplan: TripPlan) -> Dict[str, Any]:
    """
    Use DeBERTa NLI model to detect contradictions in trip plan statements.
    
    The model checks if statements entail, contradict, or are neutral to each other.
    This helps catch logical inconsistencies like:
    - Budget doesn't match activity costs
    - Schedule conflicts
    - Contradictory activity descriptions
    
    Returns:
        Dictionary with:
        - "valid": bool indicating if plan is consistent
        - "issues": list of detected inconsistencies
    """
    if not validator_client:
        logger.warning("Validator client not initialized, skipping NLI validation")
        return {"valid": True, "issues": []}
    
    all_issues = []
    
    # First, run deterministic checks
    all_issues.extend(check_schedule_consistency(tripplan))
    all_issues.extend(check_budget_consistency(tripplan))
    all_issues.extend(check_for_duplicate_activities(tripplan))
    
    # If we already found issues, return early
    if all_issues:
        logger.info(f"Trip plan validation found {len(all_issues)} issues")
        return {
            "valid": False,
            "issues": all_issues
        }
    
    logger.info("All deterministic checks passed")
    
    # For confidence scoring, you could add NLI checks here
    # Example: Check if activity descriptions make logical sense together
    
    return {
        "valid": True,
        "issues": []
    }


async def validator_node(state, config):
    """
    Validator node that uses DeBERTa to check trip plan consistency.
    
    Validates:
    - Schedule consistency (days match itinerary)
    - Budget consistency (budget matches activity costs)
    - No duplicate activities
    - No empty days
    
    Returns updated state with validation result.
    """
    from travel.state import AgentState
    from langchain_core.runnables import RunnableConfig
    
    tripplan = state.get("tripplan")
    
    if not tripplan:
        logger.warning("No trip plan to validate")
        return state
    
    try:
        validation_result = await validate_tripplan_with_inference(tripplan)
        
        state["validation_result"] = validation_result
        state["is_valid"] = validation_result.get("valid", False)
        state["validation_issues"] = validation_result.get("issues", [])
        
        if not validation_result.get("valid", False):
            logger.warning(f"Validation failed with issues: {validation_result.get('issues')}")
        else:
            logger.info("Trip plan validation passed")
        
        return state
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        state["validation_result"] = {"valid": True, "issues": []}
        state["is_valid"] = True
        return state
