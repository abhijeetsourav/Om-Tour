"""
Query Validator Node

Validates whether the user's request is a reasonable, feasible travel query.
Rejects unrealistic or impossible destinations early in the pipeline.
"""

from travel.state import AgentState
from travel.critic import evaluate_query_reasonableness
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


@traceable
async def query_validator_node(state: AgentState, config: RunnableConfig):
    """
    Validates the user's travel query for feasibility.

    Only validates trip requests. Passes through non-trip queries (greetings, Q&A, etc.)

    For trip requests, checks:
    - Is the destination realistic and on Earth?
    - Is it accessible for tourism?
    - Are there major safety/security concerns?

    If validation fails on a trip request, returns rejection and terminates the pipeline.
    If valid or non-trip query, allows the planner to proceed.

    Args:
        state: The AgentState containing user messages
        config: RunnableConfig for LangGraph execution

    Returns:
        Updated AgentState with validation result
    """
    try:
        # Extract the original user query
        user_query = ""
        for msg in state.get("messages", []):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break
            elif isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            logger.warning("No user query found in messages")
            state["critic_reject"] = False
            return state

        # Check if this is a trip request
        trip_keywords = ["trip", "travel", "plan", "itinerary", "journey", "vacation", "tour", "explore"]
        is_trip_request = any(keyword in user_query.lower() for keyword in trip_keywords)

        if not is_trip_request:
            logger.info("Non-trip query detected, skipping validation: %s", user_query[:60])
            # Pass through non-trip queries (greetings, Q&A, general questions)
            return state

        logger.info("Trip request detected, validating feasibility for: %s", user_query[:60])

        # Run query reasonableness check only for trip requests
        validation_result = await evaluate_query_reasonableness(user_query)
        reasonable = validation_result.get("reasonable", True)
        reason = validation_result.get("reason", "")

        if not reasonable:
            logger.warning("Trip query validation failed: %s", reason)
            # Return rejection message and stop processing
            state["messages"] = [AIMessage(content=reason)]
            state["critic_reject"] = False  # Not a critic rejection, just a pre-validation failure
            state["tripplan"] = None
            return state

        logger.info("Trip query validation passed: %s", reason)
        # Continue to planner
        return state

    except Exception as e:
        logger.error(f"Error in query_validator_node: {e}", exc_info=True)
        # On error, allow the query to proceed (graceful degradation)
        return state
