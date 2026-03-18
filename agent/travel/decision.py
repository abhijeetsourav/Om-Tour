"""
Decision Node with Regeneration Control

Controls the feedback loop after critic validation.
Routes back to planner for regeneration if validation fails and attempts remain,
otherwise proceeds to formatter.

Implements regeneration counter to prevent infinite loops.
"""

from travel.state import AgentState, MAX_REGEN_ATTEMPTS
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


@traceable
def decision_node(state: AgentState, config: RunnableConfig):
    """
    Decision node that controls routing after critic validation.

    Implements two-part decision logic:
    1. If critic rejected the plan AND regen_attempts < MAX_REGEN_ATTEMPTS:
       - Increment regen_attempts
       - Route back to planner_node for regeneration with critic feedback
    2. Otherwise:
       - Route to formatter_node for output (or error message if max attempts reached)

    Args:
        state: The AgentState containing critic validation result
        config: RunnableConfig for LangGraph execution

    Returns:
        The updated state with incremented regen_attempts if regenerating
    """
    try:
        critic_reject = state.get("critic_reject", False)
        regen_attempts = state.get("regen_attempts", 0)
        critic_feedback = state.get("critic_feedback", {})

        if critic_reject:
            # Critic rejected the plan
            logger.warning(f"❌ Critic rejected itinerary.")
            logger.info(f"Regeneration attempt: {regen_attempts + 1}/{MAX_REGEN_ATTEMPTS}")

            if regen_attempts < MAX_REGEN_ATTEMPTS:
                # Still have regeneration attempts left
                state["regen_attempts"] = regen_attempts + 1
                logger.info(f"Critic feedback: {critic_feedback}")
                logger.info(f"Routing back to planner for regeneration...")
                logger.info(f"Suggestion: {critic_feedback.get('suggestion', 'Please revise your itinerary')}")
                return state
            else:
                # Max attempts reached
                logger.error(f"❌ Max regeneration attempts ({MAX_REGEN_ATTEMPTS}) reached.")
                logger.info("Routing to formatter with error message.")
                state["critic_reject"] = False  # Stop rejection cycle
                return state
        else:
            # Critic accepted the plan
            logger.info("✓ Critic accepted itinerary.")
            logger.info("✓ Resetting regeneration counter.")
            state["regen_attempts"] = 0  # Reset counter after success
            logger.info("Routing to formatter for final output...")
            return state

    except Exception as e:
        logger.error(f"Error in decision_node: {e}", exc_info=True)
        return state
