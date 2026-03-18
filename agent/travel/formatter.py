"""
Formatter Node

Converts structured TripPlan into human-readable formatted output.
This is the final stage before returning to the user.
"""

from travel.state import AgentState
from travel.format_trip import format_trip_for_chat
from schemas.trip_schema import TripPlan
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


@traceable
async def formatter_node(state: AgentState, config: RunnableConfig):
    """
    Formats validated TripPlan into human-readable output.

    Takes the validated TripPlan from state and converts it to
    a formatted message ready for the user.

    Also resets regeneration counter upon successful formatting,
    indicating that the trip plan was accepted.

    CRITICAL: Never leak raw JSON to the UI. All output must be formatted text.

    Args:
        state: The AgentState containing validated TripPlan
        config: RunnableConfig for LangGraph execution

    Returns:
        Updated AgentState with formatted message in messages and reset regen_attempts
    """
    try:
        tripplan = state.get("tripplan")

        if not tripplan:
            # Check if there's already an error message in messages
            messages = state.get("messages", [])
            if messages and any("❌" in str(msg.content) or "Error" in str(msg.content) for msg in messages):
                logger.info("No tripplan but error message already present - pass through")
                return state
            
            logger.warning("No TripPlan to format")
            state["messages"] = [AIMessage(content="No trip plan available to format.")]
            return state

        # Ensure tripplan is a TripPlan object
        if isinstance(tripplan, dict):
            tripplan = TripPlan(**tripplan)
            state["tripplan"] = tripplan

        logger.info("Formatting TripPlan for city: %s", tripplan.city)

        # Format the trip for display
        formatted = format_trip_for_chat(tripplan)
        
        # GUARD: Ensure no JSON leakage to UI
        if "{" in formatted and "itinerary" in formatted.lower():
            logger.error("❌ JSON LEAKED TO OUTPUT! Formatter produced JSON instead of text.")
            logger.error(f"Output preview: {formatted[:200]}")
            state["messages"] = [
                AIMessage(content=f"🌍 Trip Plan: {tripplan.city} ({tripplan.days} days)\n\n"
                                + "\n".join([
                                    f"Day {day.day}: " + ", ".join([a.name for a in day.activities])
                                    for day in tripplan.itinerary
                                ]))
            ]
            return state

        logger.info("Trip formatted successfully")
        logger.debug(f"Output length: {len(formatted)} characters")

        # Reset regeneration counter upon successful acceptance
        regen_attempts = state.get("regen_attempts", 0)
        if regen_attempts > 0:
            logger.info(f"✓ Itinerary accepted after {regen_attempts} regeneration attempt(s). Resetting counter.")
        state["regen_attempts"] = 0

        # Return formatted message
        state["messages"] = [AIMessage(content=formatted)]
        
        # CRITICAL: Clear tripplan from state to prevent JSON serialization in API response
        # The formatted text is in messages; we don't need the structured tripplan anymore
        state["tripplan"] = None

        return state

    except Exception as e:
        logger.error(f"Error in formatter_node: {e}", exc_info=True)
        state["messages"] = [AIMessage(content="Error formatting trip plan. Please try again.")]
        state["tripplan"] = None  # Clear to prevent JSON leakage
        return state
