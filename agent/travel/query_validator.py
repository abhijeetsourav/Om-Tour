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
import re

logger = logging.getLogger(__name__)


def classify_intent(query: str) -> str:
    """
    Multi-layer intent classification: keyword + pattern + heuristic
    Returns: "trip_request" | "general_question" | "ambiguous"

    Layer 1: Keyword matching (fastest)
    Layer 2: Pattern/action matching
    Layer 3: Semantic heuristics
    """
    query_lower = query.lower().strip()

    # Layer 1: Trip-related keywords
    trip_keywords = [
        "trip", "travel", "plan", "itinerary", "journey", "vacation",
        "tour", "getaway", "escape", "adventure", "explore", "visit",
        "destination", "itinerary planning"
    ]

    # Layer 2: Action/intent patterns (regex)
    intent_patterns = [
        r"(take|book|plan|arrange|organize|create)\s+(me\s+)?a\s+(trip|tour|vacation|journey)",
        r"(i\s+)?want\s+to\s+(visit|go\s+to|explore|discover|travel\s+to)",
        r"(i\s+)?need\s+help\s+(planning|organizing|arranging)\s+",
        r"(days?|weeks?|nights?)\s+(in|at|to)",
        r"(hotel|flight|accommodation|stay)\s+(in|at)",
        r"(things?\s+to\s+do|attractions?|activities?|places?\s+to\s+visit)",
    ]

    # Layer 3: Negation handling
    negations = ["not", "don't", "don't want", "avoid", "exclude", "without"]
    has_negation = any(neg in query_lower for neg in negations)

    # Score calculation
    keyword_score = sum(1 for kw in trip_keywords if kw in query_lower)
    pattern_score = sum(
        1 for pattern in intent_patterns if re.search(pattern, query_lower))

    # High confidence: strong keyword or pattern match
    if keyword_score >= 2 or pattern_score >= 1:
        if not has_negation:
            return "trip_request"
        elif pattern_score >= 2:  # Strong pattern overrides negation
            return "trip_request"

    # Low confidence: very short, generic
    if len(query_lower) < 5:
        return "general_question"

    # Medium confidence: ambiguous, let LLM decide
    if keyword_score == 1 or pattern_score == 1:
        return "ambiguous"

    return "general_question"


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

        # Multi-layer intent classification
        intent = classify_intent(user_query)
        logger.info(
            f"Intent classification: {intent} | Query: {user_query[:60]}")

        if intent == "general_question":
            logger.info(
                "General question detected (non-trip), skipping validation")
            # Pass through non-trip queries
            return state
        elif intent == "ambiguous":
            logger.info("Ambiguous query, using LLM for semantic validation")
            # LLM will make the final call

        # Validate trip requests (trip_request) or ambiguous queries
        logger.info("Validating query feasibility for: %s", user_query[:60])
        validation_result = await evaluate_query_reasonableness(user_query)
        reasonable = validation_result.get("reasonable", True)
        reason = validation_result.get("reason", "")

        if not reasonable:
            logger.warning("Trip query validation failed: %s", reason)
            state["messages"] = [AIMessage(content=reason)]
            state["critic_reject"] = False
            state["tripplan"] = None
            return state

        logger.info("Query validation passed: %s", reason)
        return state

    except Exception as e:
        logger.error(f"Error in query_validator_node: {e}", exc_info=True)
        # Graceful degradation - allow query to proceed
        return state
