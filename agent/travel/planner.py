"""
Planner Node

Interprets user requests and generates TripPlan JSON.
Handles LLM-based trip planning and tool invocations.
Does NOT format the output (that's the formatter's job).

Uses BaseNode pattern for consistent error handling across all nodes.
"""

from travel.state import AgentState
from travel.search import search_for_places
from travel.base_node import BaseNode
from schemas.trip_schema import TripPlan, PlannerResponse, PlannerError, PlannerAction
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import os
from langsmith import traceable
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# Qwen2.5-7B-Instruct: Optimized for structured output and instruction-following
# Excellent at generating JSON output with clear formatting
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)


@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"


tools = [search_for_places, select_trip]


def fix_incomplete_tripplan(payload: dict) -> dict:
    """
    POST-PROCESSING: Fill in missing required fields in tripplan JSON.

    The LLM sometimes omits required fields like estimated_budget, confidence, 
    location (for activities), and estimated_cost (for activities).

    This function adds reasonable defaults so schema validation passes.
    """
    if not payload.get("tripplan"):
        return payload

    tripplan = payload["tripplan"]

    # Fill missing tripplan-level fields
    if "estimated_budget" not in tripplan:
        # Estimate budget: $1000 + $500 per day for average destination
        days = tripplan.get("days", 3)
        tripplan["estimated_budget"] = 1000 + (days * 500)
        logger.info(f"Added estimated_budget: ${tripplan['estimated_budget']}")

    if "confidence" not in tripplan:
        # Default to 0.85 - good confidence for recognized destinations
        tripplan["confidence"] = 0.85
        logger.info(f"Added confidence: {tripplan['confidence']}")

    # Fill missing activity-level fields
    for day_plan in tripplan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if "location" not in activity:
                # Use city as fallback location
                city = tripplan.get("city", "Unknown")
                activity["location"] = city
                logger.info(
                    f"Added location for activity '{activity.get('name', 'Unknown')}': {city}")

            if "estimated_cost" not in activity:
                # Default to $50 per activity (reasonable for most attractions)
                activity["estimated_cost"] = 50
                logger.info(
                    f"Added estimated_cost for activity '{activity.get('name', 'Unknown')}': $50")

    return payload


def parse_planner_response(llm_output: str):
    """Parse the planner's JSON response into a PlannerResponse.

    The planner is expected to return a single JSON object (possibly embedded in text)
    that matches the PlannerResponse schema (tripplan OR action OR error).

    Includes fallback handling for formatted text responses.
    """
    if not llm_output:
        logger.error("Empty LLM output - cannot parse")
        return None

    logger.debug(
        f"Parsing LLM output (length={len(llm_output)}): {llm_output[:100]}...")

    # Try to parse clean JSON first
    try:
        payload = json.loads(llm_output)
        logger.debug("Successfully parsed as clean JSON")
    except Exception as e1:
        logger.debug(f"Clean JSON parse failed: {e1}")
        # Try to parse JSON from the start of the string (allows trailing text)
        try:
            decoder = json.JSONDecoder()
            payload, _ = decoder.raw_decode(llm_output)
            logger.debug("Successfully parsed with raw_decode")
        except Exception as e2:
            logger.debug(f"raw_decode failed: {e2}")
            # Fallback: extract first JSON-like chunk
            match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if not match:
                logger.error(f"No JSON found in output: {llm_output[:200]}")
                # Last resort: check if this is formatted text (has emoji or "Day")
                if ("🌍" in llm_output or "Day 1" in llm_output or "Day 2" in llm_output):
                    logger.warning(
                        "Detected formatted text response instead of JSON - this violates prompt")
                return None
            try:
                payload = json.loads(match.group(0))
                logger.debug("Successfully parsed extracted JSON chunk")
            except Exception as e3:
                logger.error(
                    f"Failed to parse JSON chunk: {e3}, chunk: {match.group(0)[:100]}")
                return None

    # POST-PROCESSING: Fix any incomplete tripplan data before validation
    if payload.get("tripplan"):
        payload = fix_incomplete_tripplan(payload)

    try:
        response = PlannerResponse.parse_obj(payload)
        logger.info(f"Successfully created PlannerResponse: {response}")
        return response
    except Exception as e:
        logger.error(f"Payload does not match schema: {e}, payload: {payload}")
        return None


def is_greeting(text: str) -> bool:
    """Detect if the user message is a greeting."""
    greetings = ["hi", "hello", "hey", "good morning",
                 "good afternoon", "good evening"]
    text_lower = text.strip().lower()
    for greeting in greetings:
        if text_lower == greeting or text_lower.startswith(greeting + " ") or text_lower.startswith(greeting + ","):
            return True
    return False


def format_greeting_response(places: list[dict]) -> str:
    """Format search results into a friendly greeting response."""
    response = "👋 Hello! Welcome to Om Tours!\n\n"
    response += "I'm Sarathi, your AI travel planner. Here are some popular tourist destinations to inspire your next trip:\n\n"

    for i, place in enumerate(places[:5], 1):
        name = place.get("name", "Unknown")
        rating = place.get("rating", "N/A")
        address = place.get("address", "")

        response += f"{i}. **{name}**\n"
        if rating != "N/A":
            response += f"   ⭐ Rating: {rating}\n"
        if address:
            response += f"   📍 {address}\n"
        response += "\n"

    response += "Would you like to plan a trip to any of these destinations?"
    return response


@traceable
async def planner_node(state: AgentState, config: RunnableConfig):
    """
    LangGraph node wrapper for the planner.

    This function is called by the graph and delegates to PlannerNode.execute()
    which handles all error handling and recovery automatically.
    """
    planner = PlannerNode()
    return await planner.execute(state, config)


class PlannerNode(BaseNode):
    """Planner node that generates TripPlan or responds to user queries.

    Responsibilities:
    - Interpret user requests
    - Generate TripPlan JSON if trip planning is requested
    - Handle tool invocations (search, trips management)
    - Store TripPlan in state for critic validation
    - Does NOT format the output
    """

    NODE_NAME = "Planner"

    async def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Core planner logic.

        Args:
            state: The AgentState containing user messages

        Returns:
            Updated AgentState with tripplan and/or messages
        """

        # Extract the last user message
        user_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content"):
                if isinstance(msg, HumanMessage) or (not isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage)):
                    user_message = msg.content
                    break

        if not user_message:
            logger.warning("No user message found")
            return state

        # Check if the message is a greeting
        if is_greeting(user_message):
            logger.info(
                "Greeting detected, searching for popular destinations")
            try:
                import googlemaps
                GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

                if GOOGLE_MAPS_API_KEY:
                    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
                    query = "popular tourist destinations in India"
                    places = []
                    response = gmaps.places(query)

                    for result in response.get("results", []):
                        place = {
                            "id": result.get("place_id", f"{result.get('name', '')}"),
                            "name": result.get("name", ""),
                            "address": result.get("formatted_address", ""),
                            "latitude": result.get("geometry", {}).get("location", {}).get("lat", 0),
                            "longitude": result.get("geometry", {}).get("location", {}).get("lng", 0),
                            "rating": result.get("rating", 0),
                            "user_ratings_total": result.get("user_ratings_total", 0),
                        }
                        places.append(place)

                    filtered = []
                    for p in places:
                        rating = p.get("rating", 0)
                        reviews = p.get("user_ratings_total", 0)
                        if rating >= 4.0 and reviews >= 50:
                            filtered.append(p)

                    seen = set()
                    unique_places = []
                    for p in filtered:
                        name = p.get("name")
                        if name not in seen:
                            seen.add(name)
                            unique_places.append(p)

                    ranked = sorted(
                        unique_places,
                        key=lambda x: x.get("rating", 0),
                        reverse=True
                    )[:5]

                    greeting_response = format_greeting_response(ranked)

                    return {
                        "messages": [AIMessage(content=greeting_response)],
                        "selected_trip_id": state.get("selected_trip_id"),
                        "trips": state.get("trips", []),
                        "tripplan": state.get("tripplan"),
                    }
            except Exception as e:
                logger.warning(f"Error searching for places on greeting: {e}")
                fallback_response = "👋 Hello! Welcome to Om Tours! I'm Sarathi, your AI travel planner. How can I help you plan your next adventure?"
                return {
                    "messages": [AIMessage(content=fallback_response)],
                    "selected_trip_id": state.get("selected_trip_id"),
                    "trips": state.get("trips", []),
                    "tripplan": state.get("tripplan"),
                }

        # Build planner prompt
        regen_attempts = state.get("regen_attempts", 0)
        critic_feedback = state.get("critic_feedback", {})

        # Base system message (strict output contract)
        # Optimized for Qwen2.5-7B-Instruct with Few-Shot Examples
        trips_json = json.dumps(state.get("trips", []))
        system_message = f"""SYSTEM INSTRUCTION - READ CAREFULLY:

You are Sarathi, a travel planning expert. Your ONLY job is to respond with JSON.

<CRITICAL_RULE>
RESPOND ONLY WITH RAW JSON. NOTHING ELSE.
- NO human text before or after JSON
- NO emoji (🌍, 📍, ⭐, etc.)
- NO "Day 1", "Day 2" formatting
- NO bullet points (•)
- NO markdown or special characters
- Your response must START with {{ and END with }}
- Every single character outside {{ }} will cause failure
</CRITICAL_RULE>

<FORBIDDEN_RESPONSES>
WRONG: 🌍 Trip Plan: Tokyo (2 days)
WRONG: Day 1 • Visit...
WRONG: Error (INVALID_DESTINATION): Heaven...
WRONG: <html>...
WRONG: ```json ... ```
WRONG: "Here's your trip:"
ALL OF THESE WILL FAIL AND BE REJECTED
</FORBIDDEN_RESPONSES>

<ONLY_VALID_RESPONSES>
CORRECT: {{"tripplan": {{"city": "Tokyo", "days": 2, ...}}}}
CORRECT: {{"error": {{"code": "INVALID_DESTINATION", "message": "..."}}}}
CORRECT: {{"error": {{"code": "MISSING_INFO", ...}}}}
ONLY JSON OBJECTS STARTING WITH {{ AND ENDING WITH }} ARE VALID
</ONLY_VALID_RESPONSES>

<REQUIRED_JSON_SCHEMA>
Choose ONE response type (nothing else):

1. SUCCESS RESPONSE:
{{
  "tripplan": {{
    "city": "destination",
    "days": number,
    "itinerary": [
      {{
        "day": 1,
        "activities": [
          {{
            "name": "Activity",
            "description": "Description",
            "location": "Address",
            "estimated_cost": number
          }}
        ]
      }}
    ],
    "estimated_budget": number,
    "confidence": 0.95,
    "confidence_breakdown": {{
      "destination_certainty": 1.0,
      "itinerary_realism": 0.95,
      "pricing_accuracy": 0.9,
      "feasibility": 0.95
    }}
  }}
}}

2. MISSING INFO RESPONSE:
{{
  "error": {{
    "code": "MISSING_INFO",
    "message": "I need X to plan your trip",
    "clarification_questions": ["Q1?", "Q2?"]
  }}
}}

3. INVALID DESTINATION RESPONSE:
{{
  "error": {{
    "code": "INVALID_DESTINATION",
    "message": "Destination is fictional/inaccessible"
  }}
}}
</REQUIRED_JSON_SCHEMA>

<EXAMPLES_JSON_ONLY>
Example 1 - User said "3 days in Paris":
{{"tripplan": {{"city": "Paris", "days": 3, "itinerary": [{{"day": 1, "activities": [{{"name": "Eiffel Tower", "description": "Iconic landmark", "location": "Paris France", "estimated_cost": 25}}]}}], "estimated_budget": 1200, "confidence": 0.95, "confidence_breakdown": {{"destination_certainty": 1.0, "itinerary_realism": 0.9, "pricing_accuracy": 0.9, "feasibility": 0.95}}}}}}

Example 2 - User said "Trip to Atlantis":
{{"error": {{"code": "INVALID_DESTINATION", "message": "Atlantis is legendary and not accessible for tourism"}}}}

Example 3 - User said "Trip to Tokyo":
{{"error": {{"code": "MISSING_INFO", "message": "I need duration to plan your Tokyo trip", "clarification_questions": ["How many days?", "What budget?"]}}}}
</EXAMPLES_JSON_ONLY>

<INSTRUCTIONS>
- You MUST respond with valid JSON only
- JSON must start with {{ and end with }}
- All activities must have real locations
- Prices must be realistic in USD
- Confidence scores 0.8-1.0
- No text outside JSON
- Your entire response is ONE JSON object only
</INSTRUCTIONS>

Trips on file: {trips_json}
"""

        # Add regeneration feedback if this is a retry
        if regen_attempts > 0 and critic_feedback:
            system_message += f"""

REGENERATION ATTEMPT #{regen_attempts}:

Your previous itinerary had the following issues:

Issues: {", ".join(critic_feedback.get("issues", []))}

Suggestion: {critic_feedback.get("suggestion", "Please revise your itinerary to improve its quality")}

Please generate a corrected itinerary that addresses these issues. Focus on:
- Removing problematic activities or locations
- Better balancing activities across days
- Ensuring all locations are real and verified
- Creating a more realistic and enjoyable travel experience
"""

        if regen_attempts > 0:
            logger.info(f"Planner regeneration attempt #{regen_attempts}")
            logger.info(f"Critic feedback: {critic_feedback}")

        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
        if not HUGGINGFACE_API_KEY:
            return {
                "messages": [AIMessage(content="Error: HUGGINGFACE_API_KEY not set.")],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": state.get("tripplan"),
            }

        client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
        HUGGINGFACE_MODEL = os.getenv(
            "HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
        )

        messages = [{"role": "system", "content": system_message}]

        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                messages.append({"role": "tool", "content": msg.content})
            elif hasattr(msg, "content"):
                messages.append({"role": "user", "content": msg.content})

        try:
            completion = client.chat.completions.create(
                model=HUGGINGFACE_MODEL,
                messages=messages,
                temperature=0.1,  # REDUCED from 0.3 to 0.1 for strict JSON output adherence
                max_tokens=2048,
            )
            raw_content = completion.choices[0].message.content
            logger.info(f"Raw LLM output: {raw_content}")
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            # Check if it's a model availability issue
            error_str = str(e).lower()
            if "bad request" in error_str or "not found" in error_str:
                logger.warning(
                    f"Model {HUGGINGFACE_MODEL} may not be available. Check API key and model name.")
            return {
                "messages": [AIMessage(content=f"Error generating trip plan. Please try again. (Error: {str(e)[:100]})")],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": state.get("tripplan"),
            }

        logger.info(f"Planner LLM output: {raw_content[:100]}...")

        # FALLBACK: Detect if LLM returned formatted error text instead of JSON
        # Pattern: "Error (CODE): message"
        formatted_error_pattern = r"^Error\s*\((\w+)\):\s*(.+)$"
        error_match = re.match(formatted_error_pattern, raw_content.strip())
        if error_match:
            code, message = error_match.groups()
            logger.warning(
                f"FALLBACK: Detected formatted error text, converting to JSON: {code}")
            raw_content = json.dumps(
                {"error": {"code": code, "message": message}})
            logger.info(f"Converted to JSON: {raw_content}")

        planner_response = parse_planner_response(raw_content)

        state["messages"] = []
        state["action"] = None

        if not planner_response:
            logger.error("Planner response parsing failed")
            logger.error(f"Raw output: {raw_content[:300]}")
            # Better error message for debugging
            error_msg = (
                "❌ Could not generate trip plan. The model may not have responded correctly.\n\n"
                "Please try again with a clear request like:\n"
                '- "Plan a 5-day trip to Delhi"\n'
                '- "I want to go to London for 3 days"'
            )
            state["messages"] = [AIMessage(content=error_msg)]
            state["tripplan"] = None
            return {
                "messages": state["messages"],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": None,
            }

        if planner_response.error:
            err = planner_response.error
            logger.warning("Planner returned error: %s", err)
            state["messages"] = [
                AIMessage(content=f"Error ({err.code}): {err.message}")]
            state["tripplan"] = None
            return {
                "messages": state["messages"],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": None,  # No tripplan on error
            }

        if planner_response.action:
            # Route to tool nodes based on structured action
            logger.info("Planner returned action: %s",
                        planner_response.action.type)
            state["action"] = planner_response.action.dict()
            state["tripplan"] = None
            state["messages"] = []  # Clear messages to avoid JSON leakage
            return {
                "messages": state["messages"],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": None,  # No tripplan for actions
            }

        if planner_response.tripplan:
            tripplan = planner_response.tripplan
            logger.info("TripPlan parsed successfully for: %s", tripplan.city)
            state["tripplan"] = tripplan
            state["messages"] = []
            # Return to critic - tripplan is needed there
            return {
                "messages": state["messages"],
                "selected_trip_id": state.get("selected_trip_id"),
                "trips": state.get("trips", []),
                "tripplan": tripplan,  # Return the tripplan for critic validation
            }

        logger.warning(
            "Planner response did not include tripplan, action, or error")
        state["messages"] = [
            AIMessage(content="Error: Planner response missing required fields.")]
        state["tripplan"] = None
        return {
            "messages": state["messages"],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": None,  # No tripplan to return
        }

    def _format_error_message(self, error: Exception) -> str:
        """Planner-specific error messages."""
        error_str = str(error).lower()

        if "fictional" in error_str or "invalid destination" in error_str:
            return "The destination doesn't appear to exist. Could you clarify where you'd like to go?"
        elif "missing" in error_str or "invalid input" in error_str:
            return "I need more details (city name, duration, or dates) to plan your trip."
        elif "api" in error_str or "key" in error_str:
            return "Connection issue with the planning service. Please try again shortly."

        return super()._format_error_message(error)

    def _cleanup_state_on_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Planner-specific cleanup."""
        state = super()._cleanup_state_on_error(state)
        state["tripplan"] = None  # Explicitly clear failed plan
        state["action"] = None
        state["messages"] = []
        return state
