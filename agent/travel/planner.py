"""
Planner Node

Interprets user requests and generates TripPlan JSON.
Handles LLM-based trip planning and tool invocations.
Does NOT format the output (that's the formatter's job).
"""

from travel.state import AgentState
from travel.search import search_for_places
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

logger = logging.getLogger(__name__)
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# Use a reliable structured-output model by default for planner tasks.
# This model is known for good JSON output, instruction-following, and is supported by the
# current Hugging Face account (chat completions endpoint).
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
)

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"

tools = [search_for_places, select_trip]


def parse_planner_response(llm_output: str):
    """Parse the planner's JSON response into a PlannerResponse.

    The planner is expected to return a single JSON object (possibly embedded in text)
    that matches the PlannerResponse schema (tripplan OR action OR error).
    """
    if not llm_output:
        return None

    # Try to parse clean JSON first
    try:
        payload = json.loads(llm_output)
    except Exception:
        # Try to parse JSON from the start of the string (allows trailing text)
        try:
            decoder = json.JSONDecoder()
            payload, _ = decoder.raw_decode(llm_output)
        except Exception:
            # Fallback: extract first JSON-like chunk
            match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if not match:
                logger.warning("Planner response not valid JSON and no JSON chunk found")
                return None
            try:
                payload = json.loads(match.group(0))
            except Exception as e:
                logger.warning(f"Planner response JSON chunk parse failed: {e}")
                return None

    try:
        response = PlannerResponse.parse_obj(payload)
        return response
    except Exception as e:
        logger.warning(f"Planner response does not match schema: {e}")
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
    Planner node that generates TripPlan or responds to user queries.

    Responsibilities:
    - Interpret user requests
    - Generate TripPlan JSON if trip planning is requested
    - Handle tool invocations (search, trips management)
    - Store TripPlan in state for critic validation
    - Does NOT format the output

    Args:
        state: The AgentState containing user messages
        config: RunnableConfig for LangGraph execution

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
        logger.info("Greeting detected, searching for popular destinations")
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
    trips_json = json.dumps(state.get("trips", []))
    system_message = (
        "You are Sarathi, an AI travel planner for Om Tours.\n\n"
        "You help users plan trips, discover places, and answer travel questions.\n\n"
        "CRITICAL INSTRUCTIONS FOR TRIP PLANNING:\n\n"
        "1. DESTINATION VALIDATION:\n"
        "   - Accept all real, Earth-based destinations (cities, regions, countries, islands, etc.)\n"
        "   - If a destination name appears to have a spelling error but is clearly a real place, CORRECT the spelling and proceed with planning\n"
        "   - Examples: \"lakshdweep\" → Lakshadweep, \"mumbai\" → Mumbai\n"
        "   - ONLY refuse completely fictional/mythical/unrealistic places (Mars, Atlantis, fictional worlds, etc.)\n\n"
        "2. TRIP DURATION AND ACTIVITY LIMITS:\n"
        "   - If the user does not specify the number of days, ASK THE USER TO CLARIFY instead of guessing\n"
        "   - Do NOT assume a default trip duration (e.g., 5 days, 7 days)\n"
        "   - CRITICAL: Reject unrealistic activity requests:\n"
        "     * If user asks for >8 activities per day, REFUSE and explain why\n"
        "     * Example: \"You requested 20 visits per day. This is physically impossible due to travel time, \n"
        "       entrance queues, and distance between attractions. I will create a realistic itinerary with 5-6 \n"
        "       attractions per day instead.\"\n"
        "   - Example response for missing days: \"How many days would you like to spend in {destination}?\"\n\n"
        "3. ACTIVITY DENSITY RULES:\n"
        "   - Relaxed trip: 3-4 activities per day\n"
        "   - Busy itinerary: 5-6 activities per day\n"
        "   - Maximum realistic: 7-8 activities per day (only for very well-connected destinations)\n"
        "   - IMPOSSIBLE: >8 activities per day (physically impossible due to travel time and fatigue)\n"
        "   - If user requests >8/day, politely refuse and generate a realistic itinerary with 5-6/day instead\n\n"
        "4. OUTPUT CONTRACT (STRICT - CRITICAL FOR PARSING):\n"
        "   - Your response MUST start with a JSON object and contain NOTHING else\n"
        "   - NO introductory text, NO explanations, NO markdown code blocks\n"
        "   - The JSON object MUST be the very first character of your response\n"
        "   - Example valid response: {\"tripplan\": {...}}\n"
        "   - Example INVALID: Here is your trip plan: {\"tripplan\": {...}}\n"
        "   - The response MUST conform to the following schema (only these top-level keys):\n"
        "     * \"tripplan\": TripPlan object (see schema below)\n"
        "     * \"action\": {\"type\": ..., \"parameters\": {...}} (for tool calls)\n"
        "     * \"error\": {\"code\": ..., \"message\": ..., \"details\"?: {...}} (for failures)\n"
        "   - Only one of \"tripplan\", \"action\", or \"error\" may be present. Everything else is invalid.\n\n"
        "5. TRIPPLAN (REAL DESTINATIONS):\n"
        "   - Use the TripPlan schema below.\n"
        "   - If any required information is missing (e.g., days), return an \"error\" object asking for clarification.\n"
        "   - Do not return plain text responses for trips.\n\n"
        "6. TOOL ACTIONS:\n"
        "   - If you need external data (search places) or trip CRUD operations, return an \"action\" object.\n"
        "   - Allowed action types:\n"
        "     - \"search_places\" with parameters {\"queries\": [string]}\n"
        "     - \"add_trips\" with parameters {\"trips\": [Trip objects]}\n"
        "     - \"update_trips\" with parameters {\"trips\": [Trip objects]}\n"
        "     - \"delete_trips\" with parameters {\"trip_ids\": [string]}\n"
        "     - \"select_trip\" with parameters {\"trip_id\": string}\n\n"
        "7. ERROR OBJECT:\n"
        "   - Return {\"error\": {\"code\": \"...\", \"message\": \"...\", \"details\": {...}}}\n"
        "   - Use error codes like: \"MISSING_DAYS\", \"UNREALISTIC_REQUEST\", \"INVALID_OUTPUT\", \"NOT_A_TRIP_QUERY\".\n\n"
        "8. CONFIDENCE SCORING:\n"
        "   - Return confidence as a decimal between 0.0 and 1.0.\n"
        "   - Higher confidence for well-known destinations, lower for obscure ones.\n"
        "   - All activity costs should be realistic (not all $0).\n\n"
        "Current trips:\n" + trips_json + "\n\n"
        "TripPlan JSON schema (for REAL destinations only):\n"
        "{\n"
        "    \"city\": string,\n"
        "    \"days\": number,\n"
        "    \"itinerary\": [\n"
        "        {\n"
        "            \"day\": number,\n"
        "            \"activities\": [\n"
        "                {\n"
        "                    \"name\": string,\n"
        "                    \"description\": string,\n"
        "                    \"location\": string,\n"
        "                    \"estimated_cost\": number\n"
        "                }\n"
        "            ]\n"
        "        }\n"
        "    ],\n"
        "    \"estimated_budget\": number,\n"
        "    \"confidence\": number (0.0 to 1.0)\n"
        "}\n"
        "\n"
        "REMINDER: Your response must be pure JSON starting with { and ending with }. No other text.\n"
    )

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

    if not HUGGINGFACE_API_KEY:
        return {
            "messages": [AIMessage(content="Error: HUGGINGFACE_API_KEY not set.")],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

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
            temperature=0.7,
            max_tokens=1024,
        )
        raw_content = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        # Check if it's a model availability issue
        error_str = str(e).lower()
        if "bad request" in error_str or "not found" in error_str:
            logger.warning(f"Model {HUGGINGFACE_MODEL} may not be available. Check API key and model name.")
        return {
            "messages": [AIMessage(content=f"Error generating trip plan. Please try again. (Error: {str(e)[:100]})")],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    logger.info(f"Planner LLM output: {raw_content[:100]}...")

    planner_response = parse_planner_response(raw_content)

    state["messages"] = []
    state["action"] = None

    if not planner_response:
        logger.warning("Planner response did not match expected schema")
        # Fallback: return raw output so the user can still see why the planner failed.
        state["messages"] = [AIMessage(content=raw_content)]
        state["tripplan"] = None
        return {
            "messages": state["messages"],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    if planner_response.error:
        err = planner_response.error
        logger.warning("Planner returned error: %s", err)
        state["messages"] = [AIMessage(content=f"Error ({err.code}): {err.message}")]
        state["tripplan"] = None
        return {
            "messages": state["messages"],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    if planner_response.action:
        # Route to tool nodes based on structured action
        logger.info("Planner returned action: %s", planner_response.action.type)
        state["action"] = planner_response.action.dict()
        state["tripplan"] = None
        state["messages"] = []
        return {
            "messages": state["messages"],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    if planner_response.tripplan:
        tripplan = planner_response.tripplan
        logger.info("TripPlan parsed successfully for: %s", tripplan.city)
        state["tripplan"] = tripplan
        state["messages"] = []
        return {
            "messages": state["messages"],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    logger.warning("Planner response did not include tripplan, action, or error")
    state["messages"] = [AIMessage(content="Error: Planner response missing required fields.")]
    state["tripplan"] = None
    return {
        "messages": state["messages"],
        "selected_trip_id": state.get("selected_trip_id"),
        "trips": state.get("trips", []),
        "tripplan": state.get("tripplan"),
    }
