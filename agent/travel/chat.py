from travel.state import AgentState
from travel.search import search_for_places
from schemas.trip_schema import TripPlan
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import json
import os
import re
from langsmith import traceable
from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL", "MiniMaxAI/MiniMax-M2.5:fireworks-ai"
)

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)


@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"


tools = [search_for_places, select_trip]


def extract_json_block(text: str):
    """
    Extract JSON block from model output.
    Handles cases where model adds text before/after JSON.
    """

    if not text:
        return None

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    return match.group(0)


def parse_tripplan_from_llm_output(llm_output: str):
    """
    Parse LLM output into TripPlan schema if possible.
    Includes confidence normalization and retry logic.
    """

    from schemas.trip_schema import TripPlan

    json_block = extract_json_block(llm_output)

    if not json_block:
        logger.warning("No JSON block found in LLM output.")
        return None

    def repair_json_string(s):
        # Remove trailing commas
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # Add quotes around keys if missing (simple heuristic)
        s = re.sub(r"([^{\[\s]+):", r'"\1":', s)
        # Convert confidence like 90% to 0.9
        s = re.sub(r'"confidence"\s*:\s*"(\d+)%"',
                   lambda m: '"confidence": {}'.format(float(m.group(1))/100), s)
        return s

    def normalize_confidence(data: dict):
        """Ensure confidence is a float between 0 and 1."""
        conf = data.get("confidence", 0.7)

        # Handle string values (e.g., "90%" or "0.9")
        if isinstance(conf, str):
            conf = conf.replace("%", "").strip()

        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.7

        # If > 1, assume it's a percentage and convert to decimal
        if conf > 1:
            conf = conf / 100

        # Clamp to [0, 1] range
        conf = max(0.0, min(1.0, conf))
        data["confidence"] = conf

    try:
        data = json.loads(json_block)
    except Exception as e:
        logger.warning(f"Initial JSON parse failed: {e}. Attempting repair.")
        try:
            repaired = repair_json_string(json_block)
            data = json.loads(repaired)
        except Exception as e2:
            logger.warning(f"Repair JSON parse failed: {e2}")
            logger.debug(f"Invalid JSON block: {json_block}")
            return None

    # Normalize confidence field
    normalize_confidence(data)

    try:
        tripplan = TripPlan(**data)
        return tripplan
    except Exception as e:
        logger.warning(f"TripPlan instantiation failed: {e}")
        logger.debug(f"Sanitized JSON: {data}")
        return None


def is_greeting(text: str) -> bool:
    """
    Detect if the user message is a greeting.
    """
    greetings = ["hi", "hello", "hey", "good morning",
                 "good afternoon", "good evening"]
    text_lower = text.strip().lower()

    # Check if the message is exactly a greeting or starts with one
    for greeting in greetings:
        if text_lower == greeting or text_lower.startswith(greeting + " ") or text_lower.startswith(greeting + ","):
            return True

    return False


def format_greeting_response(places: list[dict]) -> str:
    """
    Format search results into a friendly greeting response.
    """
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
async def chat_node(state: AgentState, config: RunnableConfig):

    # Extract the last user message
    user_message = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and not isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
            user_message = msg.content
            break

    # Check if the message is a greeting
    if user_message and is_greeting(user_message):
        logger.info("Greeting detected, searching for popular destinations")

        # Import googlemaps here to get access to the API
        import googlemaps
        GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

        if GOOGLE_MAPS_API_KEY:
            try:
                gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

                # Search for popular tourist destinations
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

                # Filter places (rating >= 4.0 and reviews >= 50)
                filtered = []
                for p in places:
                    rating = p.get("rating", 0)
                    reviews = p.get("user_ratings_total", 0)
                    if rating >= 4.0 and reviews >= 50:
                        filtered.append(p)

                # Deduplicate and rank
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

                # Format the response
                greeting_response = format_greeting_response(ranked)

                return {
                    "messages": [AIMessage(content=greeting_response)],
                    "selected_trip_id": state.get("selected_trip_id"),
                    "trips": state.get("trips", []),
                    "tripplan": state.get("tripplan"),
                }

            except Exception as e:
                logger.warning(f"Error searching for places on greeting: {e}")
                # Fallback to generic greeting if search fails
                fallback_response = "👋 Hello! Welcome to Om Tours! I'm Sarathi, your AI travel planner. How can I help you plan your next adventure?"
                return {
                    "messages": [AIMessage(content=fallback_response)],
                    "selected_trip_id": state.get("selected_trip_id"),
                    "trips": state.get("trips", []),
                    "tripplan": state.get("tripplan"),
                }

    # Continue with normal chat flow for non-greeting messages
    system_message = f"""
You are Sarathi, an AI travel planner for Om Tours.

You help users plan trips, discover places, and answer travel questions.

CRITICAL INSTRUCTIONS FOR TRIP PLANNING:

1. DESTINATION VALIDATION:
   - Accept all real, Earth-based destinations (cities, regions, countries, islands, etc.)
   - If a destination name appears to have a spelling error but is clearly a real place, CORRECT the spelling and proceed with planning
   - Examples: "lakshdweep" → Lakshadweep, "mumbai" → Mumbai
   - ONLY refuse completely fictional/mythial/unrealistic places (Mars, Atlantis, fictional worlds, etc.)

2. TRIP DURATION:
   - If the user does not specify the number of days, ASK THE USER TO CLARIFY instead of guessing
   - Do NOT assume a default trip duration (e.g., 5 days, 7 days)
   - Example response: "How many days would you like to spend in {{destination}}?"

3. TRIP PLAN FORMAT:
   - When planning a trip to a REAL destination with all required details: Return ONLY valid JSON matching the schema below
   - When refusing a completely unrealistic destination: Return a brief, friendly text refusal - NO JSON
   - Do NOT provide long explanations, suggestions, or alternatives in either case

4. CONFIDENCE SCORING:
   - Return confidence as a decimal between 0.0 and 1.0 (e.g., 0.85 for 85%)
   - Higher confidence for well-known destinations, lower for obscure places

Current trips:
{json.dumps(state.get('trips', []))}

TripPlan JSON schema (for REAL destinations only):
{{
    "city": string,
    "days": number,
    "itinerary": [
        {{
            "day": number,
            "activities": [
                {{
                    "name": string,
                    "description": string,
                    "location": string,
                    "estimated_cost": number
                }}
            ]
        }}
    ],
    "estimated_budget": number,
    "confidence": number (0.0 to 1.0)
}}

RESPONSE FORMAT:
- Real destination request with complete details → JSON trip plan only
- Missing details (like number of days) → Ask for clarification as text
- Fictional destination request → Brief text refusal only (no JSON)
- General travel question → Normal text response
"""

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
        )
        raw_content = completion.choices[0].message.content
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", []),
            "tripplan": state.get("tripplan"),
        }

    logger.info(f"LLM raw output: {raw_content}")

    # Step 2: Extract JSON and parse TripPlan
    json_block = extract_json_block(raw_content)
    tripplan = parse_tripplan_from_llm_output(raw_content)

    # Detect if user is asking for a trip
    is_trip_query = any(keyword in user_message.lower() for keyword in
                        ["trip", "travel", "plan", "itinerary", "journey", "vacation", "tour"])

    state["messages"] = []  # Ensure only one message is returned

    # NOTE: chat_node should NOT format the trip
    # Formatting only happens in critic_node after validation
    if tripplan:
        # Store TripPlan in state for critic validation
        state["tripplan"] = tripplan
        # Return empty messages - critic will format if validation passes
        state["messages"] = []
    elif is_trip_query and not tripplan:
        # User asked for a trip but no JSON was generated
        # Check if LLM intentionally refused (substantial text response) vs parsing error

        if raw_content and len(raw_content.strip()) > 100 and not json_block:
            # LLM provided a substantial non-JSON response (likely a refusal with explanation)
            logger.info(
                "Trip query: LLM provided refusal/explanation. Passing to critic for review.")
            state["tripplan"] = None
            # Keep the refusal message
            state["messages"] = [AIMessage(content=raw_content)]
        else:
            # Likely a parsing/formatting error - retry with stricter prompt
            logger.info(
                "Trip query detected but parsing failed. Retrying with stricter JSON prompt.")

            retry_prompt = f"""Extract ONLY a valid JSON TripPlan from the previous response.
                Return ONLY valid JSON matching this schema, no explanations:

                {{
                    "city": string,
                    "days": number,
                    "itinerary": [{{"day": number, "activities": [{{"name": string, "description": string, "location": string, "estimated_cost": number}}]}}],
                    "estimated_budget": number,
                    "confidence": number (0.0 to 1.0)
                }}
                """

            retry_messages = messages.copy()
            retry_messages.append({"role": "user", "content": retry_prompt})

            try:
                retry_completion = client.chat.completions.create(
                    model=HUGGINGFACE_MODEL,
                    messages=retry_messages,
                )
                retry_content = retry_completion.choices[0].message.content
                retry_tripplan = parse_tripplan_from_llm_output(retry_content)

                if retry_tripplan:
                    state["tripplan"] = retry_tripplan
                    # Don't format here - let critic handle it
                    state["messages"] = []
                else:
                    state["tripplan"] = None
                    state["messages"] = [
                        AIMessage(
                            content="⚠️ I couldn't generate a structured trip plan. Please try rephrasing your request with more details (e.g., destination, duration, budget).")
                    ]
            except Exception as e:
                logger.warning(f"Retry parsing failed: {e}")
                state["tripplan"] = None
                state["messages"] = [
                    AIMessage(
                        content="⚠️ I couldn't generate a structured trip plan. Please try rephrasing your request.")
                ]
    elif json_block and not is_trip_query:
        # If JSON exists but not a trip query, try fallback
        try:
            data = json.loads(json_block)
            tripplan_fallback = TripPlan(**data)
            state["tripplan"] = tripplan_fallback
            state["messages"] = []  # Don't format here - let critic handle it
        except Exception:
            state["tripplan"] = None
            state["messages"] = [AIMessage(content=raw_content)]
    else:
        # No JSON found and not a trip query - return narrative response
        state["tripplan"] = None
        state["messages"] = [AIMessage(content=raw_content)]

    # Step 5: Ensure only one message is returned
    if len(state["messages"]) > 1:
        state["messages"] = [state["messages"][-1]]

    return {
        "messages": state["messages"],
        "selected_trip_id": state.get("selected_trip_id"),
        "trips": state.get("trips", []),
        "tripplan": state.get("tripplan"),
    }
