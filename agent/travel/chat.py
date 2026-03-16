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
    if "confidence" in data:
        conf = data["confidence"]
        if isinstance(conf, str) and "%" in conf:
            conf = float(conf.replace("%", "")) / 100
        data["confidence"] = conf
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

Rules:
- Ask for location if missing.
- Use search_for_places when recommending places.
- Prefer structured trip plans when the user asks to plan a trip.
- If the user asks a general travel question, answer normally.

Current trips:
{json.dumps(state.get('trips', []))}

TripPlan JSON schema (only when planning full trips):

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
    "confidence": number  # IMPORTANT: Return confidence as a decimal between 0 and 1 (e.g., 0.92 for 92%), not as a percentage or integer.
}}

If planning a trip:
Return ONLY JSON matching the schema.

If answering a general travel question:
Return normal text.
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
    tripplan = parse_tripplan_from_llm_output(json_block)

    state["messages"] = []  # Ensure only one message is returned

    from travel.format_trip import format_trip_for_chat
    if tripplan:
        formatted = format_trip_for_chat(tripplan)
        state["tripplan"] = tripplan
        state["messages"].append(AIMessage(content=formatted))
    elif json_block:
        # If TripPlan parsing fails but JSON block exists, try to display readable format
        try:
            data = json.loads(json_block)
            # Defensive: handle confidence normalization
            if "confidence" in data:
                conf = data["confidence"]
                if isinstance(conf, str) and "%" in conf:
                    conf = float(conf.replace("%", "")) / 100
                data["confidence"] = conf
            tripplan_fallback = TripPlan(**data)
            formatted = format_trip_for_chat(tripplan_fallback)
            state["tripplan"] = tripplan_fallback
            state["messages"].append(AIMessage(content=formatted))
        except Exception:
            state["tripplan"] = None
            state["messages"].append(
                AIMessage(
                    content="⚠️ I couldn't generate a structured trip plan. Please try rephrasing your request.")
            )
    else:
        # No JSON found - this is a general travel question, return raw text response
        state["tripplan"] = None
        state["messages"].append(AIMessage(content=raw_content))

    # Step 5: Ensure only one message is returned
    if len(state["messages"]) > 1:
        state["messages"] = [state["messages"][-1]]

    return {
        "messages": state["messages"],
        "selected_trip_id": state.get("selected_trip_id"),
        "trips": state.get("trips", []),
        "tripplan": state.get("tripplan"),
    }
