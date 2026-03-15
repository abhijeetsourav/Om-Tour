from schemas.trip_schema import TripPlan
from langsmith import traceable
import os
import json
from typing import cast

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langchain_groq import ChatGroq

from travel.trips import add_trips, update_trips, delete_trips
from travel.search import search_for_places
from travel.state import AgentState
from travel.format_trip import format_trip_for_chat

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"


# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key
)

tools = [search_for_places, select_trip]


def extract_json_block(text: str):
    """
    Extract JSON block from model output.
    Handles cases where model adds text before/after JSON.
    """

    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    return text[start:end + 1]


def format_if_trip_json(message_content: str) -> str:
    """
    If message_content contains TripPlan JSON, parse and format it.
    Otherwise, return content unchanged.
    """
    from travel.format_trip import format_trip_for_chat
    from schemas.trip_schema import TripPlan
    json_block = extract_json_block(message_content)
    if not json_block:
        return message_content
    try:
        tripplan = TripPlan.parse_raw(json_block)
        return format_trip_for_chat(tripplan)
    except Exception:
        return message_content


@traceable
async def chat_node(state: AgentState, config: RunnableConfig):

    if not groq_api_key:
        return {
            "messages": [AIMessage(content="Error: GROQ_API_KEY not set.")],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", [])
        }

    llm_with_tools = llm.bind_tools(
        [
            *tools,
            add_trips,
            update_trips,
            delete_trips,
            select_trip
        ],
        parallel_tool_calls=False
    )

    system_message = f"""
You are Sarathi, an AI travel planner for Om Tours.

Help users plan trips and manage itineraries.

Rules:
- Ask for location if missing.
- Use search_for_places if preferences are not provided.
- Use only the top 10 relevant search results.
- When adding/updating trips, set them as selected.

Current trips:
{json.dumps(state.get('trips', []))}

FINAL OUTPUT RULE:
After all tool calls complete, return ONLY JSON matching this schema.

TripPlan:
{{
 city: string
 days: number
 itinerary: [
   {{
     day: number,
     activities: [
       {{
         name: string
         description: string
         location: string
         estimated_cost: number
       }}
     ]
   }}
 ]
 estimated_budget: number
 confidence: number
}}

Return ONLY JSON.
"""

    try:
        response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=system_message),
                *state["messages"]
            ],
            config=config
        )
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", [])
        }

    ai_message = cast(AIMessage, response)

    print("LLM content:", ai_message.content)
    print("Tool calls:", ai_message.tool_calls)

    # -------------------------------------------------
    # STEP 1 — HANDLE TOOL CALLS FIRST
    # -------------------------------------------------

    if ai_message.tool_calls:

        if ai_message.tool_calls[0]["name"] == "select_trip":
            return {
                "selected_trip_id": ai_message.tool_calls[0]["args"].get("trip_id", ""),
                "messages": [
                    ai_message,
                    ToolMessage(
                        tool_call_id=ai_message.tool_calls[0]["id"],
                        content="Trip selected."
                    )
                ]
            }

        return {
            "messages": [ai_message],
            "selected_trip_id": state.get("selected_trip_id"),
            "trips": state.get("trips", [])
        }

    # -------------------------------------------------
    # STEP 2 — PARSE FINAL RESPONSE
    # -------------------------------------------------

    raw_content = ai_message.content

    formatted_content = format_if_trip_json(raw_content)

    return {
        "messages": [AIMessage(content=formatted_content)],
        "selected_trip_id": state.get("selected_trip_id"),
        "trips": state.get("trips", [])
    }
