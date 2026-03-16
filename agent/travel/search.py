from langsmith import traceable
import logging
from travel.state import AgentState
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from typing import cast
import googlemaps
import json
import os


def format_places(places):
    formatted = []
    for p in places:
        formatted.append({
            "name": p.get("name"),
            "rating": p.get("rating"),
            "address": p.get("address"),
            "location": {
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude")
            }
        })
    return formatted


logger = logging.getLogger(__name__)
"""
The search node is responsible for searching the internet for information.
"""


def filter_places(places):
    """Filter and rank places returned from Google Maps."""
    filtered = []
    for p in places:
        rating = p.get("rating", 0)
        reviews = p.get("user_ratings_total", 0)
        if rating >= 4.0 and reviews >= 50:
            filtered.append(p)
    # deduplicate by place name
    seen = set()
    unique_places = []
    for p in filtered:
        name = p.get("name")
        if name not in seen:
            seen.add(name)
            unique_places.append(p)
    # rank by rating
    ranked = sorted(
        unique_places,
        key=lambda x: x.get("rating", 0),
        reverse=True
    )
    return ranked[:5]


@tool
def search_for_places(queries: list[str]) -> list[dict]:
    """Search for places based on a query, returns a list of places including their name, address, and coordinates."""


GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

gmaps = None
if GOOGLE_MAPS_API_KEY:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


@traceable
async def search_node(state: AgentState, config: RunnableConfig):
    """
    The search node is responsible for searching the for places.
    """
    if not gmaps:
        state["messages"].append(ToolMessage(tool_call_id=state["messages"]
                                 [-1].tool_calls[0]["id"], content="Error: Google Maps API key missing."))
        return state
    ai_message = cast(AIMessage, state["messages"][-1])

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "search_progress",
            "tool": "search_for_places",
            "tool_argument": "search_progress",
        }],
    )

    state["search_progress"] = state.get("search_progress", [])
    queries = ai_message.tool_calls[0]["args"]["queries"]

    for query in queries:
        state["search_progress"].append({
            "query": query,
            "results": [],
            "done": False
        })

    await copilotkit_emit_state(config, state)

    def google_maps_search(query):
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
        return places

    all_filtered_places = []
    for i, query in enumerate(queries):
        places = google_maps_search(query)
        filtered_places = filter_places(places)
        logger.info(
            f"Retrieved {len(places)} places → {len(filtered_places)} after filtering"
        )
        log_msg = f"Google Maps: Query '{query}' — Filtered {len(places)} places, returned {len(filtered_places)}."
        print(log_msg)
        all_filtered_places.extend(filtered_places)
        state["search_progress"][i]["done"] = True
        await copilotkit_emit_state(config, state)

    # Deduplicate across all queries
    seen = set()
    unique_filtered_places = []
    for p in all_filtered_places:
        name = p.get("name")
        if name not in seen:
            seen.add(name)
            unique_filtered_places.append(p)

    state["search_progress"] = []
    await copilotkit_emit_state(config, state)

    formatted_places = format_places(unique_filtered_places)
    state["messages"].append(ToolMessage(
        tool_call_id=ai_message.tool_calls[0]["id"],
        content=f"Results: {json.dumps(formatted_places)}"
    ))

    return state
