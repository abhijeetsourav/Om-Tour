import os
from langchain_core.tools import tool
from typing import cast
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from travel.trips import add_trips, update_trips, delete_trips
from travel.search import search_for_places
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from travel.state import AgentState
import json
from dotenv import load_dotenv
load_dotenv()


# from langchain_openai import ChatOpenAI

groq_api_key = os.getenv("GROQ_API_KEY")


@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"


llm = ChatGroq(model="llama-3.3-70b-versatile",
               temperature=0, api_key=groq_api_key)
# llm = ChatOpenAI(model="gpt-4o")
tools = [search_for_places, select_trip]


async def chat_node(state: AgentState, config: RunnableConfig):
    """Handle chat operations"""
    # print("CHAT NODE CALLED")
    if not groq_api_key:
        return {"messages": [AIMessage(content="Error: GROQ_API_KEY not set.")], "selected_trip_id": state.get("selected_trip_id", None), "trips": state.get("trips", [])}
    llm_with_tools = llm.bind_tools(
        [
            *tools,
            add_trips,
            update_trips,
            delete_trips,
            select_trip,
        ],
        parallel_tool_calls=False,
    )

    system_message = f"""
    You are an agent that plans trips and helps the user with planning and managing their trips.
    If the user did not specify a location, you should ask them for a location.
    
    Plan the trips for the user, take their preferences into account if specified, but if they did not
    specify any preferences, call the search_for_places tool to find places of interest, restaurants, and activities.

    Unless the users prompt specifies otherwise, only use the first 10 results from the search_for_places tool relevant
    to the trip.

    When you add or edit a trip, you don't need to summarize what you added. Just give a high level summary of the trip
    and why you planned it that way.
    
    When you create or update a trip, you should set it as the selected trip.
    If you delete a trip, try to select another trip.

    If an operation is cancelled by the user, DO NOT try to perform the operation again. Just ask what the user would like to do now
    instead.

    IMPORTANT: When calling add_trips or update_trips:
    - Always include a 'zoom' level (e.g., 13 for city level, 15 for detailed)
    - Always include a 'description' for each place, even if empty
    - All other required fields (id, name, address, latitude, longitude, rating) must be included

    Current trips: {json.dumps(state.get('trips', []))}
    """

    # calling ainvoke instead of invoke is essential to get streaming to work properly on tool calls.
    try:
        response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=system_message),
                *state["messages"]
            ],
            config=config,
        )
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {str(e)}")], "selected_trip_id": state.get("selected_trip_id", None), "trips": state.get("trips", [])}

    print("LLM Response:", response)

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        if ai_message.tool_calls[0]["name"] == "select_trip":
            return {
                "selected_trip_id": ai_message.tool_calls[0]["args"].get("trip_id", ""),
                "messages": [ai_message, ToolMessage(
                    tool_call_id=ai_message.tool_calls[0]["id"],
                    content="Trip selected."
                )]
            }

    return {
        "messages": [response],
        "selected_trip_id": state.get("selected_trip_id", None),
        "trips": state.get("trips", [])
    }
