"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import
from typing import cast
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from travel.planner import planner_node
from travel.critic import critic_node
from travel.decision import decision_node
from travel.formatter import formatter_node
from travel.trips import trips_node
from travel.search import search_node
from travel.trips import perform_trips_node
from langsmith import traceable
from travel.state import AgentState
import logging

logger = logging.getLogger(__name__)

# Routing function after planner node
def planner_route(state: AgentState):
    """Route after the planner node.

    Routes to tools if the planner emitted a structured action, otherwise routes to critic.
    """
    action = state.get("action")
    if isinstance(action, dict):
        action_type = action.get("type")
        if action_type in ["add_trips", "update_trips", "delete_trips", "select_trip"]:
            logger.info("Routing to trips_node for action: %s", action_type)
            return "trips_node"
        if action_type == "search_places":
            logger.info("Routing to search_node for action: search_places")
            return "search_node"
        logger.info("Unknown action: %s, routing back to planner", action_type)
        return "planner_node"

    logger.info("No action detected, routing to critic_node")
    return "critic_node"


def decision_route(state: AgentState):
    """
    Route after decision node based on critic validation result.
    
    If critic rejected the plan, loop back to planner for regeneration.
    If critic accepted the plan, route to formatter for output.
    """
    if state.get("critic_reject"):
        logger.info("Decision: Critic rejected, routing back to planner for regeneration")
        return "planner_node"
    else:
        # Check if there's a tripplan to format
        if state.get("tripplan"):
            logger.info("Decision: Critic accepted, routing to formatter_node")
            return "formatter_node"
        else:
            logger.info("Decision: No tripplan, routing to END")
            return END


# Build the graph
graph_builder = StateGraph(AgentState)

# Add all nodes
graph_builder.add_node("planner_node", traceable(planner_node))
graph_builder.add_node("critic_node", traceable(critic_node))
graph_builder.add_node("decision_node", traceable(decision_node))
graph_builder.add_node("formatter_node", traceable(formatter_node))
graph_builder.add_node("trips_node", traceable(trips_node))
graph_builder.add_node("search_node", traceable(search_node))
graph_builder.add_node("perform_trips_node", traceable(perform_trips_node))

# Define edges
graph_builder.add_edge(START, "planner_node")

# Planner routes to tools or critic
graph_builder.add_conditional_edges(
    "planner_node", planner_route, ["search_node", "trips_node", "critic_node", "planner_node"])

# Tool nodes route back to planner
graph_builder.add_edge("search_node", "planner_node")
graph_builder.add_edge("trips_node", "perform_trips_node")
graph_builder.add_edge("perform_trips_node", "planner_node")

# Critic routes to decision
graph_builder.add_edge("critic_node", "decision_node")

# Decision routes based on validation
graph_builder.add_conditional_edges(
    "decision_node", decision_route, ["planner_node", "formatter_node", END])

# Formatter routes to END
graph_builder.add_edge("formatter_node", END)

# Compile the graph
graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["trips_node"],
)

