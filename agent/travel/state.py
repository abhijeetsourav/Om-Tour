from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from schemas.trip_schema import TripPlan


class Place(TypedDict):
    """A place."""
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    rating: float
    description: Optional[str]


class Trip(TypedDict):
    """A trip."""
    id: str
    name: str
    center_latitude: float
    center_longitude: float
    zoom: int  # 13 for city, 15 for airport
    places: List[Place]


class SearchProgress(TypedDict):
    """The progress of a search."""
    query: str
    results: list[str]
    done: bool


class PlanningProgress(TypedDict):
    """The progress of a planning."""
    trip: Trip
    done: bool


class AgentState(MessagesState):
    """The state of the agent."""
    selected_trip_id: Optional[str]
    trips: List[Trip]
    search_progress: List[SearchProgress]
    planning_progress: List[PlanningProgress]
    tripplan: Optional[TripPlan] = None
    critic_reject: bool = False
