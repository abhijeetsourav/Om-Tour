from typing import Dict, List, Optional
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
    """The state of the agent.
    
    Fields:
    - messages: List of messages in the conversation
    - selected_trip_id: Currently selected trip ID
    - trips: List of user's saved trips
    - search_progress: Progress of place searches
    - planning_progress: Progress of trip planning
    - tripplan: Generated TripPlan (structured output)
    - critic_reject: Whether critic rejected the plan
    - critic_feedback: Structured feedback from critic validation
    - user_query: Original user query (for validation)
    - retrieved_places: Places retrieved from search tools
    - regen_attempts: Counter for regeneration attempts
    """
    selected_trip_id: Optional[str]
    trips: List[Trip]
    search_progress: List[SearchProgress]
    planning_progress: List[PlanningProgress]
    tripplan: Optional[TripPlan] = None
    action: Optional[Dict] = None
    critic_reject: bool = False
    critic_feedback: Optional[dict] = None
    user_query: Optional[str] = None
    retrieved_places: List[Place] = []
    regen_attempts: int = 0


# Constants
MAX_REGEN_ATTEMPTS = 2
