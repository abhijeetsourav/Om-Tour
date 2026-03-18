from pydantic import BaseModel
from typing import List, Literal, Optional, Union


class Place(BaseModel):
    name: str
    description: str
    location: str
    estimated_cost: float


class DayPlan(BaseModel):
    day: int
    activities: List[Place]


class TripPlan(BaseModel):
    city: str
    days: int
    itinerary: List[DayPlan]
    estimated_budget: float
    confidence: float


class PlannerAction(BaseModel):
    type: Literal["search_places", "add_trips", "update_trips", "delete_trips", "select_trip"]
    parameters: dict


class PlannerError(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class PlannerResponse(BaseModel):
    tripplan: Optional[TripPlan] = None
    action: Optional[PlannerAction] = None
    error: Optional[PlannerError] = None

    class Config:
        extra = "forbid"
