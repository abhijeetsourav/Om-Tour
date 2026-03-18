from pydantic import BaseModel
from typing import List, Literal, Optional, Union, Dict


class Place(BaseModel):
    name: str
    description: str
    location: str
    estimated_cost: float


class ConfidenceBreakdown(BaseModel):
    """Granular confidence scoring by component"""
    destination_certainty: float  # 0-1: How sure we are destination is real/accessible
    itinerary_realism: float      # 0-1: How realistic the itinerary is
    pricing_accuracy: float        # 0-1: How accurate the pricing is
    feasibility: float             # 0-1: How feasible the plan is overall


class DayPlan(BaseModel):
    day: int
    activities: List[Place]


class TripPlan(BaseModel):
    city: str
    days: int
    itinerary: List[DayPlan]
    estimated_budget: float
    confidence: float
    # New field for granular confidence
    confidence_breakdown: Optional[ConfidenceBreakdown] = None


class PlannerAction(BaseModel):
    type: Literal["search_places", "add_trips",
                  "update_trips", "delete_trips", "select_trip"]
    parameters: dict


class PlannerError(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None
    # New field for missing info cases
    clarification_questions: Optional[List[str]] = None


class PlannerResponse(BaseModel):
    tripplan: Optional[TripPlan] = None
    action: Optional[PlannerAction] = None
    error: Optional[PlannerError] = None

    class Config:
        extra = "forbid"
