from pydantic import BaseModel
from typing import List


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
