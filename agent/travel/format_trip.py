from schemas.trip_schema import TripPlan


def format_trip_for_chat(trip: TripPlan) -> str:

    confidence_percent = round(trip.confidence * 100) if hasattr(
        trip, "confidence") and trip.confidence is not None else None
    lines = []
    lines.append(f"🌍 Trip Plan: {trip.city} ({trip.days} days)\n")
    for day in trip.itinerary:
        lines.append(f"Day {day.day}")
        for activity in day.activities:
            lines.append(f" • {activity.name}")
        lines.append("")
    lines.append(f"Estimated Budget: ${trip.estimated_budget}")
    if confidence_percent is not None:
        lines.append(f"Confidence: {confidence_percent}%")
    return "\n".join(lines)
