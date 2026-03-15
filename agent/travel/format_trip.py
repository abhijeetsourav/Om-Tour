from schemas.trip_schema import TripPlan


def format_trip_for_chat(trip: TripPlan) -> str:
    lines = []
    lines.append(f"🌍 Trip Plan: {trip.city} ({trip.days} days)\n")

    for day in trip.itinerary:
        lines.append(f"Day {day.day}:")

        for act in day.activities:
            lines.append(f" • {act.name} — {act.location}")

        lines.append("")

    lines.append(f"Estimated Budget: ${trip.estimated_budget}")
    lines.append(f"Confidence: {round(trip.confidence * 100)}%")

    return "\n".join(lines)
