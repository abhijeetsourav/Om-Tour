from schemas.trip_schema import TripPlan


def format_trip_for_chat(trip: TripPlan) -> str:
    """Format a TripPlan into human-readable text with optional confidence breakdown"""

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

        # Show granular confidence breakdown if available
        if hasattr(trip, "confidence_breakdown") and trip.confidence_breakdown:
            cb = trip.confidence_breakdown
            lines.append("\nConfidence Breakdown:")
            if hasattr(cb, "destination_certainty"):
                lines.append(
                    f"  📍 Destination: {round(cb.destination_certainty * 100)}%")
            if hasattr(cb, "itinerary_realism"):
                lines.append(
                    f"  📋 Itinerary Realism: {round(cb.itinerary_realism * 100)}%")
            if hasattr(cb, "pricing_accuracy"):
                lines.append(
                    f"  💰 Pricing: {round(cb.pricing_accuracy * 100)}%")
            if hasattr(cb, "feasibility"):
                lines.append(
                    f"  ✈️ Feasibility: {round(cb.feasibility * 100)}%")

    return "\n".join(lines)
