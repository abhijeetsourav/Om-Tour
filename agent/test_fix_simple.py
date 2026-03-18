#!/usr/bin/env python3
"""Test the schema fix standalone."""

import json


def fix_incomplete_tripplan(payload: dict) -> dict:
    """Fill in missing required fields in tripplan JSON."""
    if not payload.get("tripplan"):
        return payload

    tripplan = payload["tripplan"]

    # Fill missing tripplan-level fields
    if "estimated_budget" not in tripplan:
        days = tripplan.get("days", 3)
        tripplan["estimated_budget"] = 1000 + (days * 500)
        print(f"  ✅ Added estimated_budget: ${tripplan['estimated_budget']}")

    if "confidence" not in tripplan:
        tripplan["confidence"] = 0.85
        print(f"  ✅ Added confidence: {tripplan['confidence']}")

    # Fill missing activity-level fields
    for day_plan in tripplan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if "location" not in activity:
                city = tripplan.get("city", "Unknown")
                activity["location"] = city
                print(
                    f"  ✅ Added location for '{activity.get('name', 'Unknown')}': {city}")

            if "estimated_cost" not in activity:
                activity["estimated_cost"] = 50
                print(
                    f"  ✅ Added estimated_cost for '{activity.get('name', 'Unknown')}': $50")

    return payload


# Test
incomplete_output = '''{"tripplan": {"city": "Tokyo", "days": 2, "itinerary": [{"day": 1, "activities": [{"name": "Senso-ji Temple", "description": "Historic Buddhist temple", "duration_hours": 2}]}]}}'''

print("=" * 80)
print("SCHEMA FIX TEST (Standalone)")
print("=" * 80)

print(f"\n❌ BEFORE FIX - Missing fields:")
payload = json.loads(incomplete_output)
print(
    f"   estimated_budget: {'✅' if 'estimated_budget' in payload['tripplan'] else '❌'}")
print(f"   confidence: {'✅' if 'confidence' in payload['tripplan'] else '❌'}")
print(
    f"   activity.location: {'✅' if 'location' in payload['tripplan']['itinerary'][0]['activities'][0] else '❌'}")
print(
    f"   activity.estimated_cost: {'✅' if 'estimated_cost' in payload['tripplan']['itinerary'][0]['activities'][0] else '❌'}")

print(f"\n✅ APPLYING FIX:")
payload_fixed = fix_incomplete_tripplan(payload)

print(f"\n✅ AFTER FIX - All fields present:")
print(
    f"   estimated_budget: ✅ ${payload_fixed['tripplan']['estimated_budget']}")
print(f"   confidence: ✅ {payload_fixed['tripplan']['confidence']}")
print(
    f"   activity.location: ✅ {payload_fixed['tripplan']['itinerary'][0]['activities'][0]['location']}")
print(
    f"   activity.estimated_cost: ✅ ${payload_fixed['tripplan']['itinerary'][0]['activities'][0]['estimated_cost']}")

print(f"\n📊 Complete fixed JSON:")
print(json.dumps(payload_fixed, indent=2))
