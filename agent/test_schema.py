#!/usr/bin/env python3
"""Test schema validation with the exact LLM output."""

from schemas.trip_schema import PlannerResponse, TripPlan, DayPlan, Place
import json
import sys
sys.path.insert(0, '/workspaces/Om-Tour/agent')


# The exact output from the LLM
llm_output = '''{"tripplan": {"city": "Tokyo", "days": 2, "itinerary": [{"day": 1, "activities": [{"name": "Senso-ji Temple", "description": "Historic Buddhist temple", "duration_hours": 2}]}]}}'''

print("=" * 80)
print("SCHEMA VALIDATION TEST")
print("=" * 80)
print(f"\n📝 LLM Output:\n{llm_output}\n")

# Parse JSON
payload = json.loads(llm_output)
print(f"✅ JSON parsed successfully")
print(f"   Keys in payload: {list(payload.keys())}")

# Try to validate against schema
print(f"\n🔍 Validating against PlannerResponse schema...\n")

try:
    response = PlannerResponse.parse_obj(payload)
    print(f"✅ SUCCESS: PlannerResponse validated!")
    print(f"   tripplan: {response.tripplan}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print(f"\n🔧 Let's see what fields are missing:")

    # Try validating just the TripPlan
    try:
        tripplan_data = payload.get("tripplan", {})
        trip = TripPlan.parse_obj(tripplan_data)
        print(f"   TripPlan validated")
    except Exception as e2:
        print(f"   TripPlan validation error: {e2}")

        # Check required fields
        required_fields = {
            "city": "required",
            "days": "required",
            "itinerary": "required",
            "estimated_budget": "required",
            "confidence": "required",
        }

        tripplan_data = payload.get("tripplan", {})
        print(f"\n   Fields present in tripplan:")
        for field in required_fields:
            present = field in tripplan_data
            symbol = "✅" if present else "❌"
            print(
                f"     {symbol} {field}: {tripplan_data.get(field, 'MISSING')}")

        # Check activity fields
        print(f"\n   Checking activity fields:")
        for day in tripplan_data.get("itinerary", []):
            print(f"     Day {day.get('day')}:")
            for activity in day.get("activities", []):
                print(f"       Activity: {activity.get('name')}")
                for required_field in ["name", "description", "location", "estimated_cost"]:
                    present = required_field in activity
                    symbol = "✅" if present else "❌"
                    print(
                        f"         {symbol} {required_field}: {activity.get(required_field, 'MISSING')}")
