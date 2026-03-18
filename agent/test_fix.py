#!/usr/bin/env python3
"""Test the schema fix with post-processing."""

from travel.planner import fix_incomplete_tripplan, parse_planner_response
from schemas.trip_schema import PlannerResponse
import json
import sys
sys.path.insert(0, '/workspaces/Om-Tour/agent')


# The exact incomplete output from the LLM
incomplete_output = '''{"tripplan": {"city": "Tokyo", "days": 2, "itinerary": [{"day": 1, "activities": [{"name": "Senso-ji Temple", "description": "Historic Buddhist temple", "duration_hours": 2}]}]}}'''

print("=" * 80)
print("SCHEMA FIX TEST")
print("=" * 80)

print(f"\n❌ BEFORE FIX:")
print("Missing:")
print("  - tripplan.estimated_budget")
print("  - tripplan.confidence")
print("  - activity.location")
print("  - activity.estimated_cost")

# Parse and show what happens
payload_before = json.loads(incomplete_output)
try:
    response = PlannerResponse.parse_obj(payload_before)
    print(f"❌ Validation: FAILED (should have failed)")
except Exception as e:
    print(f"✅ Validation: FAILED as expected - schema mismatch")

# Now test with fix
print(f"\n✅ AFTER FIX:")
payload_after = fix_incomplete_tripplan(payload_before)

print("\nFixed output:")
print(json.dumps(payload_after, indent=2)[:500] + "...\n")

try:
    response = PlannerResponse.parse_obj(payload_after)
    print(f"✅ Validation: SUCCESS!")
    print(f"\n📊 Parsed Response:")
    print(f"   City: {response.tripplan.city}")
    print(f"   Days: {response.tripplan.days}")
    print(f"   Estimated Budget: ${response.tripplan.estimated_budget}")
    print(f"   Confidence: {response.tripplan.confidence}")
    print(f"   Activities:")
    for day in response.tripplan.itinerary:
        for activity in day.activities:
            print(
                f"     - {activity.name} (${activity.estimated_cost}) at {activity.location}")
except Exception as e:
    print(f"❌ Validation: FAILED - {e}")

# Now test parse_planner_response directly
print(f"\n" + "=" * 80)
print("TESTING parse_planner_response()")
print("=" * 80)
result = parse_planner_response(incomplete_output)
if result:
    print(f"✅ parse_planner_response: SUCCESS")
    print(
        f"   Tripplan: {result.tripplan.city if result.tripplan else 'None'}")
else:
    print(f"❌ parse_planner_response: FAILED")
