#!/usr/bin/env python3
"""Debug script to see what the LLM returns for a simple query."""

from huggingface_hub import InferenceClient
import os
import sys
import json
import logging
from dotenv import load_dotenv

# Setup logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

if not HUGGINGFACE_API_KEY:
    print("❌ HUGGINGFACE_API_KEY not set!")
    sys.exit(1)

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

# Test query
user_request = "trip to tokyo"

system_message = """SYSTEM INSTRUCTION - READ CAREFULLY:

You are Sarathi, a travel planning expert. Your ONLY job is to respond with JSON.

<CRITICAL_RULE>
RESPOND ONLY WITH RAW JSON. NOTHING ELSE.
- NO human text before or after JSON
- NO emoji (🌍, 📍, ⭐, etc.)
- NO "Day 1", "Day 2" formatting
- NO bullet points (•)
- NO markdown or special characters
- Your response must START with { and END with }
- Every single character outside { } will cause failure
</CRITICAL_RULE>

<FORBIDDEN_RESPONSES>
WRONG: 🌍 Trip Plan: Tokyo (2 days)
WRONG: Day 1 • Visit...
WRONG: Error (INVALID_DESTINATION): Heaven...
WRONG: <html>...
WRONG: ```json ... ```
WRONG: "Here's your trip:"
ALL OF THESE WILL FAIL AND BE REJECTED
</FORBIDDEN_RESPONSES>

<ONLY_VALID_RESPONSES>
CORRECT: {"tripplan": {"city": "Tokyo", "days": 2, ...}}
CORRECT: {"error": {"code": "INVALID_DESTINATION", "message": "..."}}
ONLY JSON OBJECTS STARTING WITH { AND ENDING WITH } ARE VALID
</ONLY_VALID_RESPONSES>

<REQUIRED_JSON_SCHEMA>
RESPOND WITH EXACTLY ONE of:

1. SUCCESS (if destination is valid):
{
  "tripplan": {
    "city": "Tokyo",
    "days": 2,
    "itinerary": [
      {
        "day": 1,
        "activities": [
          {"name": "Senso-ji Temple", "description": "Historic Buddhist temple", "duration_hours": 2}
        ]
      }
    ]
  }
}

2. ERROR (if destination invalid/missing):
{
  "error": {
    "code": "INVALID_DESTINATION",
    "message": "Tokyo is a valid destination..."
  }
}

REMEMBER: ONLY JSON, NOTHING ELSE!
"""

print("=" * 80)
print(f"Testing LLM with query: '{user_request}'")
print(f"Model: {HUGGINGFACE_MODEL}")
print("=" * 80)
print("\n📋 SYSTEM MESSAGE (truncated):\n")
print(system_message[:500] + "\n... [truncated] ...\n" + system_message[-300:])

try:
    completion = client.chat.completions.create(
        model=HUGGINGFACE_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request}
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw_output = completion.choices[0].message.content

    print("\n📝 RAW LLM OUTPUT:")
    print("-" * 80)
    print(raw_output)
    print("-" * 80)

    print(f"\n📊 OUTPUT ANALYSIS:")
    print(f"  Length: {len(raw_output)} chars")
    print(f"  Starts with '{{': {raw_output.strip().startswith('{')}")
    print(f"  Ends with '}}': {raw_output.strip().endswith('}')}")
    print(
        f"  Contains emoji: {'🌍' in raw_output or '📍' in raw_output or '⭐' in raw_output}")
    print(
        f"  Contains 'Day 1': {'Day 1' in raw_output or 'Day 2' in raw_output}")

    # Try to parse as JSON
    import re
    print(f"\n🔍 PARSING ATTEMPTS:")

    # Attempt 1: Direct parse
    try:
        payload = json.loads(raw_output)
        print(f"  ✅ Direct JSON parse: SUCCESS")
        print(f"     Keys: {list(payload.keys())}")
    except Exception as e:
        print(f"  ❌ Direct JSON parse: {e}")

        # Attempt 2: raw_decode
        try:
            decoder = json.JSONDecoder()
            payload, _ = decoder.raw_decode(raw_output)
            print(f"  ✅ raw_decode: SUCCESS")
            print(f"     Keys: {list(payload.keys())}")
        except Exception as e2:
            print(f"  ❌ raw_decode: {e2}")

            # Attempt 3: Extract JSON chunk
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                print(f"  ✅ Regex extraction: FOUND JSON chunk")
                try:
                    payload = json.loads(match.group(0))
                    print(f"  ✅ Parsing extracted chunk: SUCCESS")
                    print(f"     Keys: {list(payload.keys())}")
                except Exception as e3:
                    print(f"  ❌ Parsing extracted chunk: {e3}")
            else:
                print(f"  ❌ Regex extraction: NO JSON chunk found")

except Exception as e:
    print(f"❌ LLM API Error: {e}")
    import traceback
    traceback.print_exc()
