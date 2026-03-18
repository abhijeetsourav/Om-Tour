# SARATHI Evaluation Framework

Quantitative evaluation suite for the SARATHI travel planning agent.

## Overview

The evaluation framework measures reliability and quality of the planner + critic system across:
- **Schema Success Rate**: % of queries returning valid JSON tripplans
- **Critic Rejection Rate**: % of queries rejected by validation rules
- **Hallucination Rate**: % of itineraries with unverified places
- **Realism Score**: Activity density scoring (1.0 = perfect, 0.0 = unrealistic)
- **Average Activities/Day**: Mean across all accepted itineraries
- **Response Latency**: Time to process each query

## Test Dataset

**File**: `test_queries.json`

Contains 16 test cases across three categories:

### REALISTIC (5 cases)
- Valid real destinations (Rome, Manali, Agra, Paris, Delhi)
- Expected: Accept with reasonable activities per day (4-6)

### UNREALISTIC (5 cases)
- Fictional destinations (Mars, Atlantis)
- Excessive activities (20-50 per day)
- Inaccessible locations (nuclear weapons base)
- Expected: Reject or regenerate with feedback

### EDGE CASES (6 cases)
- Greetings ("hi", "hello")
- Vague requests ("weekend trip near bangalore")
- General questions without trip context
- Expected: Skip validation, respond appropriately

## Running Evaluation

```bash
cd /workspaces/Om-Tour/agent

# Run full evaluation
poetry run python evaluation/run_eval.py

# Run specific test (modify run_eval.py to filter by ID)
# Results saved to: evaluation/evaluation_report.json
```

## Output

### Console Summary
```
SARATHI TRAVEL AGENT - EVALUATION SUMMARY

Total Queries: 16
Accepted: 12 | Rejected: 3 | Errors: 1

KEY METRICS
Schema Success Rate:        93.8%
Critic Rejection Rate:      18.8%
Hallucination Rate:          5.0%
Zero-Cost Activities Rate:   2.1%

ITINERARY QUALITY
Avg Activities per Day:       5.23
Max Activities in Any Day:   12.0
Avg Realism Score:          0.890 (1.0 = perfect)
Avg Response Latency:      2234.1 ms

CATEGORY BREAKDOWN
REALISTIC
  Total:        5
  Acceptance:  100.0%
  Rejection:    0.0%

UNREALISTIC
  Total:        5
  Acceptance:   20.0%
  Rejection:   80.0%

EDGE_CASES
  Total:        6
  Acceptance:   83.3%
  Rejection:    0.0%
```

### JSON Report
**File**: `evaluation_report.json`

Detailed metrics in JSON format:
- Aggregate statistics
- Status breakdown
- Category breakdown
- Detailed per-query results with:
  - Query and category
  - Status (accepted/rejected/error)
  - Activities per day
  - Realism score
  - Hallucinations detected
  - Response latency

## Metrics Explained

### Schema Success Rate
**What**: % of responses with valid TripPlan JSON
**Good**: >90% (most trip queries return parseable JSON)

### Critic Rejection Rate
**What**: % of accepted itineraries rejected by rules
**Good**: Depends on query type
- Realistic: 0% (should accept)
- Unrealistic: >50% (should reject)

### Hallucination Rate
**What**: % of itineraries with unverified places
**Good**: <5% (most activities should be real places)

### Zero-Cost Rate
**What**: % of itineraries with cost=0.0
**Good**: <10% (placeholder values indicate hallucination)

### Realism Score
**Formula**: `1 - (max_activities_per_day / 12)`
**Interpretation**:
- 1.0: Max 0 activities (rejected query)
- 0.83: Max 2 activities (very relaxed)
- 0.60: Max 5 activities (realistic)
- 0.33: Max 8 activities (acceptable)
- 0.08: Max 11 activities (unrealistic)
- 0.0: Max 12+ activities (impossible)

### Response Latency
**What**: Time from query to response (seconds)
**Good**: <5 seconds for most queries

## Evaluation Workflow

1. **Load test queries** from `test_queries.json`
2. **Initialize agent state** for each query
3. **Invoke agent graph** asynchronously
4. **Parse response**:
   - Check for valid TripPlan JSON
   - Verify critic rejection status
   - Extract metrics from itinerary
5. **Detect hallucinations** by comparing places to known real locations
6. **Calculate realism score** based on activity density
7. **Aggregate results** across all queries and categories
8. **Generate JSON report** with detailed metrics
9. **Print console summary** with key statistics

## Extending Tests

To add more test cases, edit `test_queries.json`:

```json
{
  "id": "custom_1",
  "category": "REALISTIC",
  "query": "10 day trip to switzerland with hiking",
  "expected_outcome": "should accept",
  "expected_activities_per_day": 4,
  "is_trip_query": true
}
```

Fields:
- `id`: Unique identifier
- `category`: REALISTIC, UNREALISTIC, or EDGE_CASES
- `query`: Natural language request
- `expected_outcome`: What should happen
- `expected_activities_per_day`: Target activities/day
- `is_trip_query`: Whether this is a trip planning request

## Success Criteria

A healthy SARATHI system should achieve:
- **Schema Success Rate**: >90%
- **Critic Rejection Rate for UNREALISTIC**: >70%
- **Hallucination Rate**: <5%
- **Zero-Cost Rate**: <3%
- **Avg Realism Score**: >0.75 (mostly 5-6 activities/day)
- **Avg Response Latency**: <3 seconds

## Troubleshooting

### Timeout errors
- Increase `timeout=30.0` in `run_eval.py` if API responses are slow
- Check HuggingFace API availability

### High hallucination rate
- Update `known_places` set in `TravelAgentEvaluator`
- Check if planner is generating fabricated locations

### High zero-cost rate
- Verify planner is including cost estimation in prompt
- Check if LLM fallback is returning placeholder data

### Low schema success rate
- Check JSON parsing in planner node
- Verify LLM output format compliance

---

**Last Updated**: March 2026
**Framework Version**: 1.0
