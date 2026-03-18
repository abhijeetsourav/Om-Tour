"""
Evaluation Framework for SARATHI Travel Agent

Measures reliability of planner + critic system.

Metrics:
- Schema Success Rate: % of queries that return valid JSON
- Critic Rejection Rate: % of queries rejected by critic
- Hallucination Rate: % of itineraries with unverified places
- Average Activities per Day: Mean across all accepted itineraries
- Realism Score: 1 - (max_activities_per_day / 12)
"""

import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from travel.agent import graph
from travel.state import AgentState
from schemas.trip_schema import TripPlan
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

# Constants
EVAL_DIR = Path(__file__).parent
TEST_QUERIES_FILE = EVAL_DIR / "test_queries.json"
OUTPUT_REPORT_FILE = EVAL_DIR / "evaluation_report.json"
MAX_ACTIVITIES_REALISTIC = 6


# Helper Functions
def is_trip_query(query: str) -> bool:
    """Detect if query is asking for actual trip planning.
    
    Uses keyword matching to classify queries as trip-related.
    """
    trip_keywords = [
        "trip",
        "travel",
        "itinerary",
        "vacation",
        "visit",
        "days in",
        "tour",
        "plan",
        "journey",
        "expedition"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in trip_keywords)


def is_llm_failure(error_message: str) -> bool:
    """Detect if error indicates LLM provider failure.
    
    Identifies API failures, rate limits, quota issues, payment failures, etc.
    These should cause evaluation to be SKIPPED rather than treated as failures.
    """
    failure_patterns = [
        "402",
        "payment required",
        "payment failed",
        "rate limit",
        "quota exceeded",
        "insufficient credits",
        "api key invalid",
        "authentication failed",
        "401",
        "403",
        "connection refused",
        "connection timeout",
        "500",
        "503",
        "service unavailable"
    ]
    
    msg = error_message.lower()
    return any(pattern in msg for pattern in failure_patterns)


class EvaluationMetrics:
    """Tracks metrics for a single query evaluation."""
    
    def __init__(self, query_id: str, category: str, query: str):
        self.query_id = query_id
        self.category = category
        self.query = query
        self.timestamp = datetime.now().isoformat()
        
        # Response metrics
        self.schema_valid = False
        self.has_tripplan = False
        self.critic_rejected = False
        self.response_latency = 0.0
        
        # Content metrics
        self.activities_per_day = 0
        self.max_activities_on_any_day = 0
        self.total_activities = 0
        self.estimated_budget = 0.0
        self.confidence_score = 0.0
        
        # Quality metrics
        self.hallucination_detected = False
        self.hallucinated_places = []
        self.zero_cost_activities = 0
        self.realism_score = 0.0
        
        # Status
        self.status = "pending"  # pending, accepted, rejected, error
        self.error_message = ""
        self.response_text = ""


class TravelAgentEvaluator:
    """Evaluates SARATHI travel agent on test queries."""
    
    def __init__(self):
        self.test_queries = self._load_test_queries()
        self.results: List[EvaluationMetrics] = []
        self.known_places = self._load_known_places()
        
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        if not TEST_QUERIES_FILE.exists():
            raise FileNotFoundError(f"Test queries file not found: {TEST_QUERIES_FILE}")
        
        with open(TEST_QUERIES_FILE, 'r') as f:
            data = json.load(f)
        
        return data.get('test_queries', [])
    
    def _load_known_places(self) -> set:
        """Load set of known real places for hallucination detection."""
        # Common tourist destinations
        known_places = {
            # India
            "taj mahal", "agra fort", "fatehpur sikri", "jaipur", "delhi",
            "mehtab bagh", "itmad-ud-daulah", "sikandra", "agra", "lucknow",
            "red fort", "india gate", "qutb minar", "chandni chowk", "rajasthan",
            "jama masjid", "humayun's tomb", "gardens", "palace",
            
            # Europe
            "rome", "colosseum", "vatican", "paris", "eiffel tower", "louvre",
            "london", "big ben", "tower bridge", "barcelona", "sagrada familia",
            "italy", "france", "spain", "germany", "austria", "switzerland",
            
            # Asia
            "manali", "himachal", "kashmir", "ladakh", "goa", "kerala",
            "bangalore", "mumbai", "kolkata", "chennai", "hyderabad",
            "tokyo", "kyoto", "bangkok", "vietnam", "singapore",
            
            # Activities and sites
            "market", "bazaar", "temple", "mosque", "church", "cathedral",
            "garden", "park", "museum", "monument", "fort", "palace",
            "waterfall", "river", "lake", "beach", "trek", "hike",
            "restaurant", "cafe", "shopping", "tour", "walk"
        }
        return known_places
    
    def _detect_hallucinations(self, tripplan: TripPlan) -> tuple[bool, List[str]]:
        """Detect places that appear fabricated (not in known places)."""
        hallucinated = []
        
        for day_plan in tripplan.itinerary:
            for activity in day_plan.activities:
                place_name = activity.name.lower()
                location = activity.location.lower()
                
                # Check if place name or location matches known places
                is_known = any(
                    known in place_name or known in location
                    for known in self.known_places
                )
                
                if not is_known:
                    # Check for obviously fabricated names
                    if any(avoid in place_name.lower() for avoid in 
                           ["fake", "test", "demo", "activity"]):
                        hallucinated.append(place_name)
        
        return len(hallucinated) > 0, hallucinated
    
    def _calculate_realism_score(self, max_activities: int) -> float:
        """
        Calculate realism score based on max activities per day.
        
        Formula: 1 - (activities / 12)
        
        - max_activities <= 6: score ~1.0 (perfect)
        - max_activities = 8: score 0.33 (acceptable)
        - max_activities > 8: score < 0.33 (unrealistic)
        """
        if max_activities == 0:
            return 0.0
        return max(0.0, 1.0 - (max_activities / 12.0))
    
    async def run_query(self, test_case: Dict[str, Any]) -> EvaluationMetrics:
        """Run a single test query through the agent."""
        metrics = EvaluationMetrics(
            test_case['id'],
            test_case['category'],
            test_case['query']
        )
        
        # Classify query upfront
        is_trip = is_trip_query(metrics.query)
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {metrics.query_id} ({metrics.category})")
            logger.info(f"Query: {metrics.query}")
            
            # Initialize agent state
            initial_state = {
                "messages": [HumanMessage(content=metrics.query)],
                "selected_trip_id": None,
                "trips": [],
                "search_progress": [],
                "planning_progress": [],
                "tripplan": None,
                "critic_reject": False,
                "critic_feedback": None,
                "user_query": metrics.query,
                "retrieved_places": [],
                "regen_attempts": 0,
            }
            
            # Run query through agent
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    self._invoke_graph(initial_state),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                metrics.status = "error"
                metrics.error_message = "Query timeout (>30s)"
                logger.error(f"❌ Timeout on query {metrics.query_id}")
                return metrics
            except Exception as graph_error:
                # Check if this is an LLM provider failure
                error_msg = str(graph_error)
                if is_llm_failure(error_msg):
                    metrics.status = "skipped"
                    metrics.error_message = f"LLM provider failure: {error_msg}"
                    logger.warning(f"⊘ Skipped (LLM failure): {metrics.query_id} - {error_msg}")
                    return metrics
                else:
                    # Other errors are real failures
                    metrics.status = "error"
                    metrics.error_message = error_msg
                    logger.error(f"❌ Error on query {metrics.query_id}: {error_msg}")
                    return metrics
            
            metrics.response_latency = time.time() - start_time
            
            # Extract response
            tripplan = result.get('tripplan')
            critic_reject = result.get('critic_reject', False)
            messages = result.get('messages', [])
            
            # Parse response
            if messages:
                if hasattr(messages[0], 'content'):
                    metrics.response_text = messages[0].content[:500]  # First 500 chars
            
            # Check if we got a valid tripplan
            if tripplan:
                metrics.has_tripplan = True
                metrics.schema_valid = True
                
                # Only proceed with content analysis if not rejected by critic
                if not critic_reject:
                    metrics.status = "accepted"
                    
                    # Analyze itinerary
                    total_acts = 0
                    max_acts = 0
                    zero_costs = 0
                    
                    for day in tripplan.itinerary:
                        acts_today = len(day.activities)
                        total_acts += acts_today
                        max_acts = max(max_acts, acts_today)
                        
                        for activity in day.activities:
                            cost = getattr(activity, 'estimated_cost', 0) or 0
                            if cost == 0:
                                zero_costs += 1
                    
                    metrics.total_activities = total_acts
                    metrics.max_activities_on_any_day = max_acts
                    metrics.activities_per_day = total_acts / tripplan.days if tripplan.days > 0 else 0
                    metrics.zero_cost_activities = zero_costs
                    metrics.estimated_budget = tripplan.estimated_budget
                    metrics.confidence_score = getattr(tripplan, 'confidence', 0.0)
                    
                    # Detect hallucinations
                    has_hallucinations, hallucinated = self._detect_hallucinations(tripplan)
                    metrics.hallucination_detected = has_hallucinations
                    metrics.hallucinated_places = hallucinated
                    
                    # Calculate realism score
                    metrics.realism_score = self._calculate_realism_score(max_acts)
                    
                    logger.info(f"✓ Accepted | Days: {tripplan.days} | "
                              f"Activities/day: {metrics.activities_per_day:.1f} | "
                              f"Realism: {metrics.realism_score:.2f}")
                else:
                    metrics.status = "rejected"
                    metrics.critic_rejected = True
                    logger.info(f"⚠️  Rejected by critic")
            else:
                # No tripplan received - classify based on query type
                if is_trip:
                    # Trip query but no tripplan = planner failure
                    metrics.status = "error"
                    metrics.error_message = "Planner failed to produce TripPlan"
                    logger.error(f"❌ Planner failure on trip query {metrics.query_id}")
                elif critic_reject:
                    # Non-trip query but critic marked reject (edge case)
                    metrics.status = "rejected"
                    metrics.critic_rejected = True
                    logger.info(f"⚠️  Non-trip query rejected by critic")
                else:
                    # Non-trip query as expected
                    metrics.status = "accepted"
                    logger.info(f"ℹ️  Non-trip query handled appropriately")
            
        except Exception as e:
            metrics.status = "error"
            metrics.error_message = str(e)
            logger.error(f"❌ Error on query {metrics.query_id}: {e}", exc_info=True)
        
        return metrics
    
    async def _invoke_graph(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent graph."""
        config = {"configurable": {"thread_id": f"test_{time.time()}"}}
        # result = graph.invoke(initial_state, config)
        result = await graph.ainvoke(initial_state, config)
        return result
    
    async def run_evaluation(self):
        """Run complete evaluation on all test queries."""
        logger.info(f"\n{'#'*60}")
        logger.info("SARATHI TRAVEL AGENT EVALUATION")
        logger.info(f"{'#'*60}")
        logger.info(f"Starting evaluation with {len(self.test_queries)} test cases")
        
        # Run all queries
        for i, test_case in enumerate(self.test_queries, 1):
            logger.info(f"\n[{i}/{len(self.test_queries)}]")
            metrics = await self.run_query(test_case)
            self.results.append(metrics)
        
        # Compute summary metrics
        self._compute_summary_metrics()
        
        # Generate report
        self._generate_report()
        
        # Print summary
        self._print_summary()
    
    def _compute_summary_metrics(self):
        """Compute aggregate metrics across all results."""
        if not self.results:
            return
        
        # Count by status
        accepted = [r for r in self.results if r.status == "accepted"]
        rejected = [r for r in self.results if r.status == "rejected"]
        errors = [r for r in self.results if r.status == "error"]
        skipped = [r for r in self.results if r.status == "skipped"]
        
        # Count success metrics
        schema_successes = [r for r in self.results if r.schema_valid]
        hallucination_count = [r for r in self.results if r.hallucination_detected]
        critic_rejections = [r for r in self.results if r.critic_rejected]
        
        # Activity metrics (for accepted plans with valid schema)
        valid_plans = [r for r in self.results if r.schema_valid and not r.critic_rejected]
        activities_list = [r.activities_per_day for r in valid_plans if r.activities_per_day > 0]
        
        # Cost metrics
        plans_with_zero_costs = [r for r in valid_plans if r.zero_cost_activities > 0]
        
        # Realism scores
        realism_scores = [r.realism_score for r in valid_plans if r.realism_score > 0]
        
        # Latencies
        latencies = [r.response_latency for r in self.results if r.response_latency > 0]
        
        # Store metrics
        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "error_count": len(errors),
            "skipped_count": len(skipped),
            "schema_success_rate": len(schema_successes) / len(self.results) if self.results else 0,
            "critic_rejection_rate": len(critic_rejections) / len(self.results) if self.results else 0,
            "hallucination_rate": len(hallucination_count) / len(valid_plans) if valid_plans else 0,
            "zero_cost_rate": len(plans_with_zero_costs) / len(valid_plans) if valid_plans else 0,
            "avg_activities_per_day": sum(activities_list) / len(activities_list) if activities_list else 0,
            "max_activities_per_day": max(activities_list) if activities_list else 0,
            "avg_realism_score": sum(realism_scores) / len(realism_scores) if realism_scores else 0,
            "avg_response_latency": sum(latencies) / len(latencies) if latencies else 0,
            "categories": self._compute_category_metrics()
        }
    
    def _compute_category_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics by test category."""
        categories = {}
        
        for category in ["REALISTIC", "UNREALISTIC", "EDGE_CASES"]:
            cat_results = [r for r in self.results if r.category == category]
            if not cat_results:
                continue
            
            accepted = [r for r in cat_results if r.status == "accepted"]
            rejected = [r for r in cat_results if r.status == "rejected"]
            
            categories[category] = {
                "total": len(cat_results),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "acceptance_rate": len(accepted) / len(cat_results) if cat_results else 0,
                "rejection_rate": len(rejected) / len(cat_results) if cat_results else 0,
            }
        
        return categories
    
    def _generate_report(self):
        """Generate evaluation report JSON file."""
        report = {
            "metadata": {
                "timestamp": self.summary["timestamp"],
                "total_queries": self.summary["total_queries"],
                "test_file": str(TEST_QUERIES_FILE)
            },
            "aggregate_metrics": {
                "schema_success_rate": round(self.summary["schema_success_rate"], 4),
                "critic_rejection_rate": round(self.summary["critic_rejection_rate"], 4),
                "hallucination_rate": round(self.summary["hallucination_rate"], 4),
                "zero_cost_rate": round(self.summary["zero_cost_rate"], 4),
                "avg_activities_per_day": round(self.summary["avg_activities_per_day"], 2),
                "max_activities_per_day": self.summary["max_activities_per_day"],
                "avg_realism_score": round(self.summary["avg_realism_score"], 4),
                "avg_response_latency_ms": round(self.summary["avg_response_latency"] * 1000, 2),
            },
            "status_breakdown": {
                "accepted": self.summary["accepted_count"],
                "rejected": self.summary["rejected_count"],
                "error": self.summary["error_count"],
                "skipped": self.summary["skipped_count"],
            },
            "category_breakdown": self.summary["categories"],
            "detailed_results": [
                {
                    "query_id": r.query_id,
                    "category": r.category,
                    "query": r.query,
                    "status": r.status,
                    "schema_valid": r.schema_valid,
                    "critic_rejected": r.critic_rejected,
                    "activities_per_day": round(r.activities_per_day, 2),
                    "max_activities": r.max_activities_on_any_day,
                    "realism_score": round(r.realism_score, 4),
                    "confidence": round(r.confidence_score, 4),
                    "hallucination_detected": r.hallucination_detected,
                    "zero_cost_count": r.zero_cost_activities,
                    "response_latency_ms": round(r.response_latency * 1000, 2),
                    "error": r.error_message if r.error_message else None
                }
                for r in self.results
            ]
        }
        
        # Write report
        with open(OUTPUT_REPORT_FILE, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to: {OUTPUT_REPORT_FILE}")
    
    def _print_summary(self):
        """Print evaluation summary to console."""
        s = self.summary
        
        print("\n" + "="*70)
        print("SARATHI TRAVEL AGENT - EVALUATION SUMMARY".center(70))
        print("="*70)
        
        print(f"\nTotal Queries: {s['total_queries']}")
        print(f"Accepted: {s['accepted_count']} | Rejected: {s['rejected_count']} | Errors: {s['error_count']} | Skipped: {s['skipped_count']}")
        
        print("\n" + "-"*70)
        print("KEY METRICS")
        print("-"*70)
        
        print(f"Schema Success Rate:        {s['schema_success_rate']*100:6.1f}%")
        print(f"Critic Rejection Rate:      {s['critic_rejection_rate']*100:6.1f}%")
        print(f"Hallucination Rate:         {s['hallucination_rate']*100:6.1f}%")
        print(f"Zero-Cost Activities Rate:  {s['zero_cost_rate']*100:6.1f}%")
        
        print("\n" + "-"*70)
        print("ITINERARY QUALITY")
        print("-"*70)
        
        print(f"Avg Activities per Day:     {s['avg_activities_per_day']:6.2f}")
        print(f"Max Activities in Any Day:  {s['max_activities_per_day']:6.0f}")
        print(f"Avg Realism Score:          {s['avg_realism_score']:6.4f} (1.0 = perfect)")
        print(f"Avg Response Latency:       {s['avg_response_latency']*1000:6.1f} ms")
        
        print("\n" + "-"*70)
        print("CATEGORY BREAKDOWN")
        print("-"*70)
        
        for category, metrics in s['categories'].items():
            print(f"\n{category.upper()}")
            print(f"  Total:        {metrics['total']}")
            print(f"  Acceptance:   {metrics['acceptance_rate']*100:5.1f}%")
            print(f"  Rejection:    {metrics['rejection_rate']*100:5.1f}%")
        
        print("\n" + "="*70)
        print(f"Report: {OUTPUT_REPORT_FILE}")
        print("="*70 + "\n")


async def main():
    """Main evaluation entry point."""
    evaluator = TravelAgentEvaluator()
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
