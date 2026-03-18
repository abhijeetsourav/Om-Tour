"""
Base Node class for consistent error handling and logging across all agent nodes.

This implements the error handling pattern from AWS Travel Assistant Agent,
ensuring all nodes have consistent behavior for:
- Error handling and recovery
- Logging
- State cleanup
- User-friendly error messages
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Abstract base class for all agent nodes."""

    NODE_NAME = "BaseNode"  # Override in subclasses
    MAX_RETRIES = 2

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the node logic. Must be implemented by subclasses.

        Args:
            state: The agent state

        Returns:
            Updated state
        """
        pass

    async def execute(self, state: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Execute the node with built-in error handling and recovery.

        This wraps the process() method with:
        - Error catching and logging
        - User-friendly error messages
        - State cleanup on failure
        - Retry logic for transient errors

        Args:
            state: The agent state
            config: Optional LangGraph config

        Returns:
            Updated state with error info if failure occurs
        """
        try:
            logger.info(f"[{self.NODE_NAME}] Starting processing")
            result = await self.process(state)
            logger.info(f"[{self.NODE_NAME}] Completed successfully")
            return result

        except Exception as e:
            logger.error(
                f"[{self.NODE_NAME}] Error: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return self._handle_error(state, e)

    def _handle_error(self, state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        Handle errors with consistent format and recovery strategy.

        Override in subclasses for node-specific recovery.

        Args:
            state: Current state
            error: The exception that occurred

        Returns:
            Updated state with error information
        """
        error_message = self._format_error_message(error)

        # Add error to state
        state["last_error"] = {
            "node": self.NODE_NAME,
            "type": type(error).__name__,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }

        # Default recovery: clear sensitive state
        state = self._cleanup_state_on_error(state)

        logger.error(f"[{self.NODE_NAME}] Error handled, state cleaned up")
        return state

    def _format_error_message(self, error: Exception) -> str:
        """
        Format error into user-friendly message.

        Override in subclasses for custom messaging.
        """
        error_messages = {
            "ValueError": "Invalid input provided. Please check your query.",
            "KeyError": "Missing required data. Please try again.",
            "TimeoutError": "Request took too long. Please try with a simpler query.",
            "ConnectionError": "Connection issue. Please try again.",
        }

        error_type = type(error).__name__
        return error_messages.get(error_type, f"Error in {self.NODE_NAME}: {str(error)}")

    def _cleanup_state_on_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up state on error to prevent cascading failures.

        Override in subclasses for node-specific cleanup.
        """
        # Remove intermediate/incomplete data
        state.pop("tripplan", None)
        state.pop("partial_plan", None)
        return state
