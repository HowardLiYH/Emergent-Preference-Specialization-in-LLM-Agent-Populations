"""
Batch processing for LLM calls.

Enables efficient parallel processing of multiple agent responses.
"""

import asyncio
from typing import List, Dict, Optional, Any, Callable
import logging
from dataclasses import dataclass
import time

from .llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Single item in a batch."""
    id: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    """Result of a batch item."""
    id: str
    response: Optional[LLMResponse]
    error: Optional[str] = None
    latency_ms: float = 0


class BatchProcessor:
    """
    Batch processor for efficient LLM calls.

    Features:
    - Parallel processing with async
    - Rate limiting
    - Error handling
    - Progress tracking
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_concurrent: int = 5,
        delay_between: float = 0.1
    ):
        """
        Initialize batch processor.

        Args:
            llm_client: LLM client for API calls
            max_concurrent: Maximum concurrent requests
            delay_between: Delay between requests (seconds)
        """
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent
        self.delay_between = delay_between

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.total_tokens = 0

    async def _process_item(
        self,
        item: BatchItem,
        semaphore: asyncio.Semaphore
    ) -> BatchResult:
        """Process a single batch item."""
        async with semaphore:
            start_time = time.time()

            try:
                response = await self.llm_client.generate_async(item.prompt)
                latency = (time.time() - start_time) * 1000

                self.total_processed += 1
                self.total_tokens += response.tokens_used

                return BatchResult(
                    id=item.id,
                    response=response,
                    latency_ms=latency
                )

            except Exception as e:
                latency = (time.time() - start_time) * 1000
                self.total_errors += 1

                logger.warning(f"Batch item {item.id} failed: {e}")

                return BatchResult(
                    id=item.id,
                    response=None,
                    error=str(e),
                    latency_ms=latency
                )

    async def process_batch_async(
        self,
        items: List[BatchItem],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """
        Process a batch of items asynchronously.

        Args:
            items: List of BatchItem to process
            progress_callback: Called with (completed, total) after each item

        Returns:
            List of BatchResult
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = []
        for i, item in enumerate(items):
            # Add delay between requests
            if i > 0 and self.delay_between > 0:
                await asyncio.sleep(self.delay_between)

            task = asyncio.create_task(
                self._process_item(item, semaphore)
            )
            tasks.append(task)

        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(items))

        return results

    def process_batch(
        self,
        items: List[BatchItem],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """
        Process a batch of items (sync wrapper).

        Args:
            items: List of BatchItem to process
            progress_callback: Called with (completed, total) after each item

        Returns:
            List of BatchResult
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.process_batch_async(items, progress_callback)
            )
        finally:
            loop.close()

    def process_prompts(
        self,
        prompts: List[str],
        ids: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Process a list of prompts.

        Args:
            prompts: List of prompt strings
            ids: Optional list of IDs (auto-generated if not provided)

        Returns:
            Dict mapping ID to response text
        """
        if ids is None:
            ids = [f"prompt_{i}" for i in range(len(prompts))]

        items = [
            BatchItem(id=id_, prompt=prompt)
            for id_, prompt in zip(ids, prompts)
        ]

        results = self.process_batch(items)

        return {
            r.id: r.response.text if r.response else ""
            for r in results
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'total_tokens': self.total_tokens,
            'error_rate': (
                self.total_errors / max(1, self.total_processed + self.total_errors)
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_processed = 0
        self.total_errors = 0
        self.total_tokens = 0


async def competition_round_batched(
    agents: List[Any],
    task: str,
    llm_client: LLMClient
) -> Dict[str, str]:
    """
    Run a competition round with batched LLM calls.

    Args:
        agents: List of Agent objects
        task: Task description
        llm_client: LLM client

    Returns:
        Dict mapping agent ID to response
    """
    processor = BatchProcessor(llm_client)

    items = []
    for agent in agents:
        prompt = agent.construct_prompt(task) if hasattr(agent, 'construct_prompt') else task
        items.append(BatchItem(id=agent.id, prompt=prompt))

    results = await processor.process_batch_async(items)

    return {
        r.id: r.response.text if r.response else ""
        for r in results
    }
