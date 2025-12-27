"""Token bucket rate limiter for API calls."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket rate limiter implementation.

    Allows bursting up to the bucket capacity while maintaining
    a sustainable rate over time.
    """

    calls_per_minute: int
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize token bucket with full capacity."""
        self._tokens = float(self.calls_per_minute)
        self._last_update = time.monotonic()

    @property
    def tokens_per_second(self) -> float:
        """Calculate token replenishment rate."""
        return self.calls_per_minute / 60.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.calls_per_minute,
            self._tokens + elapsed * self.tokens_per_second,
        )
        self._last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            self._refill()

            wait_time = 0.0
            if self._tokens < tokens:
                # Calculate how long to wait for enough tokens
                deficit = tokens - self._tokens
                wait_time = deficit / self.tokens_per_second
                await asyncio.sleep(wait_time)
                self._refill()

            self._tokens -= tokens
            return wait_time

    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        self._refill()
        return self._tokens

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        self._tokens = float(self.calls_per_minute)
        self._last_update = time.monotonic()
