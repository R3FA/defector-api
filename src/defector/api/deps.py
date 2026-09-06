"""Shared dependencies.

Inference is CPU-bound and PyTorch already spreads one call across 12 threads.
anyio's default thread limiter is 40, so without a bound, forty concurrent
requests would each try to claim the whole CPU and every one of them would slow
down. The semaphore caps how many run at once; the queue behind it is what
keeps latency predictable.
"""

import asyncio

from defector.settings import settings

inference_slots = asyncio.Semaphore(settings.inference_concurrency)
