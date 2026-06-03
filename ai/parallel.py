"""Parallel inference â race multiple providers."""

import asyncio
from typing import List, Tuple, AsyncGenerator
from ai.providers.base import BaseProvider
from ai.router import SmartRouter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ParallelInference:
    def __init__(self, config, router: SmartRouter):
        self.router = router
        self.max_n = config.get("ai.parallel.max_concurrent", 3)
        self.strategy = config.get("ai.parallel.return_strategy", "fastest")

    async def generate_stream(
        self, system: str, user: str, task: str = "general", tier: str = None
    ) -> AsyncGenerator[Tuple[str, str], None]:
        targets = []
        exclude = []
        for _ in range(self.max_n):
            p, t = self.router.select(task=task, tier=tier, exclude=exclude)
            if p:
                targets.append((p, t))
                exclude.append(p.name)
        if not targets:
            raise Exception("No providers for parallel streaming")

        logger.info(f"🔄 Parallel Streaming → {[p.name for p, _ in targets]}")

        async def read_first_chunk(p, t):
            try:
                stream = p.generate_stream(system, user, t)
                first_chunk = await stream.__anext__()
                return p.name, first_chunk, stream
            except (Exception, StopAsyncIteration) as exc:
                logger.debug(f"Parallel streaming provider {p.name} failed: {exc}")
                return p.name, None, None

        tasks = [asyncio.create_task(read_first_chunk(p, t)) for p, t in targets]

        winner_name = None
        winner_stream = None
        first_chunk = None
        # H8 FIX: Collect loser streams so we can aclose() them to
        # release their underlying HTTP connections.
        loser_streams = []

        try:
            for coro in asyncio.as_completed(tasks):
                name, chunk, stream = await coro
                if chunk is not None:
                    winner_name = name
                    winner_stream = stream
                    first_chunk = chunk
                    break

            for task_obj in tasks:
                if not task_obj.done():
                    task_obj.cancel()

            # H8 FIX: Collect completed loser streams for cleanup.
            for task_obj in tasks:
                if task_obj.done() and not task_obj.cancelled():
                    try:
                        _name, _chunk, _stream = task_obj.result()
                        if _stream is not None and _stream is not winner_stream:
                            loser_streams.append(_stream)
                    except Exception:
                        pass

            if winner_name is None:
                raise Exception("All parallel providers failed to produce a first chunk")

            logger.info(f"🏆 Parallel Winner: {winner_name}")
            yield winner_name, first_chunk

            async for chunk in winner_stream:
                yield winner_name, chunk

        finally:
            for task_obj in tasks:
                if not task_obj.done():
                    task_obj.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # H8 FIX: Properly close all loser async generator streams.
            for stream in loser_streams:
                try:
                    await stream.aclose()
                except Exception:
                    pass

    async def generate(self, system: str, user: str, task: str = "general", tier: str = None) -> str:
        targets = []
        exclude = []
        for _ in range(self.max_n):
            p, t = self.router.select(task=task, tier=tier, exclude=exclude)
            if p:
                targets.append((p, t))
                exclude.append(p.name)
        if not targets:
            raise Exception("No providers for parallel")

        logger.info(f"ð Parallel â {[p.name for p, _ in targets]}")

        async def query(p, t):
            try:
                return p.name, await p.generate(system, user, t)
            except Exception as exc:
                logger.debug(f"Parallel provider {p.name} failed: {exc}")
                return p.name, None

        tasks = [asyncio.create_task(query(p, t)) for p, t in targets]

        try:
            if self.strategy == "fastest":
                for coro in asyncio.as_completed(tasks):
                    name, result = await coro
                    if result:
                        logger.info(f"🏆 Winner: {name}")
                        return result
            else:
                results = await asyncio.gather(*tasks)
                valid = [(n, r) for n, r in results if r]
                if valid:
                    return max(valid, key=lambda x: len(x[1]))[1]

            raise Exception("All parallel providers failed")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
