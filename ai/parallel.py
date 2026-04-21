"""Parallel inference â race multiple providers."""

import asyncio
from typing import List, Tuple
from ai.providers.base import BaseProvider
from ai.router import SmartRouter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ParallelInference:
    def __init__(self, config, router: SmartRouter):
        self.router = router
        self.max_n = config.get("ai.parallel.max_concurrent", 3)
        self.strategy = config.get("ai.parallel.return_strategy", "fastest")

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
