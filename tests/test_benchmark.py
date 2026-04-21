"""Benchmark all providers."""

import asyncio
import time

from rich.console import Console
from rich.table import Table

from ai.providers import init_providers

console = Console()


async def run_benchmark(config):
    console.print("\n[bold cyan]OpenAssist AI v3.0 - Provider Benchmark[/bold cyan]\n")
    providers = init_providers(config)
    if not providers:
        console.print("[red]No providers![/red]")
        return

    prompt = "Explain recursion in 3 sentences."
    results = []

    for name, provider in providers.items():
        console.print(f"  Testing [yellow]{name}[/yellow]...", end=" ")
        try:
            t0 = time.time()
            response = await provider.generate("Be concise.", prompt, "fast")
            dt = time.time() - t0
            words = len(response.split())
            tps = words / dt if dt > 0 else 0
            results.append(
                {
                    "name": name,
                    "model": provider.get_model("fast"),
                    "time": dt,
                    "words": words,
                    "tps": tps,
                    "ok": True,
                    "spd": provider.speed,
                    "qual": provider.quality,
                }
            )
            console.print(f"[green]{dt:.2f}s ({tps:.0f} w/s)[/green]")
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "model": provider.get_model("fast"),
                    "time": 0,
                    "words": 0,
                    "tps": 0,
                    "ok": False,
                    "spd": provider.speed,
                    "qual": provider.quality,
                    "err": str(exc)[:40],
                }
            )
            console.print(f"[red]FAIL: {exc}[/red]")

    results.sort(key=lambda item: item["tps"], reverse=True)

    table = Table(title="\nResults (by speed)")
    table.add_column("#", width=3)
    table.add_column("Provider", style="cyan", width=12)
    table.add_column("Model", width=30)
    table.add_column("Time", justify="right", width=7)
    table.add_column("Words/s", justify="right", style="green", width=8)
    table.add_column("Spd", width=4)
    table.add_column("Qual", width=4)
    table.add_column("Status", width=10)

    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            result["name"],
            result["model"][:28],
            f'{result["time"]:.2f}s',
            f'{result["tps"]:.0f}',
            f'{result["spd"]}/10',
            f'{result["qual"]}/10',
            "OK" if result["ok"] else f'ERR {result.get("err", "")[:20]}',
        )
    console.print(table)

    ok = [result for result in results if result["ok"]]
    if ok:
        fastest = ok[0]
        console.print(
            f"\nFastest: [bold green]{fastest['name']}[/] ({fastest['tps']:.0f} words/s)"
        )
        best_quality = max(ok, key=lambda item: item["qual"])
        console.print(
            f"Best Quality: [bold blue]{best_quality['name']}[/] (quality={best_quality['qual']}/10)"
        )
