"""Benchmark all providers."""

import asyncio
import time
from rich.console import Console
from rich.table import Table
from ai.providers import init_providers

console = Console()


async def run_benchmark(config):
    console.print("\n[bold cyan]ð OpenAssist AI v3.0 â Provider Benchmark[/bold cyan]\n")
    providers = init_providers(config)
    if not providers:
        console.print("[red]No providers![/red]")
        return

    prompt = "Explain recursion in 3 sentences."
    results = []

    for name, p in providers.items():
        console.print(f"  Testing [yellow]{name}[/yellow]...", end=" ")
        try:
            t0 = time.time()
            r = await p.generate("Be concise.", prompt, "fast")
            dt = time.time() - t0
            words = len(r.split())
            tps = words / dt if dt > 0 else 0
            results.append({"name": name, "model": p.get_model("fast"), "time": dt,
                          "words": words, "tps": tps, "ok": True, "spd": p.speed, "qual": p.quality})
            console.print(f"[green]{dt:.2f}s ({tps:.0f} w/s)[/green]")
        except Exception as e:
            results.append({"name": name, "model": p.get_model("fast"), "time": 0,
                          "words": 0, "tps": 0, "ok": False, "spd": p.speed, "qual": p.quality,
                          "err": str(e)[:40]})
            console.print(f"[red]FAIL: {e}[/red]")

    results.sort(key=lambda x: x["tps"], reverse=True)

    t = Table(title="\nð Results (by speed)")
    t.add_column("#", width=3)
    t.add_column("Provider", style="cyan", width=12)
    t.add_column("Model", width=30)
    t.add_column("Time", justify="right", width=7)
    t.add_column("Words/s", justify="right", style="green", width=8)
    t.add_column("Spd", width=4)
    t.add_column("Qual", width=4)
    t.add_column("Status", width=10)

    for i, r in enumerate(results):
        t.add_row(
            str(i+1), r["name"], r["model"][:28],
            f'{r["time"]:.2f}s', f'{r["tps"]:.0f}',
            f'{r["spd"]}/10', f'{r["qual"]}/10',
            "â" if r["ok"] else f'â {r.get("err", "")[:20]}'
        )
    console.print(t)

    ok = [r for r in results if r["ok"]]
    if ok:
        f = ok[0]
        console.print(f"\nâ¡ Fastest: [bold green]{f['name']}[/] ({f['tps']:.0f} words/s)")
        b = max(ok, key=lambda x: x["qual"])
        console.print(f"ð Best Quality: [bold blue]{b['name']}[/] (quality={b['qual']}/10)")