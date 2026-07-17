"""Small benchmark helpers for CUDA-event based microbenchmarks."""

from __future__ import annotations

import csv
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Serializable latency summary plus evidence metadata for one case."""

    name: str
    mean_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    metadata: dict

    def as_row(self) -> dict:
        """Return a stable CSV-ready mapping with formatted latency fields."""
        row = {
            "name": self.name,
            "mean_ms": f"{self.mean_ms:.6f}",
            "p50_ms": f"{self.p50_ms:.6f}",
            "p90_ms": f"{self.p90_ms:.6f}",
            "min_ms": f"{self.min_ms:.6f}",
            "max_ms": f"{self.max_ms:.6f}",
            "repeats": self.repeats,
        }
        row.update({key: str(value) for key, value in self.metadata.items()})
        return row


def git_commit(root=None):
    """Return the current short Git commit or fail before evidence is written."""
    if root is None:
        root = Path.cwd()
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise RuntimeError("benchmark evidence requires a readable Git commit")
    return commit


def percentile(values, q):
    """Return percentile q in [0, 100] using nearest-rank interpolation."""
    if not values:
        raise ValueError("values must be non-empty")
    if q < 0 or q > 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    index = round((q / 100) * (len(ordered) - 1))
    return ordered[index]


def summarize_latencies(name, latencies_ms, metadata=None):
    """Summarize a list of latency measurements in milliseconds."""
    if not latencies_ms:
        raise ValueError("latencies_ms must be non-empty")
    if metadata is None:
        metadata = {}
    return BenchmarkResult(
        name=name,
        mean_ms=statistics.fmean(latencies_ms),
        p50_ms=percentile(latencies_ms, 50),
        p90_ms=percentile(latencies_ms, 90),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        repeats=len(latencies_ms),
        metadata=dict(metadata),
    )


def cuda_event_timer(fn, warmup=20, repeat=100):
    """Measure a CUDA callable with torch.cuda.Event.

    The callable should launch the GPU work and return quickly. Synchronization is
    handled by this helper.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda_event_timer")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    return latencies_ms


def benchmark_case(name, fn, warmup=20, repeat=100, metadata=None):
    """Run and summarize a benchmark case."""
    latencies = cuda_event_timer(fn, warmup=warmup, repeat=repeat)
    return summarize_latencies(name, latencies, metadata=metadata)


def write_csv(results, path):
    """Write benchmark results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_row() for result in results]
    if not rows:
        raise ValueError("results must be non-empty")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
