"""Small dependency-free helpers used while reporting inference benchmarks."""


def percentile(values: list[float], percentage: float) -> float:
    """Compute a nearest-rank percentile for a non-empty latency sample."""

    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sample.")
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentage))
    return ordered[index]
