"""Shared train/val/test split count allocation."""

from __future__ import annotations


def allocate_split_counts(
    n: int,
    train_pct: float,
    val_pct: float,
    test_pct: float,
) -> tuple[int, int, int]:
    """Allocate *n* samples into train/val/test with largest-remainder.

    Percentages may be fractions in ``[0, 1]`` or percent values summing to ~100.
    Exact sum of returned counts is ``n``. A ``0%`` bucket always gets ``0``.
    """
    if n <= 0:
        return (0, 0, 0)

    pcts = [float(train_pct), float(val_pct), float(test_pct)]
    total = sum(pcts)
    if total <= 0:
        # Degenerate: put everything in train so sum == n
        return (n, 0, 0)

    # Accept 0–100 style inputs (e.g. 80, 10, 10)
    if total > 1.0 + 1e-6:
        pcts = [p / 100.0 for p in pcts]
        total = sum(pcts)

    # Normalize tiny float drift so quotas sum to n
    if abs(total - 1.0) > 1e-9:
        pcts = [p / total for p in pcts]

    quotas = [n * p for p in pcts]
    floors = [int(q) for q in quotas]
    # Force exact 0% → 0 (before remainder distribution)
    for i, p in enumerate(pcts):
        if abs(p) < 1e-12:
            floors[i] = 0

    remainders = [
        (quotas[i] - floors[i], -i, i)  # -i = stable tie-break: earlier bucket wins
        for i in range(3)
        if abs(pcts[i]) >= 1e-12
    ]
    remainders.sort(reverse=True)

    leftover = n - sum(floors)
    counts = list(floors)
    for _, _, i in remainders:
        if leftover <= 0:
            break
        counts[i] += 1
        leftover -= 1

    # Safety: if still leftover (all non-zero buckets exhausted somehow), dump to first non-zero pct
    if leftover > 0:
        for i, p in enumerate(pcts):
            if abs(p) >= 1e-12:
                counts[i] += leftover
                leftover = 0
                break

    assert sum(counts) == n
    assert all(c >= 0 for c in counts)
    for i, p in enumerate(pcts):
        if abs(p) < 1e-12:
            assert counts[i] == 0

    return (counts[0], counts[1], counts[2])
