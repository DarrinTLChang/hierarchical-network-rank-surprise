"""Log probabilities for the Rank-Surprise discrete-uniform rank model.

For q independent ranks in 1..N, the exact CDF numerator is the integer count
sum_k (-1)^k C(q,k) C(u-k*N,q). Integer arithmetic avoids cancellation between
large alternating floating-point terms. The historical q=30 transition to a
Gaussian approximation is retained; support endpoints are exact for every q.

Model: Gourevitch & Eggermont (2007), J Neurosci Methods 160, 349-358,
doi:10.1016/j.jneumeth.2006.09.024. Long-candidate interior probabilities remain
approximations, particularly in extreme tails; no underflow floor is imposed.
"""
from __future__ import annotations

from functools import lru_cache
import math
from operator import index

from scipy.special import log_ndtr


EXACT_RANK_COUNT_LIMIT = 30


@lru_cache(maxsize=16384, typed=True)
def rank_sum_logcdf(rank_sum: int, count: int, rank_count: int) -> float:
    """Return log P(sum of ``count`` uniform 1..``rank_count`` ranks <= sum).

    All arguments are integers, counts positive, and the sum must be inside its
    attainable support. Invalid candidate sums are errors, never apparent bursts.
    """
    try:
        u, q, n = index(rank_sum), index(count), index(rank_count)
    except TypeError as exc:
        raise ValueError("Rank sum and counts must be integers") from exc
    if q < 1 or n < 1:
        raise ValueError("Rank count and number of summed ranks must be positive")
    if u < q or u > q * n:
        raise ValueError(f"Rank sum {u} is outside attainable support [{q}, {q * n}]")
    if n == 1 or u == q * n:
        return 0.0
    if u == q:
        return -q * math.log(n)

    if q < EXACT_RANK_COUNT_LIMIT:
        numerator = sum(
            (-1 if k % 2 else 1) * math.comb(q, k) * math.comb(u - k * n, q)
            for k in range((u - q) // n + 1)
        )
        denominator = n ** q
        if not 0 < numerator <= denominator:
            raise ArithmeticError("Exact rank-sum count is outside its valid probability range")
        if numerator > denominator // 2:
            # Accurate even when the CDF is extremely close to one.
            return math.log1p(-(denominator - numerator) / denominator)
        return math.log(numerator) - q * math.log(n)

    mean = q * (n + 1) / 2.0
    variance = q * (n * n - 1) / 12.0
    value = float(log_ndtr((u - mean) / math.sqrt(variance)))
    if not math.isfinite(value) or value > 0:
        raise ArithmeticError("Gaussian rank-sum approximation produced an invalid log probability")
    return value
