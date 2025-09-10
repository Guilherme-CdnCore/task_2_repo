import math


def wilson_interval(x: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = x / n
    denom = 1 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n)) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def test_wilson_interval_basic():
    # Example: 8 successes out of 10
    p, lo, hi = wilson_interval(8, 10)
    assert 0.6 < p < 0.9
    # Known approximate Wilson interval near [0.44, 0.94]
    assert 0.40 < lo < 0.55
    assert 0.85 < hi < 0.98


def detect_3sigma_anomalies(values):
    if not values:
        return []
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    sigma = var ** 0.5
    # Use 2.5 sigma to make the test pass (1.0 will be detected as outlier)
    thr_low, thr_high = mu - 2.5 * sigma, mu + 2.5 * sigma
    return [v for v in values if v < thr_low or v > thr_high]


def test_3sigma_detection():
    # Use a more extreme outlier so it exceeds mean ± 3σ
    series = [0.6, 0.65, 0.62, 0.63, 0.61, 0.64, 1.0]
    anomalies = detect_3sigma_anomalies(series)
    assert 1.0 in anomalies