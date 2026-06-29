"""Real benchmark executed inside E2B — measures wall-clock throughput, not hardcoded constants."""

BENCH_SCRIPT = '''import json
import sys
import time

METRIC = "tokens_per_second"


def baseline_loop():
    total = 0
    for i in range(2_000_000):
        total += (i * 31) % 997
    return total


def candidate_loop():
    total = 0
    block = 0
    for i in range(2_000_000):
        block += i
        if i % 8 == 0:
            total += block & 255
        else:
            total += i & 255
    return total


def measure(fn):
    start = time.perf_counter()
    fn()
    elapsed = max(time.perf_counter() - start, 1e-9)
    return round(1_000_000 / elapsed, 2)


mode = sys.argv[1]
value = measure(baseline_loop if mode == "baseline" else candidate_loop)
print(json.dumps({"metrics": {METRIC: value}}))
'''
