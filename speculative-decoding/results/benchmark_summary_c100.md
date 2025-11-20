## Benchmark Comparison: Qwen3-32B with vs without Speculative Decoding (c100-t0.0)

**Configuration:** Both tests used 80 requests, max concurrency 100, temperature 0.0, on Qwen/Qwen3-32B model

### Key Performance Metrics

| Metric | No Spec Decoding (nosd) | With Spec Decoding (sd) | **Improvement** |
|--------|------------------------|-------------------------|-----------------|
| **Benchmark Duration** | 101.34s | 76.82s | **1.32× faster (24% faster)** |
| **Output Token Throughput** | 201.91 tok/s | 266.23 tok/s | **1.32× faster (32% gain)** |
| **Request Throughput** | 0.79 req/s | 1.04 req/s | **1.32× faster (32% gain)** |
| **Total Token Throughput** | 261.89 tok/s | 345.35 tok/s | **1.32× faster (32% gain)** |
| **Peak Output Throughput** | 240.00 tok/s | 160.00 tok/s | 33% lower peak |

### Latency Metrics

| Metric | No Spec Decoding (nosd) | With Spec Decoding (sd) | **Improvement** |
|--------|------------------------|-------------------------|-----------------|
| **Mean TTFT** | 5659.49 ms | 6173.26 ms | 9% slower |
| **Median TTFT** | 5704.09 ms | 6248.41 ms | 10% slower |
| **P99 TTFT** | 8330.00 ms | 9995.82 ms | 20% slower |
| **Mean TPOT** | 373.79 ms | 235.20 ms | **1.59× faster (37% gain)** |
| **Median TPOT** | 373.65 ms | 235.16 ms | **1.59× faster (37% gain)** |
| **P99 TPOT** | 384.58 ms | 278.10 ms | **1.38× faster (28% gain)** |
| **Mean ITL** | 373.79 ms | 654.03 ms | 75% slower |
| **Median ITL** | 364.72 ms | 639.48 ms | 75% slower |

### Summary

**At high concurrency (100), speculative decoding provides moderate but consistent gains:**
- **32% improvement** in overall throughput and 24% reduction in total benchmark time
- **37% faster** time per output token (mean/median TPOT)
- Benchmark completed in 77 seconds vs 101 seconds (saved 24 seconds)

**Trade-offs at high concurrency:**
- Time to First Token increases by ~10% (mean/median)
- Inter-token latency increases significantly (75%), likely due to speculative decoding overhead under high concurrent load
- Lower peak output throughput (160 vs 240 tok/s)

**Compared to low concurrency (c1):**
- The benefits of speculative decoding are less dramatic at high concurrency (1.32× vs 2.20× speedup)
- This suggests speculative decoding is more effective for sequential/low-concurrency workloads than highly parallel ones
