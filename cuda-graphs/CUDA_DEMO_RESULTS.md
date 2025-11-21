# CUDA Graph Demo Results

## System Information

- **GPU:** NVIDIA GB10
- **Compute Capability:** 12.1
- **Total Global Memory:** 128.53 GB
- **Multiprocessors:** 48
- **Max Threads per Block:** 1024

## Benchmark Results

### Test 1: Small Arrays (1M elements, 100 iterations)

| Run | Traditional (ms) | CUDA Graph (ms) | Speedup | Status |
|-----|-----------------|-----------------|---------|--------|
| 1 | 2.461 | 2.492 | 0.99x | ✓ |
| 2 | 2.471 | 2.074 | 1.19x | ✓ |
| 3 | 2.472 | 2.072 | 1.19x | ✓ |
| 4 | 2.468 | 2.072 | 1.19x | ✓ |
| 5 | 2.499 | 2.070 | 1.21x | ✓ |
| 6 | 2.466 | 2.067 | 1.19x | ✓ |
| 7 | 2.459 | 2.068 | 1.19x | ✓ |
| 8 | 2.465 | 2.068 | 1.19x | ✓ |
| 9 | 2.473 | 2.072 | 1.19x | ✓ |
| 10 | 2.465 | 2.067 | 1.19x | ✓ |

**Summary:**
- Average Traditional: **2.470 ms**
- Average CUDA Graph: **2.112 ms**
- Average Speedup: **1.17x**
- Time Saved: **0.358 ms (14.5% reduction)** ⭐

---

### Test 2: Medium Arrays (4M elements, 50 iterations)

| Run | Traditional (ms) | CUDA Graph (ms) | Speedup | Status |
|-----|-----------------|-----------------|---------|--------|
| 1 | 12.832 | 12.268 | 1.05x | ✓ |
| 2 | 12.823 | 12.283 | 1.04x | ✓ |
| 3 | 12.818 | 12.303 | 1.04x | ✓ |
| 4 | 12.920 | 12.318 | 1.05x | ✓ |
| 5 | 12.545 | 12.262 | 1.02x | ✓ |
| 6 | 12.849 | 12.315 | 1.04x | ✓ |
| 7 | 13.026 | 12.388 | 1.05x | ✓ |
| 8 | 12.832 | 13.296 | 0.97x | ✓ |
| 9 | 12.841 | 12.252 | 1.05x | ✓ |
| 10 | 12.551 | 12.260 | 1.02x | ✓ |

**Summary:**
- Average Traditional: **12.804 ms**
- Average CUDA Graph: **12.394 ms**
- Average Speedup: **1.03x**
- Time Saved: **0.409 ms (3.2% reduction)**

---

### Test 3: Large Arrays (16M elements, 20 iterations)

| Run | Traditional (ms) | CUDA Graph (ms) | Speedup | Status |
|-----|-----------------|-----------------|---------|--------|
| 1 | 30.923 | 31.777 | 0.97x | ✓ |
| 2 | 31.668 | 34.358 | 0.92x | ✓ |
| 3 | 31.084 | 30.630 | 1.01x | ✓ |
| 4 | 31.066 | 30.746 | 1.01x | ✓ |
| 5 | 31.453 | 30.995 | 1.01x | ✓ |
| 6 | 31.198 | 31.060 | 1.00x | ✓ |
| 7 | 31.025 | 30.935 | 1.00x | ✓ |
| 8 | 30.949 | 30.638 | 1.01x | ✓ |
| 9 | 31.198 | 30.990 | 1.01x | ✓ |
| 10 | 31.014 | 31.060 | 1.00x | ✓ |

**Summary:**
- Average Traditional: **31.158 ms**
- Average CUDA Graph: **31.319 ms**
- Average Speedup: **0.99x**
- Time Saved: **-0.161 ms (-0.5% reduction)**

---

## Overall Analysis

### Performance Summary

| Array Size | Iterations | Traditional (ms) | CUDA Graph (ms) | Speedup | Improvement |
|-----------|-----------|-----------------|-----------------|---------|-------------|
| 1M elements | 100 | 2.470 | 2.112 | 1.17x | **+14.5%** ⭐ |
| 4M elements | 50 | 12.804 | 12.394 | 1.03x | +3.2% |
| 16M elements | 20 | 31.158 | 31.319 | 0.99x | -0.5% |

### Key Findings

✓ **Best performance gains with small arrays and many iterations**
- 14.5% improvement for 1M elements with 100 iterations
- CPU launch overhead is significant relative to kernel execution time

✓ **Diminishing returns with larger workloads**
- Only 3.2% improvement for 4M elements
- Negligible/negative impact for 16M elements
- Kernel execution time dominates, overshadowing launch overhead

✓ **All results verified**
- All 30 benchmark runs produced identical outputs (✓)
- CUDA graphs maintain numerical correctness

## Key Takeaways

1. **CUDA graphs eliminate per-kernel CPU launch overhead**
   - Most effective when launch overhead is significant portion of total time

2. **Benefits increase with more kernel launches in the sequence**
   - 100 iterations × 3 kernels = 300 launches showed best improvement

3. **Graph capture happens once; replay is extremely fast**
   - One-time capture cost amortized over many executions

4. **Ideal for repetitive, fixed-structure GPU workloads**
   - Perfect for batched inference with fixed patterns

5. **Trade-off: Static graph structure (less flexible than dynamic)**
   - Cannot change kernel parameters or control flow after capture

## Use Cases in LLM Serving

- ✅ **Fixed/bucketed sequence lengths in inference**
  - Pre-capture graphs for common sequence lengths
  
- ✅ **Repetitive attention and FFN layer patterns**
  - Same operations repeated for each token
  
- ✅ **Reducing per-token latency in autoregressive generation**
  - Lower overhead = faster token generation
  
- ✅ **Maximizing throughput for batch processing**
  - Process more requests per second

## Conclusions

CUDA graphs provide **significant performance benefits (14.5% speedup) for small, repetitive GPU workloads** where CPU launch overhead is a bottleneck. The benefit is most pronounced when:

- Kernels are small and execute quickly
- Many kernel launches occur in sequence
- The same pattern repeats many times

For LLM serving, this translates to **faster per-token generation** and **higher throughput** when using fixed or bucketed sequence lengths.

---

*Generated from cuda_graph_demo output on November 20, 2025*
