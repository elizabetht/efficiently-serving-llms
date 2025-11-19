# CUDA Graphs Demo

Simple demonstration of CUDA graphs and their performance benefits for reducing CPU launch overhead.

## What is a CUDA Graph?

A CUDA graph captures a sequence of GPU operations once, then replays them with minimal CPU overhead.

**Traditional approach:** CPU launches each kernel separately (~10-20μs overhead each)  
**CUDA Graph approach:** CPU launches entire sequence at once (~1-2μs overhead total)

**Benefits:**
- Eliminates per-kernel CPU overhead
- Lower latency for repetitive patterns  
- Ideal for LLM serving (attention, FFN layers)

## Quick Start

```bash
# Compile and run
make run

# Or manually
nvcc -o cuda_graph_demo cuda_graph_demo.cu -O3
./cuda_graph_demo
```

## What the Demo Does

Compares two approaches for executing 3 simple kernels (add, multiply, ReLU):

1. **Traditional:** Each kernel launched separately from CPU
2. **CUDA Graph:** All kernels captured once and replayed

The demo runs multiple benchmark configurations and shows:
- Time for traditional approach
- Time for CUDA graph approach
- Speedup achieved
- Results verification (both produce identical output)

## Performance Results

Tested on **NVIDIA GB10** (Compute Capability 12.1, 128GB Memory):

### Small Arrays (1M elements, 100 iterations)
- **Traditional:** 2.821 ms
- **CUDA Graph:** 2.128 ms
- **Speedup:** 1.33x (24.5% faster)

### Medium Arrays (4M elements, 50 iterations)
- **Traditional:** 12.256 ms
- **CUDA Graph:** 11.969 ms
- **Speedup:** 1.02x (2.3% faster)

### Large Arrays (16M elements, 20 iterations)
- **Traditional:** 31.382 ms
- **CUDA Graph:** 30.850 ms
- **Speedup:** 1.02x (1.7% faster)

**Key Finding:** Biggest gains (~25%) occur with many small kernels where CPU launch overhead is significant relative to kernel execution time.

## Key Concepts

### Graph Lifecycle

1. **Warmup** - Initialize GPU, compile kernels
2. **Capture** - Record operations into graph using `cudaStreamBeginCapture()`
3. **Instantiate** - Optimize graph with `cudaGraphInstantiate()`
4. **Execute** - Replay with `cudaGraphLaunch()` (can repeat many times)
5. **Cleanup** - Free resources

### When to Use CUDA Graphs

✅ **Good for:**
- Repetitive workloads with fixed structure
- Many small kernel launches
- Latency-sensitive applications (e.g., LLM inference)

❌ **Not ideal for:**
- Dynamic workloads
- Conditional execution paths
- Single kernel execution

## Requirements

- CUDA-capable GPU (Compute Capability 7.0+ recommended)
- NVIDIA CUDA Toolkit
- GCC/G++ compiler

## Files

- `cuda_graph_demo.cu` - Main demo program (CUDA C++)
- `Makefile` - Build configuration
- `README.md` - This file

## Use Cases in LLM Serving

CUDA graphs reduce per-token latency in autoregressive generation:
- Fixed/bucketed sequence lengths
- Repetitive attention and FFN patterns  
- Batch processing optimization

## Further Reading

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [NVIDIA Blog on CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
