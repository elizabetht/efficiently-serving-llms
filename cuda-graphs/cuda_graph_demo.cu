/**
 * CUDA Graph Performance Demonstration
 * 
 * This program demonstrates the performance difference between:
 * 1. Traditional CUDA kernel launches (high CPU overhead)
 * 2. CUDA Graph-based execution (low CPU overhead)
 * 
 * Compile: nvcc -o cuda_graph_demo cuda_graph_demo.cu -O3
 * Run: ./cuda_graph_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel 1: Vector addition
__global__ void vectorAdd(float *x, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + value;
    }
}

// Kernel 2: Vector multiplication
__global__ void vectorMul(float *x, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] * value;
    }
}

// Kernel 3: ReLU activation
__global__ void vectorReLU(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

/**
 * Traditional approach: Launch kernels one by one
 * Each kernel launch incurs CPU overhead
 */
double traditionalApproach(float *d_data, int n, int iterations, 
                          dim3 blocks, dim3 threads) {
    // Warmup: Execute kernels a few times before timing
    for (int i = 0; i < 3; i++) {
        vectorAdd<<<blocks, threads>>>(d_data, 0.1f, n);
        vectorMul<<<blocks, threads>>>(d_data, 1.01f, n);
        vectorReLU<<<blocks, threads>>>(d_data, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Execute kernel sequence multiple times
    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocks, threads>>>(d_data, 0.1f, n);
        vectorMul<<<blocks, threads>>>(d_data, 1.01f, n);
        vectorReLU<<<blocks, threads>>>(d_data, n);
    }
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

/**
 * CUDA Graph approach: Capture kernel sequence once, replay many times
 * Dramatically reduces CPU overhead
 */
double cudaGraphApproach(float *d_data, int n, int iterations,
                        dim3 blocks, dim3 threads) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Warmup: Execute kernels a few times before capturing
    for (int i = 0; i < 3; i++) {
        vectorAdd<<<blocks, threads, 0, stream>>>(d_data, 0.1f, n);
        vectorMul<<<blocks, threads, 0, stream>>>(d_data, 1.01f, n);
        vectorReLU<<<blocks, threads, 0, stream>>>(d_data, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Begin graph capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Capture the kernel sequence
    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocks, threads, 0, stream>>>(d_data, 0.1f, n);
        vectorMul<<<blocks, threads, 0, stream>>>(d_data, 1.01f, n);
        vectorReLU<<<blocks, threads, 0, stream>>>(d_data, n);
    }
    
    // End capture
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    
    // Instantiate the graph
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    
    // Now measure the replay performance
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    
    // Launch the entire graph with a single call
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return milliseconds;
}

/**
 * Initialize array with random values
 */
void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random values in [-1, 1]
    }
}

/**
 * Verify that two arrays are approximately equal
 */
bool verifyResults(float *a, float *b, int n, float tolerance = 1e-4) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

/**
 * Run benchmark for a specific configuration
 */
void runBenchmark(int n, int iterations, int num_runs) {
    printf("\n");
    printf("======================================================================\n");
    printf("Benchmarking: Array size = %d elements, Iterations = %d\n", n, iterations);
    printf("======================================================================\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_data = (float*)malloc(bytes);
    float *h_result_trad = (float*)malloc(bytes);
    float *h_result_graph = (float*)malloc(bytes);
    
    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 threads(threadsPerBlock);
    dim3 blocks(blocksPerGrid);
    
    double total_trad_time = 0.0;
    double total_graph_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Initialize data
        initializeArray(h_data, n);
        
        // Allocate device memory for traditional approach
        float *d_data_trad;
        CUDA_CHECK(cudaMalloc(&d_data_trad, bytes));
        CUDA_CHECK(cudaMemcpy(d_data_trad, h_data, bytes, cudaMemcpyHostToDevice));
        
        // Run traditional approach
        double time_trad = traditionalApproach(d_data_trad, n, iterations, blocks, threads);
        total_trad_time += time_trad;
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_result_trad, d_data_trad, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data_trad));
        
        // Allocate device memory for graph approach
        float *d_data_graph;
        CUDA_CHECK(cudaMalloc(&d_data_graph, bytes));
        CUDA_CHECK(cudaMemcpy(d_data_graph, h_data, bytes, cudaMemcpyHostToDevice));
        
        // Run CUDA graph approach
        double time_graph = cudaGraphApproach(d_data_graph, n, iterations, blocks, threads);
        total_graph_time += time_graph;
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_result_graph, d_data_graph, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data_graph));
        
        // Verify results match
        bool match = verifyResults(h_result_trad, h_result_graph, n);
        
        double speedup = time_trad / time_graph;
        printf("Run %2d: Traditional=%7.3fms, CUDA Graph=%7.3fms, Speedup=%.2fx %s\n",
               run + 1, time_trad, time_graph, speedup, 
               match ? "✓" : "✗ MISMATCH");
    }
    
    // Calculate statistics
    double avg_trad = total_trad_time / num_runs;
    double avg_graph = total_graph_time / num_runs;
    double avg_speedup = avg_trad / avg_graph;
    
    printf("----------------------------------------------------------------------\n");
    printf("Average Traditional:  %7.3f ms\n", avg_trad);
    printf("Average CUDA Graph:   %7.3f ms\n", avg_graph);
    printf("Average Speedup:      %.2fx\n", avg_speedup);
    printf("Time Saved:           %7.3f ms (%.1f%% reduction)\n", 
           avg_trad - avg_graph, 
           (avg_trad - avg_graph) / avg_trad * 100);
    printf("======================================================================\n");
    
    // Cleanup
    free(h_data);
    free(h_result_trad);
    free(h_result_graph);
}

int main() {
    printf("CUDA Graph Performance Demonstration\n");
    printf("======================================================================\n");
    
    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("CUDA Device:          %s\n", prop.name);
    printf("Compute Capability:   %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory:  %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Multiprocessors:      %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    
    // Check if CUDA graphs are supported (requires compute capability >= 7.0)
    if (prop.major < 7) {
        printf("\nWARNING: CUDA Graphs require compute capability 7.0 or higher.\n");
        printf("Your device has compute capability %d.%d\n", prop.major, prop.minor);
        printf("The program will run but may not show expected performance benefits.\n");
    }
    
    srand(time(NULL));
    
    // Run benchmarks with different configurations
    printf("\n");
    printf("Starting benchmarks...\n");
    
    // Small array, many iterations
    runBenchmark(1024 * 1024, 100, 10);
    
    // Medium array, moderate iterations
    runBenchmark(4 * 1024 * 1024, 50, 10);
    
    // Large array, fewer iterations
    runBenchmark(16 * 1024 * 1024, 20, 10);
    
    // Summary
    printf("\n");
    printf("======================================================================\n");
    printf("KEY TAKEAWAYS:\n");
    printf("======================================================================\n");
    printf("1. CUDA graphs eliminate per-kernel CPU launch overhead\n");
    printf("2. Benefits increase with more kernel launches in the sequence\n");
    printf("3. Graph capture happens once; replay is extremely fast\n");
    printf("4. Ideal for repetitive, fixed-structure GPU workloads\n");
    printf("5. Trade-off: Static graph structure (less flexible than dynamic)\n");
    printf("\n");
    printf("USE CASES IN LLM SERVING:\n");
    printf("• Fixed/bucketed sequence lengths in inference\n");
    printf("• Repetitive attention and FFN layer patterns\n");
    printf("• Reducing per-token latency in autoregressive generation\n");
    printf("• Maximizing throughput for batch processing\n");
    printf("======================================================================\n");
    
    return 0;
}
