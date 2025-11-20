#!/usr/bin/env python3
"""
Benchmark Comparison Visualization Script

This script reads vLLM benchmark output files and generates comparative
visualizations for speculative decoding (SD) vs non-speculative decoding (noSD).
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BenchmarkResults:
    """Store parsed benchmark results"""
    name: str
    successful_requests: int
    failed_requests: int
    max_concurrency: int
    benchmark_duration: float
    total_input_tokens: int
    total_generated_tokens: int
    request_throughput: float
    output_token_throughput: float
    peak_output_token_throughput: float
    total_token_throughput: float
    
    # TTFT metrics
    mean_ttft: float
    median_ttft: float
    p99_ttft: float
    
    # TPOT metrics
    mean_tpot: float
    median_tpot: float
    p99_tpot: float
    
    # ITL metrics
    mean_itl: float
    median_itl: float
    p99_itl: float


def parse_benchmark_file(filepath: Path) -> BenchmarkResults:
    """Parse a benchmark output file and extract metrics"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Helper function to extract numeric values
    def extract_value(pattern: str) -> float:
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        return 0.0
    
    return BenchmarkResults(
        name=filepath.stem,
        successful_requests=int(extract_value(r'Successful requests:\s+(\d+)')),
        failed_requests=int(extract_value(r'Failed requests:\s+(\d+)')),
        max_concurrency=int(extract_value(r'Maximum request concurrency:\s+(\d+)')),
        benchmark_duration=extract_value(r'Benchmark duration \(s\):\s+([\d.]+)'),
        total_input_tokens=int(extract_value(r'Total input tokens:\s+(\d+)')),
        total_generated_tokens=int(extract_value(r'Total generated tokens:\s+(\d+)')),
        request_throughput=extract_value(r'Request throughput \(req/s\):\s+([\d.]+)'),
        output_token_throughput=extract_value(r'Output token throughput \(tok/s\):\s+([\d.]+)'),
        peak_output_token_throughput=extract_value(r'Peak output token throughput \(tok/s\):\s+([\d.]+)'),
        total_token_throughput=extract_value(r'Total Token throughput \(tok/s\):\s+([\d.]+)'),
        mean_ttft=extract_value(r'Mean TTFT \(ms\):\s+([\d.]+)'),
        median_ttft=extract_value(r'Median TTFT \(ms\):\s+([\d.]+)'),
        p99_ttft=extract_value(r'P99 TTFT \(ms\):\s+([\d.]+)'),
        mean_tpot=extract_value(r'Mean TPOT \(ms\):\s+([\d.]+)'),
        median_tpot=extract_value(r'Median TPOT \(ms\):\s+([\d.]+)'),
        p99_tpot=extract_value(r'P99 TPOT \(ms\):\s+([\d.]+)'),
        mean_itl=extract_value(r'Mean ITL \(ms\):\s+([\d.]+)'),
        median_itl=extract_value(r'Median ITL \(ms\):\s+([\d.]+)'),
        p99_itl=extract_value(r'P99 ITL \(ms\):\s+([\d.]+)'),
    )


def create_comparison_plots(nosd_results: BenchmarkResults, sd_results: BenchmarkResults, 
                           concurrency: str, output_dir: Path):
    """Create comprehensive comparison plots"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'nosd': '#e74c3c', 'sd': '#3498db'}
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Throughput Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    throughput_metrics = ['Request\nThroughput\n(req/s)', 'Output Token\nThroughput\n(tok/s)', 
                          'Total Token\nThroughput\n(tok/s)']
    nosd_throughput = [nosd_results.request_throughput, nosd_results.output_token_throughput,
                       nosd_results.total_token_throughput]
    sd_throughput = [sd_results.request_throughput, sd_results.output_token_throughput,
                     sd_results.total_token_throughput]
    
    x = np.arange(len(throughput_metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, nosd_throughput, width, label='Without SD', color=colors['nosd'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, sd_throughput, width, label='With SD', color=colors['sd'], alpha=0.8)
    
    ax1.set_ylabel('Throughput', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(throughput_metrics, fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. TTFT Comparison (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    ttft_metrics = ['Mean', 'Median', 'P99']
    nosd_ttft = [nosd_results.mean_ttft, nosd_results.median_ttft, nosd_results.p99_ttft]
    sd_ttft = [sd_results.mean_ttft, sd_results.median_ttft, sd_results.p99_ttft]
    
    x = np.arange(len(ttft_metrics))
    bars1 = ax2.bar(x - width/2, nosd_ttft, width, label='Without SD', color=colors['nosd'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, sd_ttft, width, label='With SD', color=colors['sd'], alpha=0.8)
    
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Time to First Token (TTFT)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ttft_metrics)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 3. TPOT Comparison (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    tpot_metrics = ['Mean', 'Median', 'P99']
    nosd_tpot = [nosd_results.mean_tpot, nosd_results.median_tpot, nosd_results.p99_tpot]
    sd_tpot = [sd_results.mean_tpot, sd_results.median_tpot, sd_results.p99_tpot]
    
    x = np.arange(len(tpot_metrics))
    bars1 = ax3.bar(x - width/2, nosd_tpot, width, label='Without SD', color=colors['nosd'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, sd_tpot, width, label='With SD', color=colors['sd'], alpha=0.8)
    
    ax3.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Time Per Output Token (TPOT)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tpot_metrics)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 4. ITL Comparison (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    itl_metrics = ['Mean', 'Median', 'P99']
    nosd_itl = [nosd_results.mean_itl, nosd_results.median_itl, nosd_results.p99_itl]
    sd_itl = [sd_results.mean_itl, sd_results.median_itl, sd_results.p99_itl]
    
    x = np.arange(len(itl_metrics))
    bars1 = ax4.bar(x - width/2, nosd_itl, width, label='Without SD', color=colors['nosd'], alpha=0.8)
    bars2 = ax4.bar(x + width/2, sd_itl, width, label='With SD', color=colors['sd'], alpha=0.8)
    
    ax4.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Inter-Token Latency (ITL)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(itl_metrics)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 5. Speedup/Slowdown Analysis (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Duration', 'TPOT\n(Mean)', 'TTFT\n(Mean)', 'ITL\n(Mean)']
    speedup = [
        nosd_results.benchmark_duration / sd_results.benchmark_duration,
        nosd_results.mean_tpot / sd_results.mean_tpot,
        nosd_results.mean_ttft / sd_results.mean_ttft,
        nosd_results.mean_itl / sd_results.mean_itl,
    ]
    
    x = np.arange(len(metrics))
    colors_speedup = ['#27ae60' if s > 1 else '#e74c3c' for s in speedup]
    bars = ax5.bar(x, speedup, color=colors_speedup, alpha=0.8)
    
    ax5.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Baseline (1x)')
    ax5.set_ylabel('Speedup Ratio (×)', fontsize=12, fontweight='bold')
    ax5.set_title('Performance Ratios (noSD / SD)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{height:.2f}×'
        if height > 1:
            label += f'\n({(height-1)*100:.0f}% faster)'
        else:
            label += f'\n({(1-height)*100:.0f}% slower)'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom' if height > 1 else 'top', fontsize=8, fontweight='bold')
    
    # 6. Overall Summary (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    BENCHMARK SUMMARY (Concurrency: {concurrency})
    
    ═══════════════════════════════════════════
    
    Total Duration:
      • Without SD: {nosd_results.benchmark_duration:.2f}s
      • With SD: {sd_results.benchmark_duration:.2f}s
      • Improvement: {(1 - sd_results.benchmark_duration/nosd_results.benchmark_duration)*100:.1f}%
    
    Output Token Throughput:
      • Without SD: {nosd_results.output_token_throughput:.2f} tok/s
      • With SD: {sd_results.output_token_throughput:.2f} tok/s
      • Improvement: {(sd_results.output_token_throughput/nosd_results.output_token_throughput - 1)*100:.1f}%
    
    Mean TPOT (Time per Token):
      • Without SD: {nosd_results.mean_tpot:.2f} ms
      • With SD: {sd_results.mean_tpot:.2f} ms
      • Improvement: {(1 - sd_results.mean_tpot/nosd_results.mean_tpot)*100:.1f}%
    
    Mean ITL (Inter-Token Latency):
      • Without SD: {nosd_results.mean_itl:.2f} ms
      • With SD: {sd_results.mean_itl:.2f} ms
      • Change: {(sd_results.mean_itl/nosd_results.mean_itl - 1)*100:.1f}%
    
    ═══════════════════════════════════════════
    
    KEY INSIGHT:
    SD provides {(sd_results.output_token_throughput/nosd_results.output_token_throughput - 1)*100:.0f}% throughput gain
    but with {(sd_results.mean_itl/nosd_results.mean_itl - 1)*100:.0f}% higher ITL (bursty delivery)
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Qwen3-32B: Speculative Decoding Performance Analysis (Concurrency={concurrency}, T=0.0)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f'benchmark_comparison_c{concurrency}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()


def create_latency_timeline_visualization(nosd_results: BenchmarkResults, 
                                         sd_results: BenchmarkResults,
                                         concurrency: str, output_dir: Path):
    """Create a visualization showing the token arrival pattern over time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Simulate token arrival pattern for a single request generating 256 tokens
    num_tokens = 256
    
    # Without SD: regular intervals
    nosd_times = [nosd_results.mean_ttft + i * nosd_results.mean_tpot for i in range(num_tokens)]
    nosd_tokens = list(range(1, num_tokens + 1))
    
    # With SD: bursty pattern (assume ~2.7 tokens per burst based on TPOT improvement)
    tokens_per_burst = nosd_results.mean_tpot / sd_results.mean_tpot
    burst_interval = sd_results.mean_itl
    
    sd_times = [sd_results.mean_ttft]
    sd_tokens = [1]
    current_time = sd_results.mean_ttft
    current_token = 1
    
    while current_token < num_tokens:
        # Tokens in this burst arrive nearly simultaneously
        burst_size = min(int(tokens_per_burst), num_tokens - current_token)
        for i in range(burst_size):
            sd_times.append(current_time + i * 5)  # 5ms spacing within burst
            sd_tokens.append(current_token + i + 1)
        
        current_token += burst_size
        current_time += burst_interval
    
    # Plot 1: Token arrival over time (first 50 tokens)
    plot_limit = min(50, num_tokens)
    ax1.scatter(nosd_times[:plot_limit], nosd_tokens[:plot_limit], 
               color='#e74c3c', alpha=0.6, s=50, label='Without SD', marker='o')
    ax1.scatter(sd_times[:plot_limit], sd_tokens[:plot_limit], 
               color='#3498db', alpha=0.6, s=50, label='With SD', marker='s')
    
    ax1.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Token Number', fontsize=12, fontweight='bold')
    ax1.set_title('Token Arrival Pattern (First 50 Tokens)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Highlight the bursty nature with vertical lines for SD
    for i in range(0, plot_limit, int(tokens_per_burst)):
        if i < len(sd_times):
            ax1.axvline(x=sd_times[i], color='#3498db', alpha=0.2, linestyle='--', linewidth=1)
    
    # Plot 2: Cumulative tokens over time
    ax2.plot(nosd_times, nosd_tokens, color='#e74c3c', linewidth=2.5, 
            label=f'Without SD ({nosd_results.mean_tpot:.0f}ms/token avg)', marker='o', 
            markersize=3, markevery=20)
    ax2.plot(sd_times[:len(sd_tokens)], sd_tokens, color='#3498db', linewidth=2.5,
            label=f'With SD ({sd_results.mean_tpot:.0f}ms/token avg, bursty)', marker='s',
            markersize=3, markevery=20)
    
    ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Tokens Generated', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Token Generation Over Time (All 256 Tokens)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add completion time annotations
    nosd_completion = nosd_times[-1]
    sd_completion = sd_times[min(len(sd_times)-1, num_tokens-1)]
    
    ax2.axvline(x=nosd_completion, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.5)
    ax2.axvline(x=sd_completion, color='#3498db', linestyle=':', linewidth=2, alpha=0.5)
    
    ax2.text(nosd_completion, num_tokens * 0.5, f'  NoSD done\n  {nosd_completion:.0f}ms',
            color='#e74c3c', fontweight='bold', fontsize=10)
    ax2.text(sd_completion, num_tokens * 0.7, f'  SD done\n  {sd_completion:.0f}ms',
            color='#3498db', fontweight='bold', fontsize=10)
    
    plt.suptitle(f'Token Generation Timeline Visualization (Concurrency={concurrency})',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f'latency_timeline_c{concurrency}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()


def main():
    """Main function to process all benchmark files and generate plots"""
    
    results_dir = Path(__file__).parent
    
    # Find all pairs of benchmark files
    benchmark_pairs = [
        ('c1', 'qwen3-32b-nosd-c1-t0.0.out', 'qwen3-32b-sd-c1-t0.0.out'),
        ('c100', 'qwen3-32b-nosd-c100-t0.0.out', 'qwen3-32b-sd-c100-t0.0.out'),
    ]
    
    for concurrency, nosd_file, sd_file in benchmark_pairs:
        nosd_path = results_dir / nosd_file
        sd_path = results_dir / sd_file
        
        if not nosd_path.exists():
            print(f"Warning: {nosd_path} not found, skipping...")
            continue
        if not sd_path.exists():
            print(f"Warning: {sd_path} not found, skipping...")
            continue
        
        print(f"\nProcessing concurrency={concurrency}...")
        
        # Parse the benchmark files
        nosd_results = parse_benchmark_file(nosd_path)
        sd_results = parse_benchmark_file(sd_path)
        
        # Generate comparison plots
        create_comparison_plots(nosd_results, sd_results, concurrency, results_dir)
        
        # Generate timeline visualization
        create_latency_timeline_visualization(nosd_results, sd_results, concurrency, results_dir)
    
    print("\n✓ All plots generated successfully!")
    print(f"Output directory: {results_dir}")


if __name__ == '__main__':
    main()
