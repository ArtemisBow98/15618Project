---
layout: page
title: "Milestone"
permalink: /milestone/
---

# Milestone Report

## Revisited schedule
We are a bit late on running performance debugging on the naive solution, and we will make that the first priority after the milestone.

- 4/16 - 4/18
  - Xiangyu: performance debugging on naive implementation of dp
  - Xuye: implementation of triangle-block decomposition
- 4/19 - 4/21
  - Xiangyu: implementation of correctness checker; investigating cross-iteration parallelization
  - Xuye: performance debugging of triangle-block decomposition
- 4/22 - 4/25
  - Xiangyu: attempt on cross-iteration parallelization
  - Xuye: locality optimization on the new decomposition schemas
- 4/26 - 4/27
  - Xiangyu & Xuye: finalize report and poster content

## Work completed

1. We implemented a serialized version of seam carving algorithm. We use opencv to load/write images, and use pure C++ in manipulation of pixel BGRA values. The whole processing is divided to 4 stages: luminance, energy computation, dynamic programming for cumulative energy calculation, and minimal seam calculation.
2. We implemented naive parallelization of the 3rd step - cumulative energy calculation with row-wise pixel splitting.

## Goals of the project

Our previous goals of the project are:

1. Parallelization with OpenMP
2. Optimization for Memory Access Pattern
3. Profiling and Metrics Reporting

And we think we are able to hit all goals. Previously we think we are able to achieve cross-iteration parallelization of the problem, but now we will make it a stretch goal in place of the SIMD goal. So the new stretch goals are:

1. **Comparison with Other Implementations**: Do comparison and performance debugging on other existing parallel solutions.
2. **Cross-iteration parallelization**: Explore possible cross-iteration parallelization strategies.

## What to show at poster session

We will be showing graphs for the performance improvements, and pictures for carving output.

## Preliminary Results

### Overview and Work Completed

Thus far, the project has focused on developing and evaluating the naive parallel seam carving algorithm. Key image processing steps—luminance calculation, Sobel filtering, gradient-to-dynamic programming conversion (grad_to_dp), and seam computation—have been implemented using OpenMP for parallelization. Benchmarks have been conducted with 1, 2, 4, and 8 threads to assess processing time and scalability.

### Experimental Setup and Timing Results

The timing results (in milliseconds) from our experiments for the various stages of the algorithm are as follows:

1. **Single Thread (Baseline)**
   - **Luminance Calculation:** 5.77 ms
   - **Sobel Filter:** 59.30 ms
   - **grad_to_dp (960 iterations):** 11014.6 ms
   - **compute_seam (960 iterations):** 36.53 ms
   - **Total Processing Time:** 13251.9 ms

2. **2 Threads**
   - **Luminance Calculation:** 6.84 ms
   - **Sobel Filter:** 50.68 ms
   - **grad_to_dp (960 iterations):** 32779 ms
   - **compute_seam (960 iterations):** 57.43 ms
   - **Total Processing Time:** 35714.9 ms

3. **4 Threads**
   - **Luminance Calculation:** 6.43 ms
   - **Sobel Filter:** 50.12 ms
   - **grad_to_dp (960 iterations):** 46080 ms
   - **compute_seam (960 iterations):** 56.30 ms
   - **Total Processing Time:** 49072.6 ms

4. **8 Threads**
   - **Luminance Calculation:** 6.57 ms
   - **Sobel Filter:** 49.92 ms
   - **grad_to_dp (960 iterations):** 54973.8 ms
   - **compute_seam (960 iterations):** 58.25 ms
   - **Total Processing Time:** 57876.9 ms

These results suggest that the grad_to_dp phase experiences significant overhead in the parallelized environment. The increase in total processing time with additional threads indicates that synchronization and workload distribution inefficiencies might be limiting the benefits of our parallel approach.

## Next Steps

To extract more parallelism without reducing the grain size, we plan to adopt a different workload partitioning strategy inspired in part by ideas from [Shwestrick's work](https://shwestrick.github.io/2020/07/29/seam-carve.html). The strategy is to partition the work using triangular regions, which will help efficiently balance the workload while respecting inter-pixel dependencies.

### Workload Partitioning

- **Strip Grouping:**
  Group adjacent rows into strips. Within each strip, cover the upper portion with downward-pointing triangles that capture the dependencies required for later computations.

- **Filling Gaps:**
  The remaining area in each strip forms upward-pointing triangles whose dependencies have been met during the first pass.

- **Parallel Rounds:**
  Process each strip in two distinct parallel rounds—first, process the downward-pointing triangles in parallel, then handle the upward-pointing ones.

### Profiling Strategy

We will profile by dividing the image into multiple strips based on how many replicated values need to be computed. This approach will also consider the cache line size at the bottom to prevent false sharing. The goal is to determine the optimal granularity, likely aiming for triangles with base-widths between 64 and 90 pixels, which should yield better performance scaling compared to our current row-major strategy.

### Action Items

- **Profiling:**
  Conduct detailed profiling to identify the exact sources of overhead, particularly in the grad_to_dp phase, and to validate the effectiveness of the multiple strips approach.

- **Optimization:**
  Explore alternative approaches for workload distribution and synchronization to mitigate the observed performance degradation.

- **Refinement:**
  Adjust the experimental methodology to better isolate the effects of parallelization on each subroutine and fine-tune the partitioning strategy.

## Issues

- **Parallelization Overhead:** The observed increase in processing time with an increased number of threads suggests that synchronization and thread management overhead are offsetting the benefits of parallel execution.

- **Load Balancing:** It appears that the grad_to_dp routine might not be evenly distributing the workload among threads, resulting in bottlenecks. Enhanced profiling and an optimized distribution strategy are required.

- **Scalability Concerns:** While the grad_to_dp phase shows significant scalability issues that need addressing, it is worth noting that other subroutines, such as luminance calculation and Sobel filtering, have not yet been parallelized. These represent additional opportunities for future parallelization improvements.

- **Resource Management:** Potential issues related to memory and cache usage under parallel execution remain uncertain. Further investigation into resource allocation and concurrent memory access is needed.

- **Schedule Adjustments:** Due to the aforementioned issues, adjustments in the project schedule might be necessary to allocate additional time for debugging, profiling, and optimization before final deliverables are completed.

