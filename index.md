<!-- Inline CSS for styling the title and navigation -->
<style>
.header-title {
  padding: 40px 20px;
  text-align: center;
  color: #333;
}
nav {
  text-align: center;
  margin: 20px 0;
}
nav a {
  margin: 0 15px;
  text-decoration: none;
  font-weight: bold;
  color: #333;
}
nav a:hover {
  color: #007acc;
}
</style>

<!-- Title Section -->
<div class="header-title">
  <h1>15-618 Project Proposal</h1>
  <p><strong>Team:</strong> Xiangyu Bao (xiangyub), Xuye He (xuyeh)</p>
</div>

<!-- Navigation Links -->
<nav>
  <a href="/">Proposal</a> |
  <a href="/milestone">Milestone Report</a> |
  <a href="/page2">Final Report</a>
</nav>

<!-- Main Content -->
## TITLE

**Parallel Seam Carving Algorithm**

## URL

<https://github.com/ArtemisBow98/15618Project/>

## SUMMARY

We're going to parallelize the seam carving algorithm using OpenMP for multi-core CPUs, and compare their performance to a serialized version. Based on our performance measurements and observation, we plan to further optimize the parallel implementation for improved efficiency and scalability.

## BACKGROUND

Seam carving is an image resizing algorithm introduced by Avidan and Shamir [Seam Carving Paper](https://dl.acm.org/doi/pdf/10.1145/1275808.1276390). In its original, serial form, the algorithm resizes images in a content-aware way by finding and removing "seams"—paths of connected pixels with low energy. This method is widely used to adapt images for different screen sizes or aspect ratios while keeping important content intact.

The serial algorithm consists of several key steps:
1. **Energy Calculation:** Compute an energy value for every pixel to measure its importance.
2. **Seam Identification:** Use dynamic programming to find the optimal seam (i.e., the connected path with the lowest total energy) from one edge of the image to the opposite edge.
3. **Seam Removal:** Remove the identified seam by shifting the remaining pixels.
4. **Iteration:** Repeat the process until the image reaches the desired dimensions.

An important aspect of this algorithm is that many of its tasks can benefit from parallelism. For example, the energy calculation for each pixel is an independent operation, and the pixel shifting during seam removal can be done concurrently. Additionally, parts of the dynamic programming used in seam identification can be restructured to run in parallel. Speeding up these operations both within each iteration and having parallel between iterations can greatly reduce overall processing time [Trobec et al., 2018](https://doi.org/10.1007/978-3-319-98833-7).

Here is a pseudocode outline for one iteration:

~~~plaintext
for each seam removal iteration:
    in parallel for each pixel:
        calculate energy(pixel)
    in parallel, compute cumulative energy map (using dynamic programming)
    backtrack to find the best seam (this step can be partly parallelized)
    in parallel for each row:
        remove the pixel on the seam and shift remaining pixels
~~~

The energy calculation and pixel shifting are originally embarrassingly parallel tasks that can run simultaneously on different cores. Our project, however, will also focus on parallelizing the dynamic programming part used in seam identification, which is more challenging due to its inherent dependencies. In addition, we aim to explore optimizations between iterations to achieve an overall more efficient seam carving process.

## THE CHALLENGE

- **Dynamic Programming Dependencies:**
  In the dynamic programming phase for seam identification, each pixel’s cumulative energy is computed based on the values of neighboring pixels from previous rows or columns. This creates a chain of dependencies that must be carefully managed to avoid race conditions and excessive synchronization overhead.

- **Inter-Iteration Dependency:**
  After each seam removal, the image is updated and the energy landscape changes. This means that every new iteration depends on the results of the previous one, limiting the overall concurrency that can be exploited in the algorithm.

- **Memory Access Characteristics:**
  While the energy calculation and pixel shifting phases benefit from good spatial locality, the dynamic programming phase may involve irregular memory access patterns. This can lead to unpredictable communication overhead and higher synchronization costs, which negatively affect load balance and overall performance.

- **System Constraints on Multicore CPUs:**
  Mapping the workload onto a multicore CPU system using OpenMP is challenging due to architectural constraints. With one level of shared cache and potentially multiple CPUs running a high number of threads, data must be organized in a cache-friendly manner to minimize the communication-to-computation ratio. Additionally, divergent execution paths when processing complex regions of the image can further reduce parallel efficiency and increase synchronization overhead.

## RESOURCES

1. **Code base**: We will make use of existing serialized solution on the problem. There are a few parallelized solutions to the problem, and we may do performance comparison and debugging on these algorithms if we have time.

2. **Computing**: We plan to do an OpenMP implementation, so we want to mostly rely on GHC & PSC clusters.

## GOALS AND DELIVERABLES

1. **PLAN TO ACHIEVE**:
   1. **Parallelization with OpenMP**: Our baseline expectation is an OpenMP solution of the existing serialized solution that can achieve 4x speedup on 8 threads in the non-embarassingly-parallel stages of the seam carving algorithm.
   2. **Optimization for Memory Access Pattern**: We will do extensive performance debugging to optimize the memory access pattern of our proposed solution, which we believe is commonly addressed but rarely solved by many of the parallel implementations we found online or in textbooks.
   3. **Profiling and Metrics Reporting**: For each iterations of optimization, we will clearly lay out reasonings and profiling results for making the choice, including metrics such as execution, sync & memory time, speedup, cache utilization.

2. **HOPE TO ACHIEVE**:

   1. **Comparison with Other Implementations**: Do comparison and performance debugging on other existing parallel solutions.
   2. **SIMD Implementation**: Explore SIMD parallelism in speeding up the implementation.

## PLATFORM CHOICE

We make OpenMP as our implementation choice as we want to closely investigate the locality problem in the algorithm. CUDA is also a choice when dealing with image processing, yet its per-block memory layout is different compared to the cache hierarchy in CPU memory, so we want to go with OpenMP.

## SCHEDULE

Here is our tentative schedule for the project:

- 3/27 - 3/30: Search of existing serial implementations & review of literature
- 3/31 - 4/2: Implement parallelzation of energy computation and seam removal stage.
- 4/3 - 4/9: Decide on strategy with parallelizing the DP problem in seam finding stage; implement at least one naive version of parallelized solution; implement correctness check on the parallelized solution.
- 4/10 - 4/13: Performance debugging on the solution, and investigate locality problem.
- 4/14 - 4/15: Work on milestone report.
- 4/16 - 4/22: Investigate parallelization across iterations, and push forward optimization on locality based on new parallelization schema.
- 4/23 - 4/25: If possible, move on scratch goals.
- 4/26 - 4/28: Work on final report and poster.
- 4/29: Poster session.
