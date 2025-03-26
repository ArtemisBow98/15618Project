---
layout: page
title: "15-618 Final Project"
permalink: /
---

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
  <a href="/page1">Midterm Report</a> |
  <a href="/page2">Final Report</a>
</nav>

<!-- Main Content -->
## TITLE

**Parallel Seam Carving Algorithm**

## URL

<https://artemisbow98.github.io/15618Project/>

## SUMMARY

## BACKGROUND

## THE CHALLENGE

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
