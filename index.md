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
  <h1>My Project Proposal</h1>
</div>

<!-- Navigation Links -->
<nav>
  <a href="/">Proposal</a> |
  <a href="/page1">Milestone Report</a> |
  <a href="/page2">Final Report</a>
</nav>

<!-- Main Content -->
## TITLE

## Project URL

<https://github.com/ArtemisBow98/15618Project/>

## SUMMARY

We're going to parallelize the seam carving algorithm using OpenMP for multi-core CPUs, and compare their performance to a serialized version. Based on our performance measurements and observation, we plan to further optimize the parallel implementation for improved efficiency and scalability.

## Background

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

## The Challenge

- **Dynamic Programming Dependencies:**  
  In the dynamic programming phase for seam identification, each pixel’s cumulative energy is computed based on the values of neighboring pixels from previous rows or columns. This creates a chain of dependencies that must be carefully managed to avoid race conditions and excessive synchronization overhead.

- **Inter-Iteration Dependency:**  
  After each seam removal, the image is updated and the energy landscape changes. This means that every new iteration depends on the results of the previous one, limiting the overall concurrency that can be exploited in the algorithm.

- **Memory Access Characteristics:**  
  While the energy calculation and pixel shifting phases benefit from good spatial locality, the dynamic programming phase may involve irregular memory access patterns. This can lead to unpredictable communication overhead and higher synchronization costs, which negatively affect load balance and overall performance.

- **System Constraints on Multicore CPUs:**  
  Mapping the workload onto a multicore CPU system using OpenMP is challenging due to architectural constraints. With one level of shared cache and potentially multiple CPUs running a high number of threads, data must be organized in a cache-friendly manner to minimize the communication-to-computation ratio. Additionally, divergent execution paths when processing complex regions of the image can further reduce parallel efficiency and increase synchronization overhead.

## RESOURCES

## GOALS AND DELIVERABLES

## PLATFORM CHOICE

## SCHEDULE
