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
  <a href="/page1">Midterm Report</a> |
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

Below is a block diagram illustrating one iteration of the seam carving process, represented using a Mermaid flowchart:

~~~mermaid
flowchart TD
    A["Input Image"] --> B["Compute Energy for Each Pixel"]
    B --> C["Identify Optimal Seam with Dynamic Programming"]
    C --> D["Remove Seam & Shift Pixels"]
    D --> E["Updated Image"]
    E --> F["Repeat Until Target Size Reached"]
~~~

And here is a pseudocode outline for one iteration:

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

The primary challenge in parallelizing the seam carving algorithm lies in its dynamic programming component and the serial dependency between iterations. While the energy calculation and pixel shifting are originally embarrassingly parallel tasks, the process of identifying the optimal seam using dynamic programming introduces significant dependencies. In this phase, each pixel’s cumulative energy is computed based on the values of neighboring pixels from previous rows or columns, creating a chain of dependencies that must be carefully managed to avoid race conditions and excessive synchronization overhead.

Furthermore, after each seam is removed, the image is updated and the energy landscape changes, so each iteration depends on the results of the previous one. This serial dependency between iterations restricts the amount of concurrency that can be exploited across the entire algorithm.

Memory access characteristics also play a crucial role. While the energy computation and pixel shifting phases exhibit good spatial locality—since neighboring pixels are processed together—the dynamic programming phase may have irregular memory access patterns. This irregularity can lead to unpredictable communication overhead between processing elements and potentially higher synchronization costs, thereby affecting load balance and overall performance.

Mapping this workload onto a multicore CPU system using OpenMP is challenging due to the system’s architecture and resource constraints. In our target system—with one level of shared cache and potentially multiple CPUs running a high number of threads—data must be organized in a cache-friendly manner to avoid a high communication-to-computation ratio during the dynamic programming phase. If memory accesses are not optimized for the shared cache, performance can suffer significantly. Additionally, when processing complex regions of the image, divergent execution paths may occur, which further reduces parallel efficiency and increases synchronization overhead.

By undertaking this project, we aim to explore methods for restructuring dynamic programming to reduce dependency overhead, improve memory access patterns, and balance workload across processing cores. Ultimately, we hope to gain deeper insights into the challenges of parallelizing algorithms with inherent serial dependencies and learn effective techniques to overcome these constraints.

## RESOURCES

## GOALS AND DELIVERABLES

## PLATFORM CHOICE

## SCHEDULE
