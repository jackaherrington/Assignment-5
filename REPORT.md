# Parallel Monte Carlo Integration Report

## Overview
This report documents the design and performance of a parallel Monte Carlo integration on the interval (0,1), implemented in C with OpenMP and benchmarked on macOS. We estimate integrals using the Monte Carlo average (1/N)∑ f(x_i), with x_i uniformly sampled in (0,1).

## Implementation Choices
- Language: C for performance and portability.
- Parallel model: OpenMP for shared-memory threading and easy runtime scheduling control.
- Device: CPU (macOS). CUDA/GPU was not selected to keep dependencies minimal and because the target platform provides many CPU cores.
- RNG: erand48 with per-thread 48-bit state (unsigned short xs[3]) for thread safety; xs[0], xs[1] constant, xs[2] derived from base seed XOR thread id.
- Runtime scheduling: schedule(runtime) to vary static/dynamic and chunk sizes via OMP_SCHEDULE without recompiling.

## Method and Monte Carlo Background
Monte Carlo integration approximates ∫_0^1 f(x) dx by drawing N random samples x_i ~ U(0,1) and computing the sample mean. The estimator is unbiased when x_i are i.i.d. uniform and its variance decreases as O(1/N). Parallelization distributes iterations across threads, each maintaining independent RNG state and accumulating a local sum, combined via reduction.

Functions integrated:
- f(x)=x (exact: 1/2)
- f(x)=x^3 (exact: 1/4)
- f(x)=cos(100x) (exact: sin(100)/100)
- f(x)=1/sqrt(x) (exact: 2)

Program outputs per run: result (estimate), exact value, error, number of points, execution time, and schedule.

## How It Was Done
- File `monte_carlo_omp.c` implements the estimator with:
  - Function selection by CLI argument (x, x3, cos100x, inv_sqrt).
  - OpenMP parallel region, per-thread `xs` state for `erand48`, and `#pragma omp for schedule(runtime)`.
  - Report prints required fields.
- Build via `Makefile` using Homebrew libomp paths.
- Benchmarks in `benchmark.py`:
  - Schedules: static and dynamic.
  - Chunk sizes: 256, 1024, 4096.
  - Threads p∈{1,2,4,8,12,16,20,24}.
  - N=10,000,000, function x, 3 repeats per point.
  - Records result, exact, error, N, time (mean/std), efficiency.
  - Saves results in `efficiency_results.json` and plot in `efficiency_plot.png`.

## Results
- Parallel efficiency curves per schedule and chunk size are shown in the figure below.
- Detailed numeric results are in `efficiency_results.json`.

![Parallel Efficiency Plot](efficiency_plot.png)

### Summary Tables (function x, chunk size = 1024)

Dynamic, chunk=1024:

| Threads p | Tp_mean (s) | Efficiency T1/(p*Tp) | Error_mean |
|-----------|-------------|-----------------------|------------|
| 1         | 0.038761    | 1.0263                | 1.118e-04  |
| 2         | 0.020234    | 0.9830                | 2.168e-05  |
| 4         | 0.009945    | 1.0000                | 1.344e-04  |
| 8         | 0.005528    | 0.8995                | 4.206e-05  |
| 12        | 0.004157    | 0.7975                | 4.256e-05  |
| 16        | 0.004320    | 0.5755                | 2.922e-05  |
| 20        | 0.004294    | 0.4632                | 2.166e-05  |
| 24        | 0.004310    | 0.3846                | 9.428e-05  |

Static, chunk=1024:

| Threads p | Tp_mean (s) | Efficiency T1/(p*Tp) | Error_mean |
|-----------|-------------|-----------------------|------------|
| 1         | 0.035617    | 1.0070                | 1.118e-04  |
| 2         | 0.018633    | 0.9624                | 2.703e-05  |
| 4         | 0.009653    | 0.9289                | 1.363e-04  |
| 8         | 0.005009    | 0.8951                | 5.272e-05  |
| 12        | 0.004932    | 0.6060                | 4.884e-05  |
| 16        | 0.005090    | 0.4404                | 2.114e-06  |
| 20        | 0.005525    | 0.3246                | 2.132e-05  |
| 24        | 0.005036    | 0.2967                | 4.511e-05  |

## Performance Discussion and Schedule Comparison
- Dynamic vs Static:
  - Dynamic scheduling generally improves load balance when function evaluation cost varies or RNG-induced variance yields uneven work per chunk. It may incur higher scheduling overhead.
  - Static scheduling has minimal overhead and can achieve higher efficiency when work is uniformly distributed and chunk size is well chosen.
  - Dynamic scheduling: Higher efficiency at mid-range threads (e.g., p=12: 0.7975 vs static 0.6060), indicating better load balance with these chunk sizes. Slightly lower at high p due to overhead (p=24: 0.3846 vs static 0.2967 but still higher than static here).
  - Static scheduling: Very competitive at low threads and maintains good efficiency up to p=8, then drops more sharply as p increases on this platform/config.
- Chunk size effects:
  - Small chunks (e.g., 256) favor dynamic load balancing at the cost of overhead; beneficial if variability exists.
  - Larger chunks (e.g., 4096) reduce overhead but may worsen balance under dynamic; static often benefits from moderate chunks (e.g., 1024) for good locality and reduced synchronization.
  - Chunk size effects (reported separately in JSON): 256 and 4096 show similar trends; 1024 provided a balanced trade-off.
- Observed behavior (typical):
  - At lower thread counts, both schedules approach linear speedup with high efficiency.
  - As p grows, dynamic may maintain better efficiency if imbalance appears, while static can edge ahead when the workload is homogeneous.
- Overall: Dynamic schedule produced equal or better efficiency across most thread counts in this experiment, with comparable accuracy (errors ~1e-5–1e-4). Static had slightly lower overhead at p=1 but lost efficiency as p increased.

## Reproducibility
- Build: `make`
- Run examples:
  - `OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,1024 ./monte_carlo_omp x 10000000 42`
  - `OMP_NUM_THREADS=8 OMP_SCHEDULE=static,1024 ./monte_carlo_omp x 10000000 42`
- Benchmark and plot: `python3 benchmark.py`

## References and Notes
- `man erand48` for RNG details; thread-safe when state is thread-local.
- Efficiency computed as T1/(p*Tp), from baseline p=1 timing.
- The Monte Carlo estimator variance decreases as 1/N; errors reported align with this rate.
