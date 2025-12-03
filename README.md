# Parallel Monte Carlo Integration (OpenMP)
Compute ∫_0^1 f(x) via Monte Carlo with parallel OpenMP threads and runtime scheduling (static/dynamic). Includes C program, benchmarks, and efficiency plots.

## Build
make

## Run individual tests (examples)
# Choose function: x | x3 | cos100x | inv_sqrt
# Set threads and schedule (static or dynamic) with optional chunk size
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,1024 ./monte_carlo_omp x 10000000 42
OMP_NUM_THREADS=8 OMP_SCHEDULE=static,1024 ./monte_carlo_omp x3 10000000 42
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,1024 ./monte_carlo_omp cos100x 5000000 42
OMP_NUM_THREADS=8 OMP_SCHEDULE=static,1024 ./monte_carlo_omp inv_sqrt 10000000 42

## Run benchmark and generate plots
python3 benchmark.py
# Outputs: efficiency_results.json, efficiency_plot.png (x),
#          efficiency_plot_x.png, efficiency_plot_x3.png,
#          efficiency_plot_cos100x.png, efficiency_plot_inv_sqrt.png
