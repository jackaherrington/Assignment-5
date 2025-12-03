#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Function selector via string (no hardcoding in the loop)
typedef double (*func_t)(double);

double f_linear(double x) { return x; }
double f_cubic(double x) { return x * x * x; }
double f_cos100(double x) { return cos(100.0 * x); }
double f_inv_sqrt(double x) { return 1.0 / sqrt(x); }

double exact_value(const char *fname) {
    if (strcmp(fname, "x") == 0) return 0.5;                 // ∫_0^1 x dx = 1/2
    if (strcmp(fname, "x3") == 0) return 0.25;                // ∫_0^1 x^3 dx = 1/4
    if (strcmp(fname, "cos100x") == 0) return sin(100.0) / 100.0; // ∫_0^1 cos(100x) dx = sin(100)/100
    if (strcmp(fname, "inv_sqrt") == 0) return 2.0;           // ∫_0^1 1/sqrt(x) dx = 2
    return NAN;
}

func_t pick_function(const char *fname) {
    if (strcmp(fname, "x") == 0) return f_linear;
    if (strcmp(fname, "x3") == 0) return f_cubic;
    if (strcmp(fname, "cos100x") == 0) return f_cos100;
    if (strcmp(fname, "inv_sqrt") == 0) return f_inv_sqrt;
    return NULL;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <function> <N> [seed]\n", prog);
    fprintf(stderr, "  <function>: x | x3 | cos100x | inv_sqrt\n");
    fprintf(stderr, "  <N>: number of random points (e.g., 10000000)\n");
    fprintf(stderr, "  [seed]: optional base seed for reproducibility\n");
    fprintf(stderr, "Environment: set OMP_NUM_THREADS and OMP_SCHEDULE (static[,chunk] or dynamic[,chunk])\n");
}

int main(int argc, char **argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }
    const char *fname = argv[1];
    long long N = atoll(argv[2]);
    unsigned short base_seed = (argc >= 4) ? (unsigned short)atoi(argv[3]) : (unsigned short)time(NULL);

    if (N <= 0) { fprintf(stderr, "Error: N must be > 0\n"); return 1; }

    func_t f = pick_function(fname);
    if (!f) {
        fprintf(stderr, "Error: unknown function '%s'\n", fname);
        print_usage(argv[0]);
        return 1;
    }

    double exact = exact_value(fname);

    // Capture schedule from environment for reporting
    const char *sched_env = getenv("OMP_SCHEDULE");
    if (!sched_env) sched_env = "(not set)";

    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif

    double start = omp_get_wtime();

    // Parallel Monte Carlo with per-thread seed for erand48
    double sum = 0.0;

    // Use reduction on sum; schedule(runtime) lets us change schedule without recompiling.
    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        // Each thread keeps its own 48-bit state, updated by erand48 in-place; this is thread-safe as long as state is thread-local.
        unsigned short xs[3];
        xs[0] = 123;       // constant per instructions
        xs[1] = 345;       // constant per instructions
        xs[2] = (unsigned short)(base_seed ^ (unsigned short)tid);

        double local_sum = 0.0;

        #pragma omp for schedule(runtime) reduction(+:sum)
        for (long long i = 0; i < N; ++i) {
            // Generate a random number in (0,1); erand48 returns double in [0.0, 1.0)
            double x = erand48(xs);
            local_sum += f(x);
            // Accumulate into reduction variable in chunks to reduce contention
            if ((i & 0x3FF) == 0x3FF) {
                sum += local_sum;
                local_sum = 0.0;
            }
        }
        // Flush remainder
        sum += local_sum;
    }

    double estimate = sum / (double)N;
    double end = omp_get_wtime();

    double error = (isnan(exact)) ? NAN : fabs(estimate - exact);

    printf("Function: %s\n", fname);
    printf("Threads: %d\n", threads);
    printf("OMP_SCHEDULE: %s\n", sched_env);
    printf("Points N: %lld\n", N);
    printf("Result: %.15f\n", estimate);
    if (!isnan(exact)) {
        printf("Exact:    %.15f\n", exact);
        printf("Error:    %.15e\n", error);
    } else {
        printf("Exact:    (unknown)\n");
        printf("Error:    (n/a)\n");
    }
    printf("Time (s): %.6f\n", end - start);

    return 0;
}
