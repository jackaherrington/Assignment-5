#!/usr/bin/env python3
import os
import subprocess
import shlex
import time
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
EXEC = str(Path(__file__).parent / 'monte_carlo_omp')
FUNCTIONS = ['x', 'x3', 'cos100x', 'inv_sqrt']
N_POINTS = 10_000_000  # adjust if needed
SEED = 42
THREAD_COUNTS = [1, 2, 4, 8, 12, 16, 20, 24]
SCHEDULE_KINDS = ['dynamic', 'static']
CHUNK_SIZES = [256, 1024, 4096]
REPEAT = 3  # run each config multiple times to average


def run_once(threads: int, schedule: str, func: str):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    env['OMP_SCHEDULE'] = schedule
    cmd = f"{shlex.quote(EXEC)} {func} {N_POINTS} {SEED}"
    start = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    end = time.time()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stderr}")
    # Parse fields from program output
    parsed = {
        'Function': None,
        'Threads': threads,
        'OMP_SCHEDULE': schedule,
        'Points N': N_POINTS,
        'Result': None,
        'Exact': None,
        'Error': None,
        'Time (s)': None,
    }
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('Function:'):
            parsed['Function'] = line.split(':', 1)[1].strip()
        elif line.startswith('Threads:'):
            try:
                parsed['Threads'] = int(line.split(':', 1)[1].strip())
            except Exception:
                pass
        elif line.startswith('OMP_SCHEDULE:'):
            parsed['OMP_SCHEDULE'] = line.split(':', 1)[1].strip()
        elif line.startswith('Points N:'):
            try:
                parsed['Points N'] = int(line.split(':', 1)[1].strip())
            except Exception:
                pass
        elif line.startswith('Result:'):
            parsed['Result'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('Exact:') and 'unknown' not in line:
            parsed['Exact'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('Error:') and 'n/a' not in line:
            parsed['Error'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('Time (s):'):
            parsed['Time (s)'] = float(line.split(':', 1)[1].strip())
    if parsed['Time (s)'] is None:
        parsed['Time (s)'] = end - start
    return parsed


def benchmark():
    results = {}
    for kind in SCHEDULE_KINDS:
        for chunk in CHUNK_SIZES:
            schedule = f"{kind},{chunk}"
            results[schedule] = {}
            for func in FUNCTIONS:
                results[schedule][func] = {}
                # Baseline T1 for this function
                t1_runs = [run_once(1, schedule, func) for _ in range(REPEAT)]
                T1_list = [r['Time (s)'] for r in t1_runs]
                T1 = statistics.mean(T1_list)
                T1_std = statistics.pstdev(T1_list) if len(T1_list) > 1 else 0.0
                results[schedule][func]['baseline'] = {
                    'threads': 1,
                    'T1_mean': T1,
                    'T1_std': T1_std,
                    'function': func,
                    'points': N_POINTS,
                    'result_mean': statistics.mean([r['Result'] for r in t1_runs if r['Result'] is not None]),
                    'exact': t1_runs[-1]['Exact'],
                    'error_mean': statistics.mean([r['Error'] for r in t1_runs if r['Error'] is not None]),
                }
                # Thread sweep for this function
                for p in THREAD_COUNTS:
                    p_runs = [run_once(p, schedule, func) for _ in range(REPEAT)]
                    Tp_list = [r['Time (s)'] for r in p_runs]
                    Tp = statistics.mean(Tp_list)
                    Tp_std = statistics.pstdev(Tp_list) if len(Tp_list) > 1 else 0.0
                    eff = T1 / (p * Tp)
                    results[schedule][func][p] = {
                        'threads': p,
                        'Tp_mean': Tp,
                        'Tp_std': Tp_std,
                        'efficiency': eff,
                        'function': func,
                        'points': N_POINTS,
                        'result_mean': statistics.mean([r['Result'] for r in p_runs if r['Result'] is not None]),
                        'exact': p_runs[-1]['Exact'],
                        'error_mean': statistics.mean([r['Error'] for r in p_runs if r['Error'] is not None]),
                    }
                    print(f"Schedule={schedule} func={func} p={p} T1={T1:.6f} Tp={Tp:.6f} Eff={eff:.3f} Time_std={Tp_std:.6f} Error_mean={results[schedule][func][p]['error_mean']:.3e}")
    return results


def save_results(results, path: Path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")


def plot_efficiency(results, out_path: Path):
    # Keep plot focused on function 'x' for comparability
    plt.figure(figsize=(10, 6))
    for schedule, data in results.items():
        if 'x' not in data:
            continue
        ps = sorted([k for k in data['x'].keys() if isinstance(k, int)])
        effs = [data['x'][p]['efficiency'] for p in ps]
        plt.plot(ps, effs, marker='o', label=schedule)
    plt.xlabel('Threads (p)')
    plt.ylabel('Parallel Efficiency (T1 / (p * Tp))')
    plt.title('Parallel Efficiency vs Threads (function x)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='OMP_SCHEDULE')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def plot_efficiency_per_function(results, out_dir: Path):
    for func in FUNCTIONS:
        plt.figure(figsize=(10, 6))
        for schedule, data in results.items():
            if func not in data:
                continue
            ps = sorted([k for k in data[func].keys() if isinstance(k, int)])
            effs = [data[func][p]['efficiency'] for p in ps]
            plt.plot(ps, effs, marker='o', label=schedule)
        plt.xlabel('Threads (p)')
        plt.ylabel('Parallel Efficiency (T1 / (p * Tp))')
        plt.title(f'Parallel Efficiency vs Threads (function {func})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='OMP_SCHEDULE')
        plt.tight_layout()
        out_path = out_dir / f'efficiency_plot_{func}.png'
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    if not Path(EXEC).exists():
        raise SystemExit(f"Executable not found: {EXEC}. Build it first with 'make'.")
    results = benchmark()
    out_json = Path(__file__).parent / 'efficiency_results.json'
    out_png = Path(__file__).parent / 'efficiency_plot.png'
    save_results(results, out_json)
    plot_efficiency(results, out_png)
    # Also create per-function plots
    plot_efficiency_per_function(results, Path(__file__).parent)
