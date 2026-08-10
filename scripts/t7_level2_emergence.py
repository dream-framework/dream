#!/usr/bin/env python3
"""
T7 Level-2 Experiment: Does aggregation of local S2 processes
converge to another stretched exponential?

The user's proposal:
    local S2s → interference → ensemble averaging → emergent S2?

Test: Generate N local S2 processes with random (D_i, λ_q,i),
compute the aggregate R_N(λ) = (1/N) Σ R_i(λ), and check whether
R_N converges to a stretched exponential as N → ∞.

Key questions:
  1. Does the aggregate converge? (LLN says yes for fixed λ)
  2. Does variance decrease as ~1/√N?
  3. Does the aggregate become more S2-like?
  4. Does D_macro converge to a stable value?
  5. Does D_macro depend on the distribution of local (D, λ_q)?
  6. Does the same behavior occur in real registry data?

Mathematical context:
  E[exp(-X)] ≠ exp(-E[X]) in general.
  So an arbitrary mixture of stretched exponentials does NOT automatically
  remain a stretched exponential.

  We need to test whether, under specific ρ(D, λ_q), the ensemble
  average is approximately closed under S2.
"""
import os, sys, json
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from s2_model_compare import m_s2


def aggregate_s2(N, D_samples, lambda_q_samples, lambda_values):
    """Compute R_N(λ) = (1/N) Σ exp[-(λ/λ_q,i)^D_i] for i=1..N."""
    R = np.zeros_like(lambda_values, dtype=float)
    for i in range(N):
        R += np.exp(-np.power(lambda_values / max(lambda_q_samples[i], 1e-6), D_samples[i]))
    return R / N


def fit_s2_to_curve(t, R):
    """Fit S2 to (t, R) and return (D, lambda_q, R²)."""
    if R[0] > 0: R_norm = R / R[0]
    else: R_norm = R.copy()
    t_mid = float(t[len(t) // 2]) if len(t) else 1.0
    best = None
    for p0 in [[1.0, t_mid, 0.5], [1.0, t_mid * 0.5, 1.0], [1.0, t_mid * 2, 0.3]]:
        try:
            popt, _ = curve_fit(m_s2, t, R_norm, p0=p0,
                                bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]),
                                maxfev=20000)
            rss = float(np.sum((R_norm - m_s2(t, *popt)) ** 2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except: pass
    if best is None: return None, None, None
    popt, rss = best
    ss_tot = float(np.sum((R_norm - np.mean(R_norm)) ** 2))
    if ss_tot == 0: return None, None, None
    return float(popt[2]), float(popt[1]), 1 - rss / ss_tot


def run_experiment(N_values, D_distribution, lambda_q_distribution, n_trials=10, seed_base=42):
    """Run the aggregation experiment for each N in N_values."""
    lambda_values = np.linspace(0.01, 50, 200)
    results = []

    for N in N_values:
        print(f'  N = {N}...', end=' ')
        trial_results = []
        for trial in range(n_trials):
            rng = np.random.RandomState(seed_base + trial * 1000 + N)
            D_samples = D_distribution(rng, N)
            lambda_q_samples = lambda_q_distribution(rng, N)
            R = aggregate_s2(N, D_samples, lambda_q_samples, lambda_values)
            D_fit, lambda_q_fit, r2 = fit_s2_to_curve(lambda_values, R)
            if D_fit is not None:
                trial_results.append({
                    'D_macro': D_fit,
                    'lambda_q_macro': lambda_q_fit,
                    'r2': r2,
                    'R_curve': R,
                })
        if not trial_results:
            print('all fits failed')
            results.append({'N': N, 'n_success': 0})
            continue

        D_macros = [t['D_macro'] for t in trial_results]
        r2s = [t['r2'] for t in trial_results]
        # Variance across trials (decreases with N if LLN applies)
        D_var = float(np.var(D_macros, ddof=1)) if len(D_macros) > 1 else 0
        D_mean = float(np.mean(D_macros))
        r2_mean = float(np.mean(r2s))

        # Also check: does the aggregate look like a single S2?
        # Compare to a perfect S2 fit
        # Use the first trial's R curve for the "best S2 fit R²"
        results.append({
            'N': N,
            'n_success': len(trial_results),
            'D_macro_mean': D_mean,
            'D_macro_std': float(np.std(D_macros, ddof=1)) if len(D_macros) > 1 else 0,
            'D_macro_var': D_var,
            'r2_mean': r2_mean,
            'r2_min': float(np.min(r2s)),
            'r2_max': float(np.max(r2s)),
        })
        print(f'D_macro={D_mean:.3f}±{np.std(D_macros, ddof=1) if len(D_macros) > 1 else 0:.3f}, R²={r2_mean:.4f}')

    return results


def main():
    print('=' * 80)
    print('T7 LEVEL-2 EXPERIMENT: Does S2 survive aggregation?')
    print('=' * 80)
    print()
    print('Hypothesis: Local S2 → interference → ensemble averaging → emergent S2')
    print('Test: Generate N local S2 processes, fit S2 to the aggregate, check convergence.')
    print()
    print('Mathematical warning: E[exp(-X)] ≠ exp(-E[X]) in general.')
    print('An arbitrary mixture of S2s does NOT automatically remain S2.')
    print()

    N_values = [1, 5, 10, 50, 100, 500, 1000, 5000]
    n_trials = 10

    # ── Distribution 1: Narrow (D near 1, λ_q near 10) ──
    print('═' * 80)
    print('DISTRIBUTION 1: Narrow — D ~ N(1.0, 0.1), λ_q ~ N(10, 1)')
    print('═' * 80)
    D_dist_1 = lambda rng, n: np.clip(rng.normal(1.0, 0.1, n), 0.01, 10)
    lambda_dist_1 = lambda rng, n: np.clip(rng.normal(10, 1, n), 0.1, 1e6)
    results_1 = run_experiment(N_values, D_dist_1, lambda_dist_1, n_trials)

    # ── Distribution 2: Broad (D ~ Uniform(0.3, 3), λ_q ~ LogN(2, 1)) ──
    print()
    print('═' * 80)
    print('DISTRIBUTION 2: Broad — D ~ Uniform(0.3, 3.0), λ_q ~ LogNormal(2, 1)')
    print('═' * 80)
    D_dist_2 = lambda rng, n: rng.uniform(0.3, 3.0, n)
    lambda_dist_2 = lambda rng, n: np.exp(rng.normal(2, 1, n))  # LogNormal
    results_2 = run_experiment(N_values, D_dist_2, lambda_dist_2, n_trials)

    # ── Distribution 3: Weibull-distributed D (matching the registry) ──
    print()
    print('═' * 80)
    print('DISTRIBUTION 3: Registry-mimicking — D ~ Weibull(1.5, 1.2), λ_q ~ LogN(2.5, 0.8)')
    print('═' * 80)
    D_dist_3 = lambda rng, n: np.clip(stats.weibull_min.rvs(1.5, loc=0, scale=1.2, size=n, random_state=rng), 0.01, 10)
    lambda_dist_3 = lambda rng, n: np.exp(rng.normal(2.5, 0.8, n))
    results_3 = run_experiment(N_values, D_dist_3, lambda_dist_3, n_trials)

    # ── Analysis ──
    print()
    print('=' * 80)
    print('ANALYSIS')
    print('=' * 80)

    for name, results in [('Narrow', results_1), ('Broad', results_2), ('Registry-like', results_3)]:
        print(f'\n{name} distribution:')
        print(f'{"N":>6} {"D_macro":>10} {"±std":>8} {"R²":>8} {"var(N)/var(N/10)":>18}')
        prev_var = None
        for r in results:
            if 'D_macro_mean' not in r: continue
            var_ratio = ''
            if prev_var is not None and prev_var > 0 and r['D_macro_var'] > 0:
                var_ratio = f'{prev_var / r["D_macro_var"]:.2f}'
            print(f'{r["N"]:>6} {r["D_macro_mean"]:>10.3f} {r["D_macro_std"]:>8.3f} {r["r2_mean"]:>8.4f} {var_ratio:>18}')
            prev_var = r['D_macro_var']

    # ── Summary ──
    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)

    # Check: does R² improve with N? (aggregate becomes more S2-like)
    for name, results in [('Narrow', results_1), ('Broad', results_2), ('Registry-like', results_3)]:
        r2s = [r.get('r2_mean', 0) for r in results if 'r2_mean' in r]
        if len(r2s) >= 2:
            r2_improves = r2s[-1] > r2s[0]
            print(f'  {name}: R² {r2s[0]:.3f} → {r2s[-1]:.3f} ({"improves" if r2_improves else "does not improve"} with N)')

    # Check: does D_macro converge?
    print()
    for name, results in [('Narrow', results_1), ('Broad', results_2), ('Registry-like', results_3)]:
        Ds = [r.get('D_macro_mean', 0) for r in results if 'D_macro_mean' in r]
        Ns = [r['N'] for r in results if 'D_macro_mean' in r]
        if len(Ds) >= 3:
            D_final = Ds[-1]
            D_var_final = results[-1].get('D_macro_std', 0)
            D_var_initial = results[0].get('D_macro_std', 1) if 'D_macro_std' in results[0] else 1
            converges = D_var_final < 0.1 * max(D_var_initial, 0.01)
            print(f'  {name}: D_macro converges to {D_final:.3f} (std={D_var_final:.3f}) — {"YES" if converges else "NO"}')

    print()
    print('If R² stays high (>0.99) at large N AND D_macro variance decreases,')
    print('then S2 is an emergent fixed point of aggregation under that distribution.')
    print()
    print('If R² drops at large N (aggregate deviates from S2), then S2 is NOT')
    print('closed under aggregation — the macroscopic law would be something else.')

    # Save
    output = {
        'experiment': 'T7 Level-2: Emergent S2 under aggregation',
        'N_values': N_values,
        'n_trials_per_N': n_trials,
        'distributions': {
            'narrow': results_1,
            'broad': results_2,
            'registry_like': results_3,
        },
    }
    out_path = os.path.join(REPO, 't7_level2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\n✓ Saved: {out_path}')


if __name__ == '__main__':
    main()
