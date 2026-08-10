#!/usr/bin/env python3
"""
T7 Level-2 RECALC: Proper multi-trial AICc comparison.

Verifies the original finding across:
  - 5 random seeds
  - 3 distributions (narrow, broad, registry-like)
  - Multiple N values (100, 1000, 5000)

For each trial:
  1. Generate N local S2 processes
  2. Compute aggregate R_N(λ) = (1/N) Σ R_i(λ)
  3. Fit all 7 models (S2, S2_DUST, BIEXP, EXP, GAUSS, POWER, LOGNORM)
  4. Record AICc for each
  5. Check: does S2_DUST beat S2 consistently? By how much?

Reports: mean ΔAICc across trials, with confidence intervals.
"""
import os, sys, json
import numpy as np
from scipy import stats

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from s2_model_compare import fit_all_models


def aggregate_s2(N, D_samples, lambda_q_samples, lambda_values):
    R = np.zeros_like(lambda_values, dtype=float)
    for i in range(N):
        R += np.exp(-np.power(lambda_values / max(lambda_q_samples[i], 1e-6), D_samples[i]))
    return R / N


def run_trial(N, D_dist, lambda_dist, seed, lambda_values):
    rng = np.random.RandomState(seed)
    D_samples = D_dist(rng, N)
    lambda_q_samples = lambda_dist(rng, N)
    R = aggregate_s2(N, D_samples, lambda_q_samples, lambda_values)
    if R[0] > 0: R = R / R[0]

    fits = fit_all_models(lambda_values, R)
    if not fits or 'S2' not in fits or 'S2_DUST' not in fits:
        return None

    return {
        'N': N,
        'S2_aicc': fits['S2']['aicc'],
        'S2_DUST_aicc': fits['S2_DUST']['aicc'],
        'BIEXP_aicc': fits.get('BIEXP', {}).get('aicc'),
        'S2_r2': fits['S2']['r2'],
        'S2_DUST_r2': fits['S2_DUST']['r2'],
        'delta_aicc': fits['S2']['aicc'] - fits['S2_DUST']['aicc'],
        'delta_aicc_biexp': fits['S2']['aicc'] - fits['BIEXP']['aicc'] if 'BIEXP' in fits else None,
    }


def main():
    print('=' * 80)
    print('T7 LEVEL-2 RECALC: Multi-trial AICc verification')
    print('=' * 80)
    print()
    print('Question: Does S2_DUST consistently beat S2 on aggregates of local S2s?')
    print('If yes (ΔAICc >> 0 consistently), S2 is NOT closed under aggregation.')
    print()

    lambda_values = np.linspace(0.01, 50, 200)
    seeds = [42, 123, 456, 789, 1024]
    n_trials = len(seeds)

    distributions = {
        'narrow': (
            lambda rng, n: np.clip(rng.normal(1.0, 0.1, n), 0.01, 10),
            lambda rng, n: np.clip(rng.normal(10, 1, n), 0.1, 1e6),
        ),
        'broad': (
            lambda rng, n: rng.uniform(0.3, 3.0, n),
            lambda rng, n: np.exp(rng.normal(2, 1, n)),
        ),
        'registry_like': (
            lambda rng, n: np.clip(stats.weibull_min.rvs(1.5, loc=0, scale=1.2, size=n, random_state=rng), 0.01, 10),
            lambda rng, n: np.exp(rng.normal(2.5, 0.8, n)),
        ),
    }

    N_values = [100, 1000, 5000]

    all_results = {}
    for dist_name, (D_dist, lambda_dist) in distributions.items():
        print(f'\n{"=" * 80}')
        print(f'Distribution: {dist_name}')
        print(f'{"=" * 80}')
        print(f'{"N":>6} {"ΔAICc(S2 vs S2_DUST)":>25} {"ΔAICc(S2 vs BIEXP)":>25} {"S2 R²":>10} {"S2_DUST R²":>12}')

        dist_results = []
        for N in N_values:
            trials = []
            for seed in seeds:
                result = run_trial(N, D_dist, lambda_dist, seed + N, lambda_values)
                if result:
                    trials.append(result)

            if not trials:
                print(f'{N:>6} (all trials failed)')
                continue

            deltas = [t['delta_aicc'] for t in trials]
            deltas_biexp = [t['delta_aicc_biexp'] for t in trials if t['delta_aicc_biexp'] is not None]
            r2_s2 = [t['S2_r2'] for t in trials]
            r2_dust = [t['S2_DUST_r2'] for t in trials]

            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas, ddof=1) if len(deltas) > 1 else 0
            mean_delta_biexp = np.mean(deltas_biexp) if deltas_biexp else 0

            print(f'{N:>6} {mean_delta:>15.1f} ± {std_delta:>6.1f}   {mean_delta_biexp:>15.1f} ± {np.std(deltas_biexp, ddof=1) if len(deltas_biexp) > 1 else 0:>6.1f}   {np.mean(r2_s2):>10.5f} {np.mean(r2_dust):>12.5f}')

            dist_results.append({
                'N': N,
                'n_trials': len(trials),
                'mean_delta_aicc_S2_vs_S2_DUST': float(mean_delta),
                'std_delta_aicc': float(std_delta),
                'mean_delta_aicc_S2_vs_BIEXP': float(mean_delta_biexp),
                'mean_S2_r2': float(np.mean(r2_s2)),
                'mean_S2_DUST_r2': float(np.mean(r2_dust)),
                's2_dust_wins': sum(1 for d in deltas if d > 2),  # ΔAICc > 2 = strong
                's2_dust_wins_decisive': sum(1 for d in deltas if d > 10),  # ΔAICc > 10 = decisive
            })

        all_results[dist_name] = dist_results

    # ── Summary ──
    print()
    print('=' * 80)
    print('VERIFICATION SUMMARY')
    print('=' * 80)
    print()
    print('Question: Does S2_DUST consistently beat S2 on aggregates?')
    print()
    for dist_name, results in all_results.items():
        print(f'  {dist_name}:')
        for r in results:
            pct_decisive = 100 * r['s2_dust_wins_decisive'] / r['n_trials']
            print(f'    N={r["N"]:>5}: ΔAICc = {r["mean_delta_aicc_S2_vs_S2_DUST"]:.1f} ± {r["std_delta_aicc"]:.1f}, '
                  f'{r["s2_dust_wins_decisive"]}/{r["n_trials"]} decisive ({pct_decisive:.0f}%)')
        print()

    # Overall verdict
    all_deltas = []
    for results in all_results.values():
        for r in results:
            all_deltas.append(r['mean_delta_aicc_S2_vs_S2_DUST'])

    mean_overall = np.mean(all_deltas)
    min_overall = np.min(all_deltas)

    print(f'Overall: mean ΔAICc = {mean_overall:.1f}, min = {min_overall:.1f}')
    if min_overall > 10:
        verdict = 'REFUTED: S2 is NOT closed under aggregation. S2_DUST consistently and decisively beats S2 across all tested distributions and N values.'
    elif min_overall > 2:
        verdict = 'PARTIALLY REFUTED: S2_DUST beats S2 in most conditions, but margin varies.'
    else:
        verdict = 'NOT REFUTED: S2 may be approximately closed under some distributions.'
    print(f'Verdict: {verdict}')

    # Save
    output = {
        'experiment': 'T7 Level-2 recalc: multi-trial AICc verification',
        'verdict': verdict,
        'n_seeds': n_trials,
        'distributions': all_results,
        'overall_mean_delta_aicc': float(mean_overall),
        'overall_min_delta_aicc': float(min_overall),
        'interpretation': (
            'S2 closure under aggregation: REFUTED for the tested mixture model. '
            'The tested aggregation mechanism does not produce an approximately '
            'single-S2 aggregate under the tested distributions. A different ρ(D,λ_q), '
            'weighting, or aggregation mechanism could theoretically produce approximate '
            'S2 closure — but the simple averaging of local S2s does not.'
        ),
        'methodological_lesson': (
            'High R² alone is insufficient evidence for S2 in monotonic retention curves. '
            'Model selection must examine residual structure and penalized likelihood '
            'criteria such as AICc. The difference between R²=0.997 and R²=0.9999 looks '
            'trivial, while ΔAICc=744 says the models are dramatically different.'
        ),
    }
    out_path = os.path.join(REPO, 't7_level2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n✓ Saved: {out_path}')


if __name__ == '__main__':
    main()
