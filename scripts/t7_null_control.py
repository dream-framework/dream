#!/usr/bin/env python3
"""
T7 Null-Model Control Experiment (P1 decisive test)
====================================================

The user's crucial critique: P1 (D_eff varies with scale) is only meaningful
if the observed D-variation EXCEEDS what a pure single-S2 process would
produce under the same fitting procedure.

This script:
  1. Generates synthetic single-S2 data: R(λ) = exp[-(λ/λ_q)^D]
     with known D, λ_q, and realistic noise/sampling.
  2. Passes each synthetic dataset through the SAME 3-scale fitting pipeline
     (fine/medium/coarse ACF ranges → fit S2 → record D).
  3. Computes ΔD_null = D_max - D_min across the 3 scales for each synthetic.
  4. Compares ΔD_null distribution to ΔD_real (from the 20 real datasets).

If ΔD_real >> ΔD_null  → P1 is strong evidence for T7.
If ΔD_null ≈ ΔD_real   → P1 collapses (variation is a fitting artifact).

Also runs:
  - P2 paired Wilcoxon test on ΔAICc_fine vs ΔAICc_full
  - P3 hierarchical variance decomposition
"""
import os, sys, json, csv, io
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import m_s2, m_s2_dust
from t7_hypothesis_test import (
    refetch_values, fit_s2_on_range, fit_single_s2_aic, fit_s2_dust_on_range
)


# ── Synthetic single-S2 generator ──────────────────────────────────

def generate_synthetic_s2(n=500, D_true=1.0, lambda_q_true=20.0, noise_sd=0.02, seed=0):
    """Generate a synthetic time series whose ACF will follow a single S2.

    Strategy: generate values such that the empirical ACF of |demeaned values|
    approximates exp[-(λ/λ_q)^D]. The cleanest way is to generate an AR(1)-like
    process with stretched-exponential autocorrelation. We use a direct method:
    construct a series with the desired autocorrelation by inverse-transforming
    a Gaussian process through the target ACF.

    For simplicity and robustness, we use a rejection-based approach:
    generate x_t = sum_{s<t} w_s * ε_{t-s} where weights are chosen so the
    empirical ACF matches the target. In practice, we just generate many points
    from a process with known autocorrelation structure and verify the ACF.

    A simpler and more honest approach: directly synthesize an ACF curve from
    the S2 formula, add noise, and fit. This tests whether the FITTING
    procedure introduces spurious D-variation when the true signal is pure S2.
    """
    rng = np.random.RandomState(seed)
    # Generate ACF points: λ = 0, 1, 2, ..., max_lag
    max_lag = 200  # match real-data pipeline
    t = np.arange(max_lag, dtype=float)
    R_true = np.exp(-np.power(t / lambda_q_true, D_true))
    # Add realistic noise (similar to what empirical ACFs have)
    R_noisy = R_true + rng.normal(0, noise_sd, max_lag)
    # ACF must start at 1 and decay; clip
    R_noisy[0] = 1.0
    R_noisy = np.clip(R_noisy, -0.5, 1.0)
    return t, R_noisy, D_true, lambda_q_true


def generate_synthetic_timeseries_acf(n=2000, D_true=1.0, lambda_q_true=20.0, noise_sd=0.1, seed=0):
    """Generate an actual time series whose empirical ACF follows S2.

    Method: generate a long AR(1)-like process but with stretched-exponential
    memory kernel. We use a simpler trick: generate the time series as a
    moving average of white noise with S2-shaped weights.
    """
    rng = np.random.RandomState(seed)
    # Weights: w[s] = exp[-(s/λ_q)^D] - exp[-((s+1)/λ_q)^D]
    # This gives a process whose ACF is approximately S2
    max_lag = 500
    s = np.arange(max_lag, dtype=float)
    w = np.exp(-np.power(s / lambda_q_true, D_true))
    # Normalize weights
    w = w / np.sum(w)

    # Generate white noise
    eps = rng.normal(0, 1, n + max_lag)
    # Convolve
    x = np.zeros(n)
    for t in range(n):
        x[t] = np.sum(w * eps[t:t+max_lag])
    # Add measurement noise
    x = x + rng.normal(0, noise_sd, n)
    return x


# ── Run the null-model control ─────────────────────────────────────

def run_p1_null_control(B=500):
    """Generate B synthetic single-S2 datasets, run through 3-scale pipeline.

    Returns the distribution of ΔD_null (D_max - D_min across 3 scales)
    for synthetic data with KNOWN constant D.

    We test across a range of true D values (0.5, 1.0, 1.5, 2.0) to cover
    the regimes seen in real data.
    """
    print(f'P1 NULL-MODEL CONTROL: {B} synthetic single-S2 datasets')
    print(f'  Testing true D ∈ {{0.5, 1.0, 1.5, 2.0}}, λ_q = 20')
    print()

    true_D_values = [0.5, 1.0, 1.5, 2.0]
    null_delta_D = []

    for D_true in true_D_values:
        print(f'  True D = {D_true}...')
        for i in range(B // len(true_D_values)):
            seed = int(D_true * 10000 + i)
            # Generate synthetic ACF directly (faster, tests fitting pipeline)
            t, R_noisy, _, _ = generate_synthetic_s2(
                D_true=D_true, lambda_q_true=20.0, noise_sd=0.03, seed=seed)

            max_lag = float(t[-1])
            fine = fit_s2_on_range(t, R_noisy, 0, max_lag * 0.25)
            medium = fit_s2_on_range(t, R_noisy, max_lag * 0.25, max_lag * 0.75)
            coarse = fit_s2_on_range(t, R_noisy, max_lag * 0.5, max_lag)

            if not all([fine, medium, coarse]):
                continue

            d_range = max(fine['D'], medium['D'], coarse['D']) - min(fine['D'], medium['D'], coarse['D'])
            null_delta_D.append({'D_true': D_true, 'delta_D': d_range,
                                 'D_fine': fine['D'], 'D_medium': medium['D'], 'D_coarse': coarse['D']})

    return null_delta_D


def run_p1_real_data():
    """Run the 3-scale pipeline on the 20 real datasets (from t7_test_runner)."""
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    candidates = []
    seen_urls = set()
    for t in data['tests']:
        if t.get('D') is None or (t.get('r2') or 0) < 0.3: continue
        url = t.get('url', '')
        if url in seen_urls: continue
        if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org',
                                    'open-meteo', 'ndbc.noaa.gov', 'binance.com']):
            seen_urls.add(url)
            candidates.append(t)
    candidates = candidates[:20]

    real_delta_D = []
    for entry in candidates:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20: continue
        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)
        max_lag = float(t_arr[-1])
        fine = fit_s2_on_range(t_arr, R_arr, 0, max_lag * 0.25)
        medium = fit_s2_on_range(t_arr, R_arr, max_lag * 0.25, max_lag * 0.75)
        coarse = fit_s2_on_range(t_arr, R_arr, max_lag * 0.5, max_lag)
        if not all([fine, medium, coarse]): continue
        d_range = max(fine['D'], medium['D'], coarse['D']) - min(fine['D'], medium['D'], coarse['D'])
        real_delta_D.append({'name': entry['name'][:40], 'delta_D': d_range,
                             'D_fine': fine['D'], 'D_medium': medium['D'], 'D_coarse': coarse['D']})
    return real_delta_D


# ── P2: paired Wilcoxon test ───────────────────────────────────────

def run_p2_paired_test():
    """Paired Wilcoxon signed-rank test on ΔAICc_fine vs ΔAICc_full.

    For each dataset, we have:
      ΔAICc_fine = AICc(S2+dust) - AICc(S2) at fine scale  (negative = dust wins)
      ΔAICc_full = AICc(S2+dust) - AICc(S2) at full scale

    T7 predicts: ΔAICc_fine should be MORE negative than ΔAICc_full
    (multi-component more favored at finer resolution).

    Test: Wilcoxon signed-rank on (ΔAICc_full - ΔAICc_fine).
    If the median is significantly > 0, P2 is supported.
    """
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    candidates = []
    seen_urls = set()
    for t in data['tests']:
        if t.get('D') is None or (t.get('r2') or 0) < 0.3: continue
        url = t.get('url', '')
        if url in seen_urls: continue
        if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org',
                                    'open-meteo', 'ndbc.noaa.gov', 'binance.com']):
            seen_urls.add(url)
            candidates.append(t)
    candidates = candidates[:20]

    paired_diffs = []  # ΔAICc_full - ΔAICc_fine (positive = dust more favored at fine)
    for entry in candidates:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20: continue
        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)
        max_lag = float(t_arr[-1])

        s2_fine = fit_single_s2_aic(t_arr, R_arr, 0, max_lag * 0.25)
        dust_fine = fit_s2_dust_on_range(t_arr, R_arr, 0, max_lag * 0.25)
        s2_full = fit_single_s2_aic(t_arr, R_arr, 0, max_lag)
        dust_full = fit_s2_dust_on_range(t_arr, R_arr, 0, max_lag)
        if not all([s2_fine, dust_fine, s2_full, dust_full]): continue

        delta_fine = dust_fine['aic'] - s2_fine['aic']
        delta_full = dust_full['aic'] - s2_full['aic']
        diff = delta_full - delta_fine  # positive = dust MORE favored at fine
        paired_diffs.append(diff)

    if len(paired_diffs) < 5:
        return {'test': 'wilcoxon', 'n': len(paired_diffs), 'significant': False,
                'reason': 'too few pairs'}

    diffs = np.array(paired_diffs)
    # Wilcoxon signed-rank test (one-sided: median > 0)
    stat, p_value = stats.wilcoxon(diffs, alternative='greater')
    median_diff = float(np.median(diffs))
    mean_diff = float(np.mean(diffs))

    return {
        'test': 'wilcoxon_signed_rank',
        'n': len(paired_diffs),
        'median_diff': median_diff,
        'mean_diff': mean_diff,
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'direction': 'dust more favored at fine scale' if median_diff > 0 else 'dust less favored at fine scale',
    }


# ── P3: hierarchical variance test ──────────────────────────────────

def run_p3_hierarchical():
    """Hierarchical variance decomposition.

    For each dataset with multi-scale fits, decompose D variance into:
      - Between-scale variance (within same system, across probe scales)
      - Between-system variance (across different datasets)

    T7 predicts: between-scale variance should be substantial (D depends on scale).
    Alternative (universal D): between-scale variance should be ~0 (D is system property).

    We compute:
      V_between_scale = mean over systems of Var(D_fine, D_medium, D_coarse)
      V_between_system = Var of mean-D across systems
      Ratio = V_between_scale / V_between_system
    """
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    candidates = []
    seen_urls = set()
    for t in data['tests']:
        if t.get('D') is None or (t.get('r2') or 0) < 0.3: continue
        url = t.get('url', '')
        if url in seen_urls: continue
        if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org',
                                    'open-meteo', 'ndbc.noaa.gov', 'binance.com']):
            seen_urls.add(url)
            candidates.append(t)
    candidates = candidates[:20]

    system_D_means = []
    within_system_vars = []
    n_systems = 0
    for entry in candidates:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20: continue
        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)
        max_lag = float(t_arr[-1])
        fine = fit_s2_on_range(t_arr, R_arr, 0, max_lag * 0.25)
        medium = fit_s2_on_range(t_arr, R_arr, max_lag * 0.25, max_lag * 0.75)
        coarse = fit_s2_on_range(t_arr, R_arr, max_lag * 0.5, max_lag)
        if not all([fine, medium, coarse]): continue
        Ds = [fine['D'], medium['D'], coarse['D']]
        system_D_means.append(np.mean(Ds))
        within_system_vars.append(np.var(Ds, ddof=1))
        n_systems += 1

    if n_systems < 5:
        return {'test': 'hierarchical_variance', 'n_systems': n_systems,
                'significant': False, 'reason': 'too few systems'}

    V_within = float(np.mean(within_system_vars))  # between-scale, within-system
    V_between = float(np.var(system_D_means, ddof=1))  # between-system

    # T7: V_within should be substantial (D varies with scale within same system)
    # Universal-D null: V_within ≈ 0 (D is system property, doesn't vary with scale)
    ratio = V_within / V_between if V_between > 0 else float('inf')

    return {
        'test': 'hierarchical_variance',
        'n_systems': n_systems,
        'V_within_system': V_within,  # between-scale variance
        'V_between_system': V_between,
        'ratio_within_to_between': ratio,
        'interpretation': 'substantial within-system (between-scale) variance' if V_within > 0.1 * V_between else 'within-system variance small relative to between-system',
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('T7 NULL-MODEL CONTROL & REVISED TESTING')
    print('=' * 80)
    print()

    # ── P1: Null-model control ──
    print('═' * 80)
    print('P1 (REVISED): NULL-MODEL CONTROL')
    print('═' * 80)
    print('Question: Does observed D-variation exceed what pure single-S2')
    print('produces under the same fitting procedure?')
    print()

    null_results = run_p1_null_control(B=500)
    null_delta_D = [r['delta_D'] for r in null_results]
    real_results = run_p1_real_data()
    real_delta_D = [r['delta_D'] for r in real_results]

    print(f'\nNull model (synthetic single-S2): n={len(null_delta_D)}')
    print(f'  ΔD_null: mean={np.mean(null_delta_D):.3f}, median={np.median(null_delta_D):.3f}, '
          f'95th pct={np.percentile(null_delta_D, 95):.3f}, max={np.max(null_delta_D):.3f}')
    print(f'\nReal data: n={len(real_delta_D)}')
    print(f'  ΔD_real: mean={np.mean(real_delta_D):.3f}, median={np.median(real_delta_D):.3f}, '
          f'max={np.max(real_delta_D):.3f}')

    # Statistical comparison
    null_median = np.median(null_delta_D)
    real_median = np.median(real_delta_D)
    # What fraction of null exceeds real median?
    pct_null_exceeds_real = float(np.mean(np.array(null_delta_D) >= real_median))
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(real_delta_D, null_delta_D, alternative='greater')

    print(f'\nComparison:')
    print(f'  Real median ΔD ({real_median:.3f}) vs null median ΔD ({null_median:.3f})')
    print(f'  Ratio: {real_median / null_median:.1f}x')
    print(f'  Fraction of null ≥ real median: {pct_null_exceeds_real:.4f}')
    print(f'  Mann-Whitney U (one-sided, real > null): p = {u_p:.6f}')

    p1_pass = real_median > np.percentile(null_delta_D, 95) and u_p < 0.01
    p1_strength = real_median / null_median if null_median > 0 else float('inf')
    print(f'\n  P1 VERDICT: {"STRONG SUPPORT" if p1_pass else "WEAK/COLLAPSED"}')
    print(f'  Real ΔD is {p1_strength:.1f}x the null median')

    # ── P2: Paired test ──
    print('\n' + '═' * 80)
    print('P2 (REVISED): PAIRED STATISTICAL TEST')
    print('═' * 80)
    p2 = run_p2_paired_test()
    print(f'  Test: {p2["test"]}')
    print(f'  n = {p2["n"]} pairs')
    if 'median_diff' in p2:
        print(f'  Median (ΔAICc_full - ΔAICc_fine): {p2["median_diff"]:.2f}')
        print(f'  Mean: {p2["mean_diff"]:.2f}')
        print(f'  p-value: {p2["p_value"]:.4f}')
        print(f'  Direction: {p2["direction"]}')
        print(f'  P2 VERDICT: {"SUPPORTED" if p2["significant"] and p2["median_diff"] > 0 else "NOT SUPPORTED"}')
    else:
        print(f'  {p2.get("reason", "failed")}')

    # ── P3: Hierarchical variance ──
    print('\n' + '═' * 80)
    print('P3 (REVISED): HIERARCHICAL VARIANCE TEST')
    print('═' * 80)
    p3 = run_p3_hierarchical()
    print(f'  n_systems = {p3["n_systems"]}')
    if 'V_within_system' in p3:
        print(f'  V_within_system (between-scale): {p3["V_within_system"]:.4f}')
        print(f'  V_between_system: {p3["V_between_system"]:.4f}')
        print(f'  Ratio (within/between): {p3["ratio_within_to_between"]:.3f}')
        print(f'  Interpretation: {p3["interpretation"]}')
        p3_pass = p3['V_within_system'] > 0.1 * p3['V_between_system']
        print(f'  P3 VERDICT: {"SUPPORTED" if p3_pass else "NOT SUPPORTED"} (within-system variance substantial)')
    else:
        print(f'  {p3.get("reason", "failed")}')

    # ── P4: Reframe ──
    print('\n' + '═' * 80)
    print('P4 (REFRAMED): META-S2 FALSIFICATION (not T7 confirmation)')
    print('═' * 80)
    snap_path = os.path.join(REPO, 'meta_s2_snapshot.json')
    with open(snap_path) as f:
        snap = json.load(f)
    p_lil = snap.get('ks_p_lilliefors')
    print(f'  Lilliefors KS p = {p_lil}')
    print(f'  Interpretation: Meta-S2 (Weibull law for D distribution) is FALSIFIED.')
    print(f'  This does NOT independently confirm T7 — it only removes one alternative.')
    print(f'  P4 VERDICT: Meta-S2 falsified (supports dropping it, not confirming T7)')

    # ── Overall ──
    print('\n' + '═' * 80)
    print('REVISED T7 ASSESSMENT')
    print('═' * 80)
    print(f'  P1: {"STRONG" if p1_pass else "WEAK/COLLAPSED"} (real ΔD = {p1_strength:.1f}x null)')
    p2_str = 'SUPPORTED' if (p2.get('significant') and p2.get('median_diff', 0) > 0) else 'NOT SUPPORTED'
    print(f'  P2: {p2_str} (paired Wilcoxon p={p2.get("p_value", "?"):.4f})')
    p3_str = 'SUPPORTED' if (p3.get('V_within_system', 0) > 0.1 * p3.get('V_between_system', 1)) else 'NOT SUPPORTED'
    print(f'  P3: {p3_str} (within/between ratio = {p3.get("ratio_within_to_between", "?"):.3f})')
    print(f'  P4: Meta-S2 falsified (not T7 confirmation)')

    n_strong = sum([
        p1_pass,
        p2.get('significant', False) and p2.get('median_diff', 0) > 0,
        p3.get('V_within_system', 0) > 0.1 * p3.get('V_between_system', 1),
    ])
    print(f'\n  Strongly supported predictions: {n_strong}/3 (P4 is reframed)')
    if n_strong == 3:
        verdict = 'STRONGLY SUPPORTED (pending more datasets)'
    elif n_strong == 2:
        verdict = 'PRELIMINARILY SUPPORTED'
    elif n_strong == 1:
        verdict = 'PARTIALLY SUPPORTED'
    else:
        verdict = 'NOT SUPPORTED'
    print(f'\n  T7 VERDICT: {verdict}')

    # Save
    results = {
        'verdict': verdict,
        'n_strong_predictions': n_strong,
        'total_testable_predictions': 3,
        'P1': {
            'test': 'null_model_control',
            'real_median_delta_D': float(real_median),
            'null_median_delta_D': float(null_median),
            'null_95th_pct': float(np.percentile(null_delta_D, 95)),
            'ratio_real_to_null': float(p1_strength),
            'mann_whitney_p': float(u_p),
            'pass': bool(p1_pass),
            'n_null': len(null_delta_D),
            'n_real': len(real_delta_D),
        },
        'P2': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in p2.items()} if 'median_diff' in p2 else p2,
        'P3': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in p3.items()} if 'V_within_system' in p3 else p3,
        'P4': {
            'test': 'meta_s2_falsification',
            'lilliefors_p': float(p_lil) if p_lil else None,
            'interpretation': 'Meta-S2 falsified; does not independently confirm T7',
        },
        'honest_summary': (
            f'T7 INTERFERENCE HYPOTHESIS — {verdict}. '
            f'P1 (scale-dependent D_eff): real ΔD = {p1_strength:.1f}x null median '
            f'(Mann-Whitney p={u_p:.6f}), {"STRONG" if p1_pass else "weak"}. '
            f'P2 (multi-component at fine scale): {p2_str} (Wilcoxon p={p2.get("p_value", 0):.4f}). '
            f'P3 (hierarchical variance): {p3_str} (within/between={p3.get("ratio_within_to_between", 0):.3f}). '
            f'P4: Meta-S2 falsified (Lilliefors p={p_lil}), but this does not independently confirm T7. '
            f'Further null-model and hierarchical testing required to distinguish interference from '
            f'alternative explanations such as finite-sample fitting effects.'
        ),
    }
    out = os.path.join(REPO, 't7_test_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Saved: {out}')


if __name__ == '__main__':
    main()
