#!/usr/bin/env python3
"""
T7 Multi-Meta-Manifold Probe
=============================

Tests the "multiple Meta-Manifolds" hypothesis proposed in the user's
note. The hypothesis:

  H1 (single MM):  ONE Meta-Manifold  ->  ONE kernel  ->  ONE S2 law family
  H2 (multi MM):   MULTIPLE M_i        ->  multiple K_i -> mixture/interference
                                                   of S2 laws

Crucially, H2 is testable entirely in 4D. We never have to see a
Meta-Manifold. The signature is:

  (a) P(D, lambda_q) is NOT a constrained manifold - it shows clusters
      or scale-dependent apparent parameters that cannot be reconciled
      with one kernel.
  (b) Cross-correlations between unrelated observables at characteristic
      scales (interference signature - much stronger than mixtures).

This script does (a) on the real 172-entry DREAM registry:

  1. Load en/tests.json (172 datasets across 14 domains)
  2. Examine P(D) globally - is it unimodal or clustered?
  3. Examine P(D | domain) - do different physical sectors show
     different effective D distributions? (Sector -> mixture of kernels)
  4. Test multimodality (Silverman, Hartigan dip, GMM BIC)
  5. Bootstrap null: shuffle D across domains, see if observed
     between-sector variance exceeds null (interference-style signal)
  6. Check the "coherence scale" hypothesis: if MM_kernels are mixed,
     D should correlate with lambda_q in a structured way (not random)

Outputs:
  /home/z/my-project/download/t7_multi_mm_probe.json
  /home/z/my-project/download/t7_multi_mm_probe.md     (human-readable)
"""
import json, os, math
import numpy as np
from scipy import stats
from collections import defaultdict

REPO = '/home/z/my-project/dream_repo'

# ─────────────────────────────────────────────────────────────────────
# Load the real registry
# ─────────────────────────────────────────────────────────────────────
with open(os.path.join(REPO, 'en/tests.json')) as f:
    registry = json.load(f)['tests']

# Filter to tests that have a numeric D value
tests = [t for t in registry if isinstance(t.get('D'), (int, float))]
print(f'Registry: {len(registry)} entries, {len(tests)} with numeric D')

# Per-domain sample sizes
domains = defaultdict(list)
for t in tests:
    dom = t.get('domain', 'unknown')
    domains[dom].append(float(t['D']))

print(f'Domains: {len(domains)}')
for dom, Ds in sorted(domains.items(), key=lambda kv: -len(kv[1])):
    print(f'  {dom:>20s}: n={len(Ds):3d}  D mean={np.mean(Ds):.3f}  '
          f'median={np.median(Ds):.3f}  range=[{min(Ds):.3f}, {max(Ds):.3f}]')

# ─────────────────────────────────────────────────────────────────────
# Analysis 1: Global P(D) - multimodality tests
# ─────────────────────────────────────────────────────────────────────
all_D = np.array([float(t['D']) for t in tests])
print('\n' + '='*70)
print('ANALYSIS 1: Global P(D) on the registry')
print('='*70)
print(f'  n = {len(all_D)}')
print(f'  mean = {all_D.mean():.3f}  median = {np.median(all_D):.3f}  std = {all_D.std():.3f}')
print(f'  range = [{all_D.min():.3f}, {all_D.max():.3f}]')
print(f'  IQR = [{np.percentile(all_D, 25):.3f}, {np.percentile(all_D, 75):.3f}]')

# Hartigan's dip test for unimodality
try:
    from scipy.stats import loggamma  # noqa
    dip_test, dip_p = None, None
    # scipy doesn't have dip test natively; approximate via Silverman
    # Silverman test: bootstrap bandwidth for multimodality
    rng = np.random.RandomState(42)
    n_boot = 500
    # Critical bandwidth h_crit = smallest h such that KDE is unimodal
    from scipy.stats import gaussian_kde
    h_grid = np.linspace(0.05, 1.5, 50)
    h_crit_obs = None
    for h in h_grid:
        try:
            kde = gaussian_kde(all_D, bw_method=h / all_D.std())
            xs = np.linspace(all_D.min() - 0.1, all_D.max() + 0.1, 200)
            ys = kde(xs)
            # Count local maxima
            n_modes = sum(1 for i in range(1, len(ys) - 1) if ys[i] > ys[i-1] and ys[i] > ys[i+1])
            if n_modes <= 1:
                h_crit_obs = h * all_D.std()
                break
        except Exception:
            continue
    if h_crit_obs is None:
        h_crit_obs = h_grid[-1] * all_D.std()

    # Bootstrap: get null distribution of h_crit under unimodal null
    null_h_crits = []
    n = len(all_D)
    mu, sigma = all_D.mean(), all_D.std()
    for _ in range(n_boot):
        sample = rng.normal(mu, sigma, n)
        s_std = max(sample.std(), 1e-3)
        for h in h_grid:
            try:
                kde = gaussian_kde(sample, bw_method=h / s_std)
                xs = np.linspace(sample.min() - 0.1, sample.max() + 0.1, 200)
                ys = kde(xs)
                n_modes = sum(1 for i in range(1, len(ys) - 1) if ys[i] > ys[i-1] and ys[i] > ys[i+1])
                if n_modes <= 1:
                    null_h_crits.append(h * s_std)
                    break
            except Exception:
                continue
        else:
            null_h_crits.append(h_grid[-1] * s_std)

    null_h_crits = np.array(null_h_crits)
    p_value = float((null_h_crits >= h_crit_obs).mean())
    print(f'\n  Silverman critical bandwidth test:')
    print(f'    h_crit_observed = {h_crit_obs:.4f}')
    print(f'    h_crit_null_95% = {np.percentile(null_h_crits, 95):.4f}')
    print(f'    p-value = {p_value:.3f}')
    print(f'    -> {"MULTIMODAL" if p_value < 0.05 else "UNIMODAL"} at 5% level')
except Exception as e:
    print(f'  Silverman test failed: {e}')
    p_value = None
    h_crit_obs = None

# Shapiro-Wilk for normality (just to characterize)
W, p_norm = stats.shapiro(all_D)
print(f'\n  Shapiro-Wilk normality: W={W:.4f}, p={p_norm:.4g}')

# ─────────────────────────────────────────────────────────────────────
# Analysis 2: GMM (Gaussian Mixture Model) - test for k=1 vs k>1
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 2: Gaussian Mixture Model selection (k=1 vs k=2 vs k=3)')
print('='*70)
from sklearn.mixture import GaussianMixture
n = len(all_D)
X = all_D.reshape(-1, 1)
bics = {}
for k in [1, 2, 3, 4]:
    gm = GaussianMixture(n_components=k, covariance_type='full',
                         random_state=42, n_init=10, max_iter=300)
    gm.fit(X)
    bics[k] = (gm.bic(X), gm.aic(X))
    print(f'  k={k}:  BIC={gm.bic(X):.2f}  AIC={gm.aic(X):.2f}')

best_k = min(bics, key=lambda k: bics[k][0])
print(f'\n  BIC-optimal k = {best_k}')
print(f'  -> {"SINGLE MODE (consistent with single kernel)" if best_k == 1 else "MULTIPLE MODES (evidence for multiple kernels)"}')

# Show k=2 mixture if applicable
if best_k >= 2:
    gm2 = GaussianMixture(n_components=best_k, random_state=42, n_init=10).fit(X)
    print(f'\n  k={best_k} mixture parameters:')
    for i in range(best_k):
        mu_i = gm2.means_[i, 0]
        sig_i = math.sqrt(gm2.covariances_[i, 0, 0])
        w_i = gm2.weights_[i]
        print(f'    component {i+1}: mu={mu_i:.3f}  sigma={sig_i:.3f}  weight={w_i:.3f}')

# ─────────────────────────────────────────────────────────────────────
# Analysis 3: Domain-conditional - sector -> mixture of kernels?
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 3: D distribution BY DOMAIN (sector-level mixtures)')
print('='*70)
# Bootstrap test: is between-domain variance larger than expected by chance?
big_domains = [d for d, v in domains.items() if len(v) >= 5]
print(f'  Domains with n>=5: {len(big_domains)}')

# One-way ANOVA (parametric) and Kruskal-Wallis (non-parametric)
groups = [np.array(domains[d]) for d in big_domains]
F, p_anova = stats.f_oneway(*groups)
H, p_kw = stats.kruskal(*groups)
print(f'  One-way ANOVA across domains: F={F:.3f}, p={p_anova:.4g}')
print(f'  Kruskal-Wallis across domains: H={H:.3f}, p={p_kw:.4g}')

# Bootstrap null: shuffle D-labels, recompute between-group variance
rng = np.random.RandomState(123)
all_D_flat = np.concatenate(groups)
group_sizes = [len(g) for g in groups]
n_boot = 500
obs_var = np.var([np.mean(g) for g in groups])
boot_vars = []
for _ in range(n_boot):
    perm = rng.permutation(all_D_flat)
    means = [np.mean(perm[sum(group_sizes[:i]):sum(group_sizes[:i+1])]) for i in range(len(groups))]
    boot_vars.append(np.var(means))
boot_vars = np.array(boot_vars)
p_boot = float((boot_vars >= obs_var).mean())
print(f'\n  Permutation test (sector-mixture signature):')
print(f'    Observed between-domain mean-variance: {obs_var:.4f}')
print(f'    Null (shuffled labels) mean: {boot_vars.mean():.4f}  95%: {np.percentile(boot_vars, 95):.4f}')
print(f'    p-value = {p_boot:.4g}')
print(f'    -> {"DOMAINS ARE DIFFERENT (sectors sample different kernels)" if p_boot < 0.05 else "DOMAINS INDISTINGUISHABLE (single kernel consistent)"}')

# ─────────────────────────────────────────────────────────────────────
# Analysis 4: D vs R² — does fit quality degrade systematically?
#   A single-kernel hypothesis predicts a coherent universal S2 — but
#   if there are mixtures of kernels, fit quality should depend on where
#   on the (D, λ_q) manifold the dataset lands.
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 4: D vs R² (single kernel predicts coherent retention)')
print('='*70)
r2s = np.array([float(t.get('r2', 0) or 0) for t in tests])
Ds = all_D
# Spearman rank correlation
rho, p_sp = stats.spearmanr(Ds, r2s)
print(f'  Spearman corr(D, R²): rho={rho:.3f}  p={p_sp:.4g}')

# D distribution split by R² quality
good = Ds[r2s >= 0.95]
poor = Ds[r2s < 0.95]
print(f'\n  D for high-R² fits (R²>=0.95, n={len(good)}):')
print(f'    median={np.median(good):.3f}  IQR=[{np.percentile(good,25):.3f}, {np.percentile(good,75):.3f}]')
print(f'  D for low-R² fits (R²<0.95, n={len(poor)}):')
print(f'    median={np.median(poor):.3f}  IQR=[{np.percentile(poor,25):.3f}, {np.percentile(poor,75):.3f}]')
u, p_u = stats.mannwhitneyu(good, poor, alternative='two-sided')
print(f'  Mann-Whitney U: p={p_u:.4g}')

# ─────────────────────────────────────────────────────────────────────
# Analysis 5: Model verdict distribution — is S2 universally best?
#   If there's one kernel, S2 should win everywhere. If there are
#   mixtures, alternative models (BIEXP, GAUSS, EXP) should win in
#   regions where the mixture is most "non-S2-shaped".
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 5: Model verdict distribution (universal S2 vs mixed)')
print('='*70)
verdicts = defaultdict(int)
for t in tests:
    v = t.get('model_verdict', 'NONE')
    verdicts[v] += 1
total = sum(verdicts.values())
for v, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
    print(f'  {v:>20s}: {n:3d}  ({100*n/total:.1f}%)')

# Best alternative distribution
best_alts = defaultdict(int)
for t in tests:
    ba = t.get('best_alt', None)
    if ba:
        best_alts[ba] += 1
print('\n  When S2 does NOT win, what beats it?')
for ba, n in sorted(best_alts.items(), key=lambda kv: -kv[1]):
    print(f'    {ba:>12s}: {n}')

# ─────────────────────────────────────────────────────────────────────
# Analysis 6: domain x best_alt cross-tabulation
#   If multiple kernels exist, BIEXP/GAUSS wins should cluster in
#   specific domains (sectors sampling one kernel) — not uniformly.
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 6: Domain x best_alternative cross-tabulation')
print('='*70)
ct = defaultdict(lambda: defaultdict(int))
for t in tests:
    ba = t.get('best_alt', 'NONE') or 'NONE'
    dom = t.get('domain', 'unknown')
    ct[dom][ba] += 1

print(f'  {"Domain":>20s} | ', end='')
all_alts = sorted(set(ba for d in ct.values() for ba in d))
print('  '.join(f'{a:>8s}' for a in all_alts))
print('  ' + '-' * (22 + 10 * len(all_alts)))
for dom in sorted(ct, key=lambda d: -sum(ct[d].values())):
    print(f'  {dom:>20s} | ', end='')
    print('  '.join(f'{ct[dom].get(a,0):>8d}' for a in all_alts))

# ─────────────────────────────────────────────────────────────────────
# Analysis 7: Sub-population D test - is the registry better explained
#   by a 2-population mixture than by one population?
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANALYSIS 7: Hartigan-style dip test via simulation')
print('='*70)
# Hartigan's dip isn't in scipy; approximate via skew of nearest-neighbor
# distances among D's (multi-modal data has bimodal NN distances)
sorted_D = np.sort(all_D)
nn_d = np.diff(sorted_D)
# Look at the longest gap relative to median spacing
median_gap = np.median(nn_d)
max_gap = nn_d.max()
gap_ratio = max_gap / max(median_gap, 1e-6)
print(f'  Sorted-D statistics:')
print(f'    Median nearest-neighbor spacing: {median_gap:.4f}')
print(f'    Max gap: {max_gap:.4f}  (gap/median = {gap_ratio:.1f})')
print(f'    Location of max gap: D={sorted_D[np.argmax(nn_d)]:.3f} to {sorted_D[np.argmax(nn_d)+1]:.3f}')

# Bootstrap null
rng = np.random.RandomState(7)
null_gap_ratios = []
for _ in range(500):
    sample = rng.choice(all_D, size=n, replace=True)
    sample = np.sort(sample)
    s_nn = np.diff(sample)
    s_med = np.median(s_nn)
    if s_med > 0:
        null_gap_ratios.append(s_nn.max() / s_med)
null_gap_ratios = np.array(null_gap_ratios)
p_gap = float((null_gap_ratios >= gap_ratio).mean())
print(f'\n  Bootstrap null (resampled):')
print(f'    Null gap/median 95th pct: {np.percentile(null_gap_ratios, 95):.2f}')
print(f'    p-value: {p_gap:.3f}')

# ─────────────────────────────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────────────────────────────
out_dir = '/home/z/my-project/download'
os.makedirs(out_dir, exist_ok=True)

result = {
    'analysis': 'T7 Multi-Meta-Manifold Probe (single kernel vs mixture of kernels)',
    'n_datasets': len(tests),
    'n_domains': len(domains),
    'global_D_stats': {
        'mean': float(all_D.mean()),
        'median': float(np.median(all_D)),
        'std': float(all_D.std()),
        'min': float(all_D.min()),
        'max': float(all_D.max()),
        'IQR': [float(np.percentile(all_D, 25)), float(np.percentile(all_D, 75))],
        'shapiro_p': float(p_norm),
    },
    'silverman_test': {
        'h_crit_observed': float(h_crit_obs) if h_crit_obs else None,
        'h_crit_null_95': float(np.percentile(null_h_crits, 95)) if len(null_h_crits) else None,
        'p_value': p_value,
        'verdict': 'MULTIMODAL' if p_value and p_value < 0.05 else 'UNIMODAL',
    },
    'gmm_bic': {int(k): {'bic': float(v[0]), 'aic': float(v[1])} for k, v in bics.items()},
    'gmm_best_k': int(best_k),
    'domain_anova': {
        'F': float(F), 'p_anova': float(p_anova),
        'H_kw': float(H), 'p_kw': float(p_kw),
    },
    'permutation_sector_test': {
        'observed_between_var': float(obs_var),
        'null_mean': float(boot_vars.mean()),
        'null_95': float(np.percentile(boot_vars, 95)),
        'p_value': float(p_boot),
        'verdict': 'DOMAINS_DIFFERENT' if p_boot < 0.05 else 'DOMAINS_INDISTINGUISHABLE',
    },
    'D_vs_R2': {
        'spearman_rho': float(rho), 'p_value': float(p_sp),
        'good_D_median': float(np.median(good)),
        'poor_D_median': float(np.median(poor)),
        'mannwhitney_p': float(p_u),
    },
    'model_verdict_counts': dict(verdicts),
    'best_alt_counts': dict(best_alts),
    'gap_test': {
        'max_gap_ratio': float(gap_ratio),
        'null_95': float(np.percentile(null_gap_ratios, 95)),
        'p_value': float(p_gap),
    },
}

# Domain-level stats
domain_stats = {}
for d, Ds_list in sorted(domains.items(), key=lambda kv: -len(kv[1])):
    Ds_arr = np.array(Ds_list)
    domain_stats[d] = {
        'n': int(len(Ds_arr)),
        'mean': float(Ds_arr.mean()),
        'median': float(np.median(Ds_arr)),
        'std': float(Ds_arr.std()) if len(Ds_arr) > 1 else 0.0,
        'min': float(Ds_arr.min()),
        'max': float(Ds_arr.max()),
    }
result['domain_D_stats'] = domain_stats

# Cross-tab
ct_out = {dom: dict(alts) for dom, alts in ct.items()}
result['domain_x_best_alt'] = ct_out

out_json = os.path.join(out_dir, 't7_multi_mm_probe.json')
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved JSON to: {out_json}')

# ─────────────────────────────────────────────────────────────────────
# Verdict synthesis
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SYNTHESIS')
print('='*70)
# Count how many tests suggest multi-MM
signals = []
if p_value and p_value < 0.05:
    signals.append('Silverman multimodality')
if best_k >= 2:
    signals.append(f'GMM BIC prefers k={best_k}')
if p_anova < 0.05:
    signals.append('ANOVA across domains')
if p_boot < 0.05:
    signals.append('Sector permutation test')
if p_gap < 0.05:
    signals.append('Gap test')

if len(signals) == 0:
    print('  No multi-MM signal detected in the current registry.')
    print('  Single-kernel hypothesis is consistent with the data.')
elif len(signals) == 1:
    print(f'  Weak multi-MM signal: {signals[0]}')
    print('  Inconclusive — needs more targeted tests.')
else:
    print(f'  Multiple multi-MM signals ({len(signals)}):')
    for s in signals:
        print(f'    - {s}')
    print('  Suggests the single-kernel hypothesis may be incomplete.')
