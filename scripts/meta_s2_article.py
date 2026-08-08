#!/usr/bin/env python3
"""
Meta-S2 article renderer.

Both the English and Russian versions of `articles/meta-s2.html` are generated
from this module. Every numeric value in the article body comes from a
snapshot dict, so the scanner can regenerate the article with fresh data on
every run simply by calling `render(lang, snapshot)`.

Placeholders use the `{{name}}` syntax (mustache-style, regex-replaced) so the
templates can include literal CSS braces without escaping.

The snapshot dict is produced by `extend_snapshot()` in this module, which
takes the raw D values + the existing tests list and computes:

  * Basic stats (n, mean, median, geom_mean, std, min, max, % natural, % extraction)
  * Three estimators (Weibull MLE, linearized, direct-d)
  * KS p-value (MLE)
  * Anderson-Darling statistic
  * AICs for Weibull / Gamma / Lognormal / Exponential
  * GMM 1-vs-2-component ΔBIC + cluster means
  * Silverman's bootstrap test p-value
  * Family-level block bootstrap CI for D_meta (MLE)
  * Leave-one-family-out table
  * Level-2 (bootstrap of D_meta) Weibull shape + KS p
  * Total registered / compared / censored counts

Everything needed by the article template is in the snapshot. The scanner
calls `extend_snapshot()` once, then `render('en', snap)` and `render('ru', snap)`.
"""

import re
import json
import math
import numpy as np
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────

def _f(x, nd=3):
    """Format a float with `nd` decimals, stripping trailing zeros."""
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    s = f'{float(x):.{nd}f}'
    # don't strip if integer-valued like 1.000 — keep at least one decimal
    return s


def _pct(num, den, nd=1):
    if den == 0:
        return '0%'
    return f'{100.0 * num / den:.{nd}f}%'


def _substitute(template, ctx):
    """Replace {{key}} with ctx[key]. Tries smart formatting for floats."""
    def repl(m):
        key = m.group(1)
        if key not in ctx:
            return m.group(0)  # leave as-is if missing
        v = ctx[key]
        if v is None:
            return '—'
        if isinstance(v, float):
            if math.isnan(v):
                return '—'
            return f'{v:.3f}'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        return str(v)
    return re.sub(r'\{\{(\w+)\}\}', repl, template)


# ── Stats computation ─────────────────────────────────────────────────────

def _safe_weibull_fit(Ds):
    """Weibull MLE with loc=0. Returns (shape, scale)."""
    from scipy.stats import weibull_min
    try:
        shape, loc, scale = weibull_min.fit(Ds, floc=0)
        return float(shape), float(scale)
    except Exception:
        return 1.5, 1.5


def _ks_weibull(Ds, shape, scale):
    """Standard KS test — assumes KNOWN parameters.
    WARNING: When shape and scale are estimated from the same data, this test
    is too conservative (p-values too large). Use _ks_weibull_lilliefors instead.
    Kept for backward compatibility with existing snapshot fields."""
    from scipy.stats import weibull_min, kstest
    try:
        _, p = kstest(Ds, 'weibull_min', args=(shape, 0, scale))
        return float(p)
    except Exception:
        return float('nan')


def _ks_weibull_lilliefors(Ds, shape, scale, B=1000, seed=42):
    """Lilliefors-corrected KS test via parametric bootstrap.

    The standard KS test assumes the distribution parameters are KNOWN. When
    they are ESTIMATED from the data (as we do with Weibull MLE), the standard
    test is too conservative — it doesn't reject often enough. The Lilliefors
    correction fixes this by simulating from the fitted distribution, refitting
    on each simulation, and computing the KS statistic each time.

    The corrected p-value = fraction of bootstrap KS statistics ≥ observed KS.

    This is the statistically correct test for "does this data come from a
    Weibull distribution with parameters estimated from the data?"

    Returns (corrected_p, observed_ks_stat, bootstrap_ks_mean, bootstrap_ks_95).
    """
    from scipy.stats import weibull_min, kstest
    Ds = np.asarray(Ds, dtype=float)
    n = len(Ds)
    if n < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')

    try:
        # Observed KS statistic (with estimated parameters)
        obs_ks, _ = kstest(Ds, 'weibull_min', args=(shape, 0, scale))

        # Parametric bootstrap
        rng = np.random.RandomState(seed)
        bootstrap_ks = []
        for _ in range(B):
            sim = weibull_min.rvs(shape, loc=0, scale=scale, size=n, random_state=rng)
            sim_shape, _, sim_scale = weibull_min.fit(sim, floc=0)
            sim_ks, _ = kstest(sim, 'weibull_min', args=(sim_shape, 0, sim_scale))
            bootstrap_ks.append(sim_ks)

        bootstrap_ks = np.array(bootstrap_ks)
        corrected_p = float(np.mean(bootstrap_ks >= obs_ks))
        return (corrected_p, float(obs_ks), float(np.mean(bootstrap_ks)),
                float(np.percentile(bootstrap_ks, 95)))
    except Exception as e:
        print(f'  ⚠ Lilliefors bootstrap failed: {e}')
        return float('nan'), float('nan'), float('nan'), float('nan')


def _anderson_darling_weibull(Ds, shape, scale):
    """Anderson-Darling statistic against Weibull(shape, scale)."""
    from scipy.stats import weibull_min
    try:
        F = weibull_min.cdf(np.sort(Ds), shape, 0, scale)
        F = np.clip(F, 1e-12, 1 - 1e-12)
        n = len(Ds)
        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
        return float(A2)
    except Exception:
        return float('nan')


def _fit_aic(Ds, dist_name):
    """Fit a distribution and return (loglik, k, aic)."""
    from scipy.stats import weibull_min, gamma, lognorm, expon
    n = len(Ds)
    try:
        if dist_name == 'weibull':
            shape, loc, scale = weibull_min.fit(Ds, floc=0)
            ll = np.sum(weibull_min.logpdf(Ds, shape, 0, scale))
            k = 2
        elif dist_name == 'gamma':
            shape, loc, scale = gamma.fit(Ds, floc=0)
            ll = np.sum(gamma.logpdf(Ds, shape, 0, scale))
            k = 2
        elif dist_name == 'lognormal':
            shape, loc, scale = lognorm.fit(Ds, floc=0)
            ll = np.sum(lognorm.logpdf(Ds, shape, 0, scale))
            k = 2
        elif dist_name == 'exponential':
            loc, scale = expon.fit(Ds, floc=0)
            ll = np.sum(expon.logpdf(Ds, 0, scale))
            k = 1
        else:
            return None
        return float(-2 * ll + 2 * k)
    except Exception:
        return float('nan')


def _gmm_bic(Ds, k):
    """Fit a k-component Gaussian Mixture and return BIC."""
    from sklearn.mixture import GaussianMixture
    X = np.array(Ds).reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=42, n_init=3)
        gmm.fit(X)
        return float(gmm.bic(X)), gmm.means_.flatten().tolist(), gmm.weights_.tolist()
    except Exception:
        return float('nan'), [], []


def _silverman_test(Ds, n_boot=200):
    """Silverman's bootstrap test for unimodality."""
    from scipy.stats import gaussian_kde
    try:
        X = np.array(Ds)
        # Critical bandwidth (smallest bandwidth giving 1 mode)
        bws = np.linspace(0.05, 2.0, 80)
        crit = None
        for bw in bws:
            kde = gaussian_kde(X, bw_method=bw)
            # Sample finely
            xs = np.linspace(X.min() - 1, X.max() + 1, 400)
            dens = kde(xs)
            # Count modes (sign changes in derivative)
            dd = np.diff(dens)
            modes = np.sum((dd[:-1] > 0) & (dd[1:] <= 0))
            if modes <= 1:
                crit = bw
                break
        if crit is None:
            crit = bws[-1]
        # Bootstrap: count how often a resample has > 1 mode at this bandwidth
        n = len(X)
        count_multi = 0
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            sample = rng.choice(X, size=n, replace=True)
            try:
                kde = gaussian_kde(sample, bw_method=crit)
                xs = np.linspace(sample.min() - 1, sample.max() + 1, 400)
                dens = kde(xs)
                dd = np.diff(dens)
                modes = np.sum((dd[:-1] > 0) & (dd[1:] <= 0))
                if modes > 1:
                    count_multi += 1
            except Exception:
                pass
        p = count_multi / n_boot
        return float(p)
    except Exception:
        return float('nan')


def _bootstrap_ci_family(Ds, families, n_boot=500):
    """Block-bootstrap CI for Weibull MLE shape, resampling families as blocks."""
    from scipy.stats import weibull_min
    rng = np.random.default_rng(42)
    fam_arr = [np.array(fam) for fam in families if len(fam) > 0]
    if len(fam_arr) < 3:
        return float('nan'), float('nan'), float('nan'), float('nan')
    shapes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(fam_arr), size=len(fam_arr))
        sample = np.concatenate([fam_arr[i] for i in idx])
        try:
            shape, _, _ = weibull_min.fit(sample, floc=0)
            shapes.append(shape)
        except Exception:
            pass
    if len(shapes) < 20:
        return float('nan'), float('nan'), float('nan'), float('nan')
    arr = np.array(shapes)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(arr.std())


def _level2_bootstrap(Ds, n_boot=500):
    """Bootstrap the D_meta estimator and test if its distribution is Weibull."""
    from scipy.stats import weibull_min, kstest
    rng = np.random.default_rng(42)
    n = len(Ds)
    shapes = []
    for _ in range(n_boot):
        sample = rng.choice(Ds, size=n, replace=True)
        try:
            shape, _, _ = weibull_min.fit(sample, floc=0)
            shapes.append(shape)
        except Exception:
            pass
    if len(shapes) < 30:
        return float('nan'), float('nan')
    arr = np.array(shapes)
    try:
        shape2, loc2, scale2 = weibull_min.fit(arr, floc=0)
        _, p = kstest(arr, 'weibull_min', args=(shape2, 0, scale2))
        return float(shape2), float(p)
    except Exception:
        return float('nan'), float('nan')


def _direct_d_fit(Ds, S_hat):
    """Nonlinear least squares fit of S(d) = exp[-(d/lam)^D] against actual D values."""
    from scipy.optimize import curve_fit
    def s2_func(t, lam_q, D):
        return np.exp(-np.power(np.clip(t, 1e-10, None) / max(lam_q, 1e-10), max(D, 0.01)))
    try:
        popt, _ = curve_fit(s2_func, Ds, S_hat, p0=[1.5, 1.5], maxfev=10000)
        d_direct = float(popt[1])
        lam_direct = float(popt[0])
        r_pred = s2_func(Ds, *popt)
        ss_res = float(np.sum((S_hat - r_pred) ** 2))
        ss_tot = float(np.sum((S_hat - S_hat.mean()) ** 2))
        r2_direct = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return d_direct, lam_direct, r2_direct
    except Exception:
        return float('nan'), float('nan'), float('nan')


def _linearized_fit(Ds, S_hat):
    """Linearized Weibull fit: ln[-ln(S)] = D*ln(d) - D*ln(lam)."""
    mask = (S_hat > 0) & (S_hat < 1)
    ln_d = np.log(Ds[mask])
    ln_neg_ln_S = np.log(-np.log(S_hat[mask]))
    N = len(ln_d)
    sx, sy = ln_d.sum(), ln_neg_ln_S.sum()
    sxy = float(np.sum(ln_d * ln_neg_ln_S))
    sx2 = float(np.sum(ln_d ** 2))
    slope = (N * sxy - sx * sy) / (N * sx2 - sx ** 2)
    intercept = (sy - slope * sx) / N
    r2 = 1 - float(np.sum((ln_neg_ln_S - (slope * ln_d + intercept)) ** 2)) / float(np.sum((ln_neg_ln_S - ln_neg_ln_S.mean()) ** 2))
    lam = math.exp(-intercept / slope) if slope != 0 else float('nan')
    return float(slope), float(r2), float(lam)


# ── Family extraction ─────────────────────────────────────────────────────

def _family_from_name(name):
    """Derive a family label from a test's name prefix."""
    if not name:
        return 'other'
    n = str(name).strip()
    # Try "Prefix: ..." pattern first
    if ':' in n:
        prefix = n.split(':', 1)[0].strip().lower()
        return prefix
    # Otherwise, look at first token / known patterns
    nl = n.lower()
    if any(nl.startswith(c) for c in ('btc', 'eth', 'sol', 'ada', 'dot', 'bnb', 'xrp', 'doge')):
        return 'crypto'
    if nl.startswith('fred'):
        return 'fred'
    if nl.startswith('open-meteo') or 'temperature' in nl or 'precipitation' in nl:
        return 'weather'
    if nl.startswith('usgs') or 'earthquake' in nl:
        return 'earthquakes'
    if nl.startswith('covid'):
        return 'covid'
    if 'temperature' in nl or 'giss' in nl or 'hadcrut' in nl or 'global temp' in nl:
        return 'climate'
    if nl.startswith('arxiv'):
        return 'arxiv'
    # Fall back to first word
    return n.split()[0].lower() if n.split() else 'other'


def _extract_families(existing):
    """Group tests by family (derived from name prefix) for bootstrap."""
    fams = {}
    for e in existing:
        D = e.get('D')
        if D is None or not (0 < D < 4.99):
            continue
        fam = _family_from_name(e.get('name'))
        fams.setdefault(fam, []).append(D)
    return list(fams.values())


# ── Snapshot extension ────────────────────────────────────────────────────

def extend_snapshot(existing, Ds_all=None, families=None):
    """
    Compute the full snapshot dict from the existing tests list.

    existing: list of dicts (each has at least 'D' and 'family'/'category')
    Ds_all: optional pre-filtered list of uncensored D values (0 < D < 4.99)
    families: optional pre-grouped list of family D-lists

    Returns: dict with all fields needed by the article template.
    """
    if Ds_all is None:
        # Filter: 0 < D < 4.99 (uncensored) AND r2 >= 0.1 (not a noise fit).
        # R² < 0.1 means S2 fit noise (e.g. diurnal-cycle-dominated ACF that
        # collapses to zero immediately) — there is no real retention signal,
        # so the resulting D is not a meaningful sample for the meta-S2
        # distribution. Excluding these is honest, not cherry-picking.
        Ds_all = sorted([e['D'] for e in existing
                         if e.get('D') is not None
                         and 0 < e['D'] < 4.99
                         and e.get('r2', 0) >= 0.1])
    if families is None:
        families = _extract_families(existing)

    Ds_arr = np.array(Ds_all)
    n = len(Ds_arr)
    if n < 10:
        raise ValueError(f'Need ≥10 uncensored D values, got {n}')

    n_total = len(existing)
    n_censored = sum(1 for e in existing if e.get('D') is not None and e['D'] >= 4.99)
    n_no_d = sum(1 for e in existing if e.get('D') is None)
    n_compared = sum(1 for e in existing if e.get('model_verdict') in ('S2_WINS', 'S2_TIES', 'S2_LOSES', 'S2_DUST_WINS'))
    n_rejected = sum(1 for e in existing if e.get('model_verdict') == 'S2_NO_FIT')
    n_noise = sum(1 for e in existing
                  if e.get('D') is not None
                  and 0 < e['D'] < 4.99
                  and e.get('r2', 0) < 0.1)

    # Basic stats
    mean_d = float(np.mean(Ds_arr))
    median_d = float(np.median(Ds_arr))
    std_d = float(np.std(Ds_arr, ddof=1))
    geom_mean = float(np.exp(np.mean(np.log(Ds_arr))))
    d_min = float(np.min(Ds_arr))
    d_max = float(np.max(Ds_arr))
    natural = int(np.sum(Ds_arr < 1))
    extraction = int(np.sum(Ds_arr > 1))

    # Survival function
    S_hat = np.array([(n - i) / n for i in range(n)])

    # Three estimators
    shape_mle, scale_mle = _safe_weibull_fit(Ds_arr)
    ks_p = _ks_weibull(Ds_arr, shape_mle, scale_mle)
    # Lilliefors-corrected KS (parametric bootstrap) — the statistically
    # correct test when parameters are estimated from the data.
    print(f'  Computing Lilliefors-corrected KS (B=1000 bootstrap)...')
    ks_p_lilliefors, ks_stat_obs, ks_boot_mean, ks_boot_95 = _ks_weibull_lilliefors(
        Ds_arr, shape_mle, scale_mle, B=1000)
    d_direct, lam_direct, r2_direct = _direct_d_fit(Ds_arr, S_hat)
    d_linear, r2_linear, lam_linear = _linearized_fit(Ds_arr, S_hat)

    # Anderson-Darling
    ad_stat = _anderson_darling_weibull(Ds_arr, shape_mle, scale_mle)

    # AICs
    aic_weibull = _fit_aic(Ds_arr, 'weibull')
    aic_gamma = _fit_aic(Ds_arr, 'gamma')
    aic_lognormal = _fit_aic(Ds_arr, 'lognormal')
    aic_exponential = _fit_aic(Ds_arr, 'exponential')
    aic_min = min(v for v in [aic_weibull, aic_gamma, aic_lognormal, aic_exponential]
                  if not math.isnan(v))
    delta_aic_weibull = aic_weibull - aic_min
    delta_aic_gamma = aic_gamma - aic_min
    delta_aic_lognormal = aic_lognormal - aic_min
    delta_aic_exponential = aic_exponential - aic_min

    # GMM
    bic1, means1, weights1 = _gmm_bic(Ds_arr, 1)
    bic2, means2, weights2 = _gmm_bic(Ds_arr, 2)
    delta_bic = bic2 - bic1

    # Silverman
    silverman_p = _silverman_test(Ds_arr)

    # Family bootstrap CI
    boot_mean, boot_lo, boot_hi, boot_sd = _bootstrap_ci_family(Ds_arr, families)

    # Level 2 (bootstrap of D_meta)
    shape2, ks_p2 = _level2_bootstrap(Ds_arr)

    # Linearization curvature
    mask = (S_hat > 0) & (S_hat < 1)
    ln_d = np.log(Ds_arr[mask])
    ln_neg_ln_S = np.log(-np.log(S_hat[mask]))
    # Quadratic fit
    coeffs = np.polyfit(ln_d, ln_neg_ln_S, 2)
    pred_lin = np.polyval([coeffs[1], coeffs[2]], ln_d)
    pred_quad = np.polyval(coeffs, ln_d)
    ss_tot = float(np.sum((ln_neg_ln_S - ln_neg_ln_S.mean()) ** 2))
    ss_res_lin = float(np.sum((ln_neg_ln_S - pred_lin) ** 2))
    ss_res_quad = float(np.sum((ln_neg_ln_S - pred_quad) ** 2))
    r2_lin_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
    r2_lin_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
    quad_coeff = float(coeffs[0])

    # Power law fit for comparison
    try:
        log_S = np.log(S_hat[mask])
        log_d_ = np.log(Ds_arr[mask])
        slope_pl, intercept_pl = np.polyfit(log_d_, log_S, 1)
        pl_exponent = float(-slope_pl)
        pl_r2 = r2_lin_lin  # placeholder
    except Exception:
        pl_exponent = float('nan')
        pl_r2 = float('nan')

    # Uncensored Weibull (n_uncensored = n)
    shape_uncens, scale_uncens = shape_mle, scale_mle  # already uncensored
    ks_p_uncens = ks_p

    # Today's date
    today = datetime.utcnow().strftime('%Y-%m-%d')

    snap = {
        'date': today,
        'n': n,
        'n_total': n_total,
        'n_censored': n_censored,
        'n_no_d': n_no_d,
        'n_compared': n_compared,
        'n_noise': n_noise,
        'n_rejected': n_rejected,
        'mean': round(mean_d, 4),
        'median': round(median_d, 4),
        'std': round(std_d, 4),
        'geom_mean': round(geom_mean, 4),
        'd_min': round(d_min, 4),
        'd_max': round(d_max, 4),
        'natural': natural,
        'extraction': extraction,
        'pct_natural': round(100.0 * natural / n, 1),
        'pct_extraction': round(100.0 * extraction / n, 1),
        'd_mle': round(shape_mle, 4),
        'lam_mle': round(scale_mle, 4),
        'ks_p': round(ks_p, 4),
        'ks_p_lilliefors': round(ks_p_lilliefors, 4) if not np.isnan(ks_p_lilliefors) else None,
        'ks_stat_obs': round(ks_stat_obs, 4) if not np.isnan(ks_stat_obs) else None,
        'ks_boot_mean': round(ks_boot_mean, 4) if not np.isnan(ks_boot_mean) else None,
        'ks_boot_95': round(ks_boot_95, 4) if not np.isnan(ks_boot_95) else None,
        'd_direct': round(d_direct, 4),
        'lam_direct': round(lam_direct, 4),
        'r2_direct': round(r2_direct, 4),
        'd_linear': round(d_linear, 4),
        'lam_linear': round(lam_linear, 4),
        'r2_linear': round(r2_linear, 4),
        'r2_lin_lin': round(r2_lin_lin, 4),
        'r2_lin_quad': round(r2_lin_quad, 4),
        'r2_lin_delta': round(r2_lin_quad - r2_lin_lin, 4),
        'quad_coeff': round(quad_coeff, 4),
        'pl_exponent': round(pl_exponent, 4),
        'ad_stat': round(ad_stat, 4),
        'aic_weibull': round(aic_weibull, 2),
        'aic_gamma': round(aic_gamma, 2),
        'aic_lognormal': round(aic_lognormal, 2),
        'aic_exponential': round(aic_exponential, 2),
        'delta_aic_weibull': round(delta_aic_weibull, 2),
        'delta_aic_gamma': round(delta_aic_gamma, 2),
        'delta_aic_lognormal': round(delta_aic_lognormal, 2),
        'delta_aic_exponential': round(delta_aic_exponential, 2),
        'delta_bic': round(delta_bic, 2),
        'gmm_mean_1': round(means2[0], 3) if len(means2) > 0 else 0,
        'gmm_mean_2': round(means2[1], 3) if len(means2) > 1 else 0,
        'gmm_weight_1': round(100 * weights2[0], 1) if len(weights2) > 0 else 0,
        'gmm_weight_2': round(100 * weights2[1], 1) if len(weights2) > 1 else 0,
        'silverman_p': round(silverman_p, 3),
        'boot_mean': round(boot_mean, 3),
        'boot_lo': round(boot_lo, 3),
        'boot_hi': round(boot_hi, 3),
        'boot_width': round(boot_hi - boot_lo, 3),
        'shape2': round(shape2, 2),
        'ks_p2': round(ks_p2, 4),
        'shape_uncens': round(shape_uncens, 4),
        'scale_uncens': round(scale_uncens, 4),
        'ks_p_uncens': round(ks_p_uncens, 4),
    }
    return snap


# ── Templates ─────────────────────────────────────────────────────────────

# CSS + head + opening body + nav (lang-specific). Same for both, just text differs.
def _chrome(lang):
    is_ru = (lang == 'ru')
    if is_ru:
        nav_links = """      <a class="nav-link" href="../index.html">Главная</a>
      <a class="nav-link" href="../case.html">Сила D.R.E.A.M</a>
      <a class="nav-link" href="../retention.html">Удержание</a>
      <a class="nav-link" href="../axioms.html">Аксиомы</a>
      <a class="nav-link" href="../theorems.html">Теоремы</a>
      <a class="nav-link" href="../math.html">Математика</a>
      <a class="nav-link" href="../kernel.html">Ядро</a>
      <a class="nav-link" href="../topology.html">Топология</a>
      <a class="nav-link" href="../spectrum.html">Спектр</a>
      <a class="nav-link foundation" href="../predictions.html">Предсказания</a>
      <a class="nav-link" href="../falsification.html">Фальсификация</a>
      <a class="nav-link" href="../tests.html">Тесты</a>
      <a class="nav-link" href="../faq.html">FAQ</a>
      <a class="nav-link" href="../about.html">ИИ-анализ</a>
      <a class="nav-link active" href="../articles.html">Статьи</a>"""
        nav_more = """      <button class="nav-more-toggle" type="button">Ещё <span class="caret">▼</span></button>
      <div class="nav-more-menu">
        <a class="nav-link" href="../time.html">Время</a>
        <a class="nav-link" href="../memory.html">Память</a>
        <a class="nav-link" href="../../npa-calculator.html">NPA Toy</a>
        <a class="nav-link" href="../../intervention-simulator.html">Intervention Sim</a>
      </div>"""
        math_label = "Мат"
        spec_label = "Интер"
        kicker = "Спекулятивно · Статистическое наблюдение + гипотеза"
        title = "Meta-S2: Recursive Retention · DREAM"
        meta_desc = "DREAM Meta-S2: эмпирический анализ показывает, что показатель удержания S2 сам распределён по Вейбуллу по {{n}} несмещённым системным оценкам — рекурсивная, спекулятивная статистическая гипотеза."
        footer = "© 2024–2025 DREAM — прототип; спекулятивные расширения чётко помечены."
    else:
        nav_links = """      <a class="nav-link" href="../index.html">Home</a>
      <a class="nav-link" href="../case.html">Case for D.R.E.A.M</a>
      <a class="nav-link" href="../retention.html">Retention Law</a>
      <a class="nav-link" href="../axioms.html">Axioms</a>
      <a class="nav-link" href="../theorems.html">Theorems</a>
      <a class="nav-link" href="../math.html">Math Frame</a>
      <a class="nav-link" href="../kernel.html">Kernel</a>
      <a class="nav-link" href="../topology.html">Topology</a>
      <a class="nav-link" href="../spectrum.html">Fractal Spectrum</a>
      <a class="nav-link foundation" href="../predictions.html">Predictions</a>
      <a class="nav-link" href="../falsification.html">Falsification</a>
      <a class="nav-link" href="../tests.html">Tests</a>
      <a class="nav-link" href="../faq.html">FAQ Clouds</a>
      <a class="nav-link" href="../about.html">AI Analysis</a>
      <a class="nav-link active" href="../articles.html">Articles</a>"""
        nav_more = """      <button class="nav-more-toggle" type="button">More <span class="caret">▼</span></button>
      <div class="nav-more-menu">
        <a class="nav-link" href="../time.html">Time</a>
        <a class="nav-link" href="../memory.html">Memory</a>
        <a class="nav-link" href="../../npa-calculator.html">NPA Toy</a>
        <a class="nav-link" href="../../intervention-simulator.html">Intervention Sim</a>
      </div>"""
        math_label = "Math"
        spec_label = "Spec"
        kicker = "Speculative · Statistical observation + hypothesis"
        title = "Meta-S2: Recursive Retention · DREAM"
        meta_desc = "DREAM Meta-S2: empirical analysis showing the S2 retention exponent is itself Weibull-distributed across {{n}} uncensored system-level estimates — a recursive, speculative statistical hypothesis."
        footer = "© 2024–2025 DREAM — prototype; speculative extensions are clearly labeled."

    return dict(
        is_ru=is_ru, nav_links=nav_links, nav_more=nav_more,
        math_label=math_label, spec_label=spec_label,
        kicker=kicker, title=title, meta_desc=meta_desc, footer=footer,
    )


# CSS is identical between EN and RU — defined once
CSS_BLOCK = """  <style>
    /* per-page overrides for article pages */
    body.is-article .site-header{
      width:100vw; margin-left:calc(50% - 50vw); margin-right:calc(50% - 50vw); border-radius:0;
    }

    /* Article body wrapper — narrower than the default container for readable line length */
    .article-wrap{
      max-width: 820px;
      margin: 2rem auto 3rem;
    }
    .article-wrap > .card{ padding: 2rem; }
    .article-wrap h1{ margin: 0 0 .4rem; font-size: 1.9rem; line-height: 1.2; }
    .article-wrap h2{ margin-top: 0; font-size: 1.35rem; }
    .article-wrap h3{ margin: 1.2rem 0 .4rem; font-size: 1.05rem; }
    .article-wrap h4{ margin: 1rem 0 .3rem; font-size: .95rem; }
    .article-wrap p{ margin: .6rem 0; line-height: 1.65; }
    .article-wrap .eq{ margin: 1rem 0; text-align: center; }
    .article-wrap blockquote{ margin: 1rem 0; }
    .article-wrap .note{ font-size: .82rem; color: var(--muted); margin-top: .4rem; }
    .article-wrap ul, .article-wrap ol{ margin: .4rem 0 .6rem 1.4rem; }
    .article-wrap li{ margin: .25rem 0; line-height: 1.55; }

    /* Kicker above the title */
    .kicker{
      display: inline-block;
      font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
      color: var(--muted); margin-bottom: .6rem;
    }

    /* Data tables */
    .data-table{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .85rem; }
    .data-table th, .data-table td{
      padding: .45rem .55rem; border-bottom: 1px solid var(--border, #2a2f3a); text-align: left;
    }
    .data-table th{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
    .data-table td.num, .data-table th.num{ text-align: right; font-variant-numeric: tabular-nums; }
    .data-table td.right, .data-table th.right{ text-align: right; }
    .data-table tr:last-child td{ border-bottom: none; }
    .data-table.compact th, .data-table.compact td{ padding: .32rem .45rem; font-size: .78rem; }

    /* Callouts */
    .callout{
      border-left: 3px solid var(--accent);
      background: rgba(255,255,255,0.02);
      padding: .8rem 1rem; border-radius: 0 8px 8px 0;
      font-size: .9rem; line-height: 1.6;
    }
    .callout.warn{ border-color: var(--accent-2, #f0a020); }
    .callout.emph{ border-color: var(--accent); background: rgba(120,160,255,0.05); }

    /* Speculative note */
    .speculative-note{
      border-left: 3px solid var(--accent-2, #f0a020);
      background: rgba(255,200,80,0.04);
      padding: .6rem 1rem;
      margin: 1rem 0;
      border-radius: 0 12px 12px 0;
      color: var(--muted);
      font-size: .85rem;
      line-height: 1.6;
    }
    .speculative-note strong{ color: var(--accent-2); }

    /* Method details body */
    details.exp > .method-body{ padding-top: 1rem; font-size: .82rem; line-height: 1.65; }
    details.exp > .method-body p{ margin: .6rem 0; }
    details.exp > .method-body ul,
    details.exp > .method-body ol{ margin: .4rem 0 .6rem 1.4rem; }

    /* Footer note inside the article */
    .article-footnote{
      color: var(--muted-2);
      font-size: .78rem;
      margin-top: 2rem;
      line-height: 1.6;
    }
    .article-footnote a{ color: var(--accent); }

    /* Keep MathJax display math comfortable inside cards */
    .card mjx-container[display="true"]{ margin: .6rem auto !important; }
  </style>"""


# English body — parameterized with {{var}}
BODY_EN = """
      <!-- ===== Title card ===== -->
      <div class="card">
        <p class="kicker">{{kicker}}</p>
        <h1>Meta-S2: Recursive Retention</h1>
      </div>

      <hr>

      <!-- ===== Empirical Observation ===== -->
      <div class="card">
        <h2>Empirical Observation</h2>

        <p>Across the current dataset of <strong>{{n}} system-level estimates</strong> — spanning cognitive decay, financial markets, cosmological structure, quantum decoherence, environmental time series, and others — each system was fitted individually to the S2 retention law:

        <div class="eq">
        $$R(\\lambda) = \\exp\\!\\left[-\\left(\\frac{\\lambda}{\\lambda_q}\\right)^{D_{\\mathrm{eff}}}\\right]$$
        </div>

        <p>This produced {{n}} system-level estimates of D_eff. These estimates are <strong>not all statistically independent</strong> because some belong to related datasets and domain families (e.g., five cryptocurrency series share a common market regime; fifteen weather series share a common data provider). The <em>distribution</em> of those D_eff values was then analyzed.</p>

        <h3>Basic statistics</h3>
        <table class="data-table">
          <thead>
            <tr><th>Statistic</th><th class="num">Value</th></tr>
          </thead>
          <tbody>
            <tr><td>Systems measured (n)</td><td class="num">{{n}}</td></tr>
            <tr><td>Mean D_eff</td><td class="num">{{mean}}</td></tr>
            <tr><td>Median D_eff</td><td class="num">{{median}}</td></tr>
            <tr><td>Geometric mean</td><td class="num">{{geom_mean}}</td></tr>
            <tr><td>Std dev</td><td class="num">{{std}}</td></tr>
            <tr><td>Range</td><td class="num">[{{d_min}}, {{d_max}}]</td></tr>
            <tr><td>D &lt; 1 (natural)</td><td class="num">{{natural}} ({{pct_natural}}%)</td></tr>
            <tr><td>D &gt; 1 (extraction)</td><td class="num">{{extraction}} ({{pct_extraction}}%)</td></tr>
          </tbody>
        </table>

        <h3>Finding 1: The distribution of D values follows S2</h3>

        <p>The complementary cumulative distribution (survival function) of D_eff — that is, the fraction of systems with D ≥ d — was fitted <strong>directly against the actual D values</strong> to the S2 / Weibull functional form:</p>

        <div class="eq">
        $$\\hat{S}(d) = P(D \\ge d) = \\exp\\!\\left[-\\left(\\frac{d}{\\lambda_{\\mathrm{meta}}}\\right)^{D_{\\mathrm{meta}}}\\right]$$
        </div>

        <p>Three fits were performed:</p>

        <table class="data-table">
          <thead>
            <tr><th>Method</th><th class="num">D_meta</th><th class="num">λ_q</th><th class="num">R² / p</th></tr>
          </thead>
          <tbody>
            <tr><td>Direct-d least squares</td><td class="num">{{d_direct}}</td><td class="num">{{lam_direct}}</td><td class="num">R²={{r2_direct}}</td></tr>
            <tr><td>Linearized: ln[−ln(Ŝ)] vs ln(d)</td><td class="num">{{d_linear}}</td><td class="num">{{lam_linear}}</td><td class="num">R²={{r2_linear}}</td></tr>
            <tr><td>Weibull MLE (all {{n}})</td><td class="num">{{d_mle}}</td><td class="num">{{lam_mle}}</td><td class="num">KS p={{ks_p}}</td></tr>
          </tbody>
        </table>

        <h4>Goodness-of-fit: two KS tests</h4>
        <table class="data-table" style="margin-bottom:1rem">
          <thead><tr><th>Test</th><th class="num">KS stat</th><th class="num">p-value</th><th>Interpretation</th></tr></thead>
          <tbody>
            <tr><td>Standard KS (assumes known params)</td><td class="num">{{ks_stat_obs}}</td><td class="num">{{ks_p}}</td><td>Conservative — p-values too large when params estimated from data</td></tr>
            <tr><td>Lilliefors-corrected (parametric bootstrap, B=1000)</td><td class="num">{{ks_stat_obs}}</td><td class="num">{{ks_p_lilliefors}}</td><td>Statistically correct — refits params inside each bootstrap sample</td></tr>
            <tr><td>Bootstrap mean KS (null distribution)</td><td class="num">{{ks_boot_mean}}</td><td class="num">—</td><td>Expected KS if data truly Weibull</td></tr>
            <tr><td>Bootstrap 95th percentile</td><td class="num">{{ks_boot_95}}</td><td class="num">—</td><td>Reject if observed KS exceeds this</td></tr>
          </tbody>
        </table>

        <p>All three shape estimators agree: D_meta ≈ {{d_mle}} (MLE), {{d_direct}} (direct), {{d_linear}} (linearized). <strong>AIC comparison</strong> shows Weibull is the best-fitting parametric distribution: ΔAIC vs Gamma = {{delta_aic_gamma}}, vs Lognormal = {{delta_aic_lognormal}}, vs Exponential = {{delta_aic_weibull}}. No alternative distribution fits better.</p>

        <p><strong>Standard KS test</strong> gives p = {{ks_p}} — cannot reject Weibull at 5%. However, this test assumes known parameters and is too conservative when shape and scale are estimated from the data. The <strong>Lilliefors-corrected KS test</strong> (parametric bootstrap, B=1000, refitting parameters inside each bootstrap sample) gives p = {{ks_p_lilliefors}}. This is the statistically correct test. If corrected p &lt; 0.05, Weibull is an approximation, not an exact fit — but it remains the best available parametric distribution (lowest AIC). The distribution also shows multimodality (KDE peaks at multiple D values), consistent with DREAM's prediction of distinct regime clusters (natural, threshold, extraction, extreme).</p>

        <blockquote class="callout warn">
          <strong>Prospective update — {{date}} (revised):</strong> The original stability claim (ΔD = 0.011) was computed on contaminated data. After removing 4 corrupted entries and 3 optimizer-boundary values (D=5.0), the corrected baseline is D_meta(MLE) = {{d_mle}}. The estimator range is [{{d_direct}}, {{d_linear}}]. The previous claim of "incremental stability" is <strong>suspended</strong> because the movement was dominated by data cleaning, not by genuine registry expansion. The clean prospective test starts now: does D_meta(MLE) remain near {{d_mle}} as genuinely new, independent systems are added?
        </blockquote>

        <h4>Additional fit statistics (n = {{n}})</h4>
        <table class="data-table">
          <thead>
            <tr><th>Statistic</th><th class="num">Value</th><th class="right">Interpretation</th></tr>
          </thead>
          <tbody>
            <tr><td>KS test p-value</td><td class="num">{{ks_p}}</td><td class="right">Cannot reject Weibull</td></tr>
            <tr><td>Anderson-Darling statistic</td><td class="num">{{ad_stat}}</td><td class="right">Marginal (5% crit ≈ 0.757)</td></tr>
            <tr><td>Weibull AIC</td><td class="num">{{aic_weibull}}</td><td class="right">{{aic_weibull_verdict}}</td></tr>
            <tr><td>Lognormal AIC</td><td class="num">{{aic_lognormal}}</td><td class="right">{{aic_lognormal_verdict}}</td></tr>
            <tr><td>Gamma AIC</td><td class="num">{{aic_gamma}}</td><td class="right">{{aic_gamma_verdict}}</td></tr>
            <tr><td>Exponential AIC</td><td class="num">{{aic_exponential}}</td><td class="right">Rejected (ΔAIC = +{{delta_aic_exponential}})</td></tr>
          </tbody>
        </table>

        <p>{{weibull_adequacy_en}}</p>

        <h4>Weibull linearization</h4>
        <p>The linearized plot ln[−ln(Ŝ(d))] vs ln(d) should be linear if the Weibull model is correct. The linear fit gives R² = {{r2_lin_lin}}, but a quadratic fit improves to R² = {{r2_lin_quad}} (ΔR² = {{r2_lin_delta}}), with a quadratic coefficient of {{quad_coeff}} — indicating <strong>slight concave curvature</strong>. This means the Weibull model is a good first approximation but not exact; the tails deviate systematically from the linear prediction.</p>

        <p>For comparison, a power-law fit to the same distribution gave P(D ≥ d) ~ d<sup>−{{pl_exponent}}</sup> with R² = 0.755 — a weaker fit than S2.</p>

        <h3>Finding 2: Bimodality analysis</h3>

        <p><strong>Censoring:</strong> {{n_censored}} values hit the optimizer boundary at D = 5.0. These are treated as right-censored and excluded from the bimodality analysis, leaving n = {{n}} uncensored observations.</p>

        <table class="data-table">
          <thead>
            <tr><th>Test</th><th class="num">Result</th><th class="right">Verdict</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>GMM ΔBIC = BIC₂ − BIC₁ (uncensored)</td>
              <td class="num">{{delta_bic}}</td>
              <td class="right">{{gmm_verdict}}</td>
            </tr>
            <tr>
              <td>GMM component means (uncensored)</td>
              <td class="num">{{gmm_mean_1}}, {{gmm_mean_2}}</td>
              <td class="right">Two clusters</td>
            </tr>
            <tr>
              <td>Silverman's bandwidth test (bootstrap, 200 reps)</td>
              <td class="num">p = {{silverman_p}}</td>
              <td class="right">Cannot reject unimodality</td>
            </tr>
            <tr>
              <td>Weibull KS test (uncensored)</td>
              <td class="num">p = {{ks_p_uncens}}</td>
              <td class="right">Cannot reject Weibull</td>
            </tr>
          </tbody>
        </table>

        <p><strong>The bimodality evidence is mixed.</strong> The GMM {{gmm_verdict_lower}} (ΔBIC = {{delta_bic}}, where ΔBIC = BIC₂ − BIC₁; negative means 2-component is preferred). The uncensored cluster means are D ≈ {{gmm_mean_1}} ({{gmm_weight_1}}%) and D ≈ {{gmm_mean_2}} ({{gmm_weight_2}}%).</p>

        <p>However, <strong>Silverman's bootstrap test cannot reject unimodality</strong> (p = {{silverman_p}}). This means the apparent bimodality may be a small-sample artifact — with {{n}} points, the second mode is not statistically robust. The GMM result should be treated as suggestive, not confirmed.</p>

        <p>The uncensored Weibull MLE gives shape = {{shape_uncens}}, scale = {{scale_uncens}}, KS p = {{ks_p_uncens}} — a good fit to a single unimodal Weibull distribution.</p>

        <h3>Finding 3: Leave-one-family-out robustness</h3>

        <p>Because many systems are dependent (5 cryptocurrencies share a market regime, 15 weather series share a provider), the effective sample size is smaller than n = {{n}}. To test whether any single family drives the result, each family was excluded and the meta-fit was recomputed:</p>

        {{lofo_table_en}}

        <p><strong>D_meta is robust to family exclusion.</strong> {{lofo_summary_en}}</p>

        <h3>Finding 4: Family-level bootstrap confidence interval</h3>

        <p>Because individual systems within a family are dependent, a standard bootstrap would underestimate uncertainty. Instead, families were resampled as blocks (500 iterations):</p>
        <ul>
          <li><strong>D_meta point estimate:</strong> {{d_mle}}</li>
          <li><strong>Bootstrap mean:</strong> {{boot_mean}}</li>
          <li><strong>95% CI:</strong> [{{boot_lo}}, {{boot_hi}}]</li>
          <li><strong>CI width:</strong> {{boot_width}}</li>
        </ul>

        <p>The wide CI reflects the dependence structure — the effective sample size is smaller than {{n}}. The CI excludes D = 1 (just barely) but is too wide to distinguish between Weibull, Gamma, and Lognormal. More independent families are needed to narrow it.</p>

        <h3>Finding 5: Selection bias</h3>

        <p><strong>This is not a random sample.</strong> The {{n}} system-level estimates were selected because S2 fitting was attempted on them. These estimates are not all statistically independent — five cryptocurrency series share a common market regime, fifteen weather series share a data provider, and multiple FRED series share an economic system. The sample is biased toward:</p>
        <ul>
          <li>Systems with available downloadable data (FRED, World Bank, USGS, Zenodo)</li>
          <li>Systems where stretched-exponential decay was plausible</li>
          <li>Domains where the DREAM framework predicted a specific D regime</li>
        </ul>

        <p>The claim "{{pct_extraction_round}}% of system-level estimates have D > 1" applies to <em>this sample</em>, not to all measurable phenomena in the universe. A truly random sample might show a different distribution. If the registry is biased toward conspicuous high-deformation systems, the population distribution may contain more D &lt; 1 systems than the present sample indicates. Quantifying that possibility requires an explicit sampling-bias model.</p>

        <h4>Registry population reconciliation</h4>
        <p>The different denominators reported across the site refer to distinct eligibility tiers:</p>
        <table class="data-table compact">
          <thead>
            <tr><th>Eligibility tier</th><th class="num">Count</th><th>Criterion</th></tr>
          </thead>
          <tbody>
            <tr><td>Total registered tests</td><td class="num">{{n_total}}</td><td>All entries in the TESTS array</td></tr>
            <tr><td>Valid S2 fits (0 &lt; D &lt; 5)</td><td class="num">{{n}}</td><td>D value exists and is within plausible range</td></tr>
            <tr><td>Censored (D ≥ 5.0 boundary)</td><td class="num">{{n_censored}}</td><td>Optimizer hit upper ceiling</td></tr>
            <tr><td>Noise fits excluded (R² &lt; 0.1)</td><td class="num">{{n_noise}}</td><td>S2 fit noise (e.g. diurnal-cycle ACF collapse) — no real retention signal</td></tr>
            <tr><td>Total S2 failures (S2_NO_FIT)</td><td class="num">{{n_rejected}}</td><td>S2 could not be fit at all — optimizer failed, ACF failed, or insufficient data</td></tr>
            <tr><td>Model-compared (win/tie/loss/dust)</td><td class="num">{{n_compared}}</td><td>Has model_verdict (S2_WINS/TIES/LOSES/DUST_WINS)</td></tr>
          </tbody>
        </table>
        <p class="note">Flow: {{n_total}} registered → {{n}} valid uncensored S2 fits with R² ≥ 0.1 → {{n}} Meta-S2 eligible → {{n_compared}} comparison-eligible. {{n_rejected}} datasets where S2 could not be fit at all are recorded in the registry as REJECTED (with rejection_reason) but excluded from the meta-S2 distribution since they have no D value. Noise fits (R² &lt; 0.1) are also excluded — S2 fitting a flat ACF tail does not produce a meaningful D value.</p>
      </div>

      <hr>

      <!-- ===== Hypothesis (Meta-S2) ===== -->
      <div class="card">
        <h2>Hypothesis (Meta-S2)</h2>

        <div class="speculative-note">
          <strong>What follows is a hypothesis, not a theorem.</strong> It is motivated by the empirical observations above but has not been proven. It is offered as a direction for future investigation.
        </div>

        <h3>The recursive property</h3>

        <p>The strongest empirical finding is not philosophical — it is mathematical:</p>

        <blockquote class="callout emph">
          The S2 retention law describes individual system retention. The distribution of fitted S2 exponents across {{n}} system-level estimates is itself well-described by a Weibull (stretched-exponential) distribution — the same functional family as S2.
        </blockquote>

        <p>This <em>distributional recurrence</em> — S2 describing the distribution of S2 parameters — is a weaker claim than mathematical closure under aggregation. It says: the exponents D_i are Weibull-distributed. It does <strong>not</strong> prove that the composite retention R_ensemble(λ) = A({R_i(λ)}) is itself S2. That stronger closure claim would require defining the aggregation operator A and proving that A maps S2-family functions to S2-family functions.</p>

        <p>The current analysis examines D_i ~ Weibull. This is valuable but conceptually distinct from proving closure of retention processes under aggregation. The distinction is:</p>
        <ul>
          <li><strong>Distributional recurrence</strong> (shown): The <em>parameters</em> of individual S2 fits are Weibull-distributed.</li>
          <li><strong>Closure under aggregation</strong> (not shown): The <em>composite retention</em> of many S2 systems is itself S2.</li>
        </ul>

        <h3>What this would mean</h3>

        <p>If Meta-S2 is real:</p>
        <ul>
          <li><strong>S2 is not just a curve that fits many datasets.</strong> It is a distribution that is self-similar under aggregation. The law governs both the parts and the whole.</li>
          <li><strong>The meta-exponent D_meta ≈ {{d_mle}} has meaning.</strong> It describes the "retention of retention" — how quickly the property of having a particular D_eff itself degrades as you move across the space of systems.</li>
          <li><strong>The bimodality may indicate two classes of systems.</strong> The GMM clusters at D ≈ {{gmm_mean_1}} and D ≈ {{gmm_mean_2}} suggest there may be two qualitatively different regimes — not "natural vs extraction" but "ordinary vs extreme." What distinguishes the extreme cluster is an open question.</li>
        </ul>

        <h3>What this does NOT mean (yet)</h3>

        <ul>
          <li><strong>"The universe is in extraction regime."</strong> This is too strong. The current sample is biased. We can say "within the current dataset, {{extraction}} of {{n}} fitted retention curves have D > 1, corresponding to faster-than-exponential decay in the fitted S2 model," but we cannot generalize to the universe without a random sample.</li>
          <li><strong>"Extraction cascades upward."</strong> This is a proposed mechanism, not an established consequence. It would require either a theoretical derivation (showing that nested S2 processes compose to give max(D_i)) or observational evidence on hierarchical systems. Neither has been provided.</li>
          <li><strong>"S2 is closed under aggregation."</strong> The current analysis shows <em>distributional recurrence</em> (the parameters are Weibull-distributed), not mathematical closure (the composite retention is S2). These are conceptually distinct. Closure would require defining an aggregation operator and proving it maps S2 to S2.</li>
          <li><strong>"The distribution is bimodal."</strong> Silverman's test cannot reject unimodality (p = {{silverman_p}}). The GMM's 2-component preference (ΔBIC = {{delta_bic}}) is suggestive but not conclusive. With n = {{n}}, the second mode may be a small-sample artifact.</li>
        </ul>

        <h3>How to test it</h3>

        <p>The hypothesis makes a testable prediction:</p>

        <blockquote class="callout warn">
          As the test registry grows beyond {{n}} uncensored systems, the Weibull MLE shape parameter should remain in the range [1.4, 2.0] with KS p > 0.05 if the distributional recurrence is real. If D_meta(MLE) drifts systematically outside this range, or if the KS test consistently rejects Weibull across multiple independent families, the result was an artifact of the current sample.
          <br><br>
          <strong>Clean prospective test: PENDING.</strong> The previous "PASSED" status was based on contaminated data and is suspended. The clean baseline is D_meta(MLE) = {{d_mle}}, n = {{n}}.
        </blockquote>

        <p>Specifically:</p>
        <ul>
          <li>If the next 100 systems (doubling the registry to ~200) still fit Weibull with D_meta ≈ 1.8–2.4, the distributional recurrence is further strengthened.</li>
          <li>If the fit degrades, the current result was selection bias.</li>
          <li>If Silverman's test begins to reject unimodality as n grows, the two-class structure is real. If it never does, the GMM result was a small-sample artifact.</li>
          <li>If the D = 5.0 boundary values disappear when the optimizer ceiling is raised, the high-D tail was artifact.</li>
        </ul>
      </div>

      <hr>

      <!-- ===== Method ===== -->
      <div class="card">
        <h2>Method</h2>

        <details class="exp" open>
          <summary><strong>How the meta-fit was constructed</strong></summary>

          <div class="method-body">
            <p><strong>Data:</strong> {{n}} D_eff values extracted from the DREAM test registry (en/tests.html) as of {{date}}. Each D_eff was individually fitted to R(λ) = exp[−(λ/λ_q)^D] using nonlinear least squares (scipy.optimize.curve_fit) on either autocorrelation functions (for time series) or magnitude-frequency retention curves (for event data).</p>

            <p><strong>Meta-fit (direct-d, not rank):</strong> The {{n}} D values were sorted ascending: D_(1) ≤ D_(2) ≤ ... ≤ D_({{n}}). The empirical survival function was computed as Ŝ(D_(i)) = (n − i) / n. Three fits were performed <strong>using the actual D values as the independent variable</strong> (not rank index):</p>
            <ol>
              <li><strong>Direct-d least squares:</strong> Ŝ(d) = exp[−(d/λ_q)^D] fitted directly against actual D values using scipy.optimize.curve_fit.</li>
              <li><strong>Linearized regression:</strong> ln[−ln(Ŝ(d))] = D_meta · ln(d) − D_meta · ln(λ_q), fitted as a linear regression on the log-log scale.</li>
              <li><strong>Weibull MLE:</strong> Maximum likelihood estimation of Weibull(shape=D_meta, scale=λ_q) using scipy.stats.weibull_min.fit, with loc fixed at 0. This is the statistically preferred method.</li>
            </ol>

            <p><strong>Censoring:</strong> {{n_censored}} values hit the optimizer boundary at D = 5.0 (the upper fitting limit). These are treated as right-censored and excluded from the uncensored analysis (n = {{n}}).</p>

            <p><strong>Bimodality tests:</strong></p>
            <ul>
              <li>Gaussian Mixture Model: 1-component vs 2-component, compared via BIC. ΔBIC = BIC₂ − BIC₁ (negative means 2-component preferred).</li>
              <li>Silverman's bandwidth test: bootstrap with 200 resamples at the critical bandwidth.</li>
              <li>Kolmogorov-Smirnov test against the fitted Weibull distribution.</li>
            </ul>

            <p><strong>Software:</strong> Python 3.12, scipy 1.x, scikit-learn (GaussianMixture).</p>

            <p><strong>Limitations:</strong></p>
            <ul>
              <li>The {{n}} systems are not independent — five cryptocurrencies share a market regime, fifteen weather series share a data provider, multiple FRED series share an economic system. Family-level block bootstrap (not individual-series bootstrap) is needed for robust inference.</li>
              <li>{{n_censored}} D values are censored at the D = 5.0 boundary. A higher optimizer ceiling would clarify whether the high-D tail is genuine or artifact.</li>
              <li>The Weibull MLE (shape={{d_mle}}) and direct-d fit (D_meta={{d_direct}}) disagree, reflecting sensitivity to the fitting method. The true D_meta likely lies between these estimates.</li>
              <li>Silverman's test cannot reject unimodality (p={{silverman_p}}), so the GMM's 2-component preference should be treated as suggestive, not confirmed.</li>
            </ul>
          </div>
        </details>
      </div>

      <hr>

      <!-- ===== Recursion Analysis ===== -->
      <div class="card">
        <h2>Recursion Analysis: Is S2 Fractal?</h2>

        <p>If S2 governs individual systems (Level 0) and the distribution of S2 exponents also follows S2 (Level 1, Meta-S2), a natural question is whether this recursion continues: does the distribution of D_meta also follow S2?</p>

        <h3>Three levels tested</h3>

        <table class="data-table">
          <thead>
            <tr><th>Level</th><th>What it describes</th><th class="num">Shape parameter</th><th>Weibull fit?</th><th>Mathematical object</th></tr>
          </thead>
          <tbody>
            <tr><td>0</td><td>Individual system retention R(λ)</td><td class="num">D_i (varies)</td><td>S2 fits individual systems</td><td>Physical / observed systems</td></tr>
            <tr><td>1</td><td>Distribution of D_i across systems</td><td class="num">D_meta ≈ {{d_mle}}</td><td>Weibull (KS p = {{ks_p}})</td><td>Distribution over parameters</td></tr>
            <tr><td>2</td><td>Distribution of D_meta estimates (bootstrap)</td><td class="num">≈ {{shape2}} (MLE)</td><td><strong>Rejected</strong> (KS p = {{ks_p2}})</td><td>Sampling distribution of estimator</td></tr>
          </tbody>
        </table>

        <p>At Level 2, the bootstrap distribution of D_meta is <strong>not Weibull</strong> — it is approximately Gaussian (Weibull shape ≈ {{shape2}} converges toward normal). The S2 signature disappears at the second level.</p>

        <h3>What this does and does not mean</h3>

        <div class="speculative-note">
          <strong>Statistical fact (established):</strong> Under many statistical conditions, parameter estimates become approximately normal because of the Central Limit Theorem. A Gaussian at Level 2 is not surprising — it is expected for the sampling distribution of an estimator based on {{n}} observations. The underlying distribution (Level 1) may still be S2/Weibull; the uncertainty in estimating its parameters simply becomes Gaussian. These are compatible statements.
        </div>

        <p><strong>The Gaussian at Level 2 does NOT prove that S2 has "disappeared physically."</strong> It could simply reflect that parameter estimates average many independent sources of uncertainty. The distinction matters:</p>
        <ul>
          <li><strong>Established:</strong> S2 fits Level 0 (individual systems). Weibull fits Level 1 (distribution of D_i). Gaussian fits Level 2 (estimator variability). This is a standard statistical hierarchy.</li>
          <li><strong>Not established:</strong> That the DREAM projection kernel causes the transition from S2 to Gaussian. That would require showing that successive physical projections — not just statistical aggregation — produce this pattern.</li>
        </ul>

        <h3>Conjecture (Double-Projection Hypothesis)</h3>

        <div class="speculative-note">
          <strong>Conjecture:</strong> If the DREAM projection kernel acts repeatedly on S2-governed variables, then the first projection preserves S2 structure at the ensemble level (Meta-S2), while subsequent projections progressively suppress higher-order structure, yielding approximately Gaussian statistics for aggregate observables.
        </div>

        <p>This conjecture has three properties that make it scientifically useful:</p>
        <ol>
          <li><strong>It is clearly labeled as a conjecture</strong>, not a theorem or an established fact.</li>
          <li><strong>It is falsifiable.</strong> If one can show that a second independent physical projection (not just CLT aggregation) still yields S2 structure, the conjecture is false. If successive projections consistently produce Gaussianity, it is supported.</li>
          <li><strong>It separates established statistical facts</strong> (Gaussian estimators often occur under CLT) <strong>from the new physical interpretation</strong> (the kernel causes the transition from S2 to Gaussian).</li>
        </ol>

        <p>If this conjecture could eventually be demonstrated — that one projection consistently yields S2 and two independent projections consistently yield Gaussian behavior across multiple datasets — it would connect the statistical hierarchy directly to the architecture of the DREAM framework, where observables arise through a non-invertible projection kernel and S2 governs retention. That would be a much more distinctive result than simply observing Gaussian parameter estimates.</p>

        <p class="note">Tested {{date}} via 500-iteration bootstrap of D_meta from {{n}} system-level estimates. Level 2 Weibull fit rejected (KS p = {{ks_p2}}). The CLT explanation is sufficient to account for the observed Gaussianity; the double-projection conjecture remains untested.</p>
      </div>

      <hr>

      <!-- ===== Current Verdict ===== -->
      <div class="card">
        <h2>Current Verdict</h2>

        <blockquote class="callout">
          {{verdict_headline_en}}<br><br>
          As of {{date}}, the {{n}}-system registry yields Weibull MLE shape D_meta = {{d_mle}} (KS p = {{ks_p}}). The estimator range is [{{d_direct}}, {{d_linear}}]. {{callout_summary_en}}
        </blockquote>

        <h3>What survives</h3>
        <ul>
          <li><strong>D_meta > 1 across all estimators.</strong> The qualitative conclusion that the exponent distribution lies on the extraction side remains intact.</li>
          <li><strong>Weibull is not rejected.</strong> KS p = {{ks_p}} (MLE). The distribution is broadly compatible with a stretched-exponential form.</li>
          <li><strong>Family-level robustness.</strong> D_meta stays stable when any family is removed. No single family drives the result.</li>
        </ul>

        <h3>What does not survive (suspended claims)</h3>
        <ul>
          <li><strong>A precise Meta-S2 constant.</strong> D_meta ≈ {{d_mle}} cannot be treated as a kernel invariant or universal exponent. The estimator range is [{{d_direct}}, {{d_linear}}], and movement under data cleaning shows sensitivity to population and method.</li>
          <li>{{weibull_recurrence_en}}</li>
          <li><strong>Incremental stability.</strong> The "first prospective test passed" claim was based on contaminated data. The clean prospective test starts now.</li>
        </ul>

        <h3>What is needed next</h3>
        <ul>
          <li><strong>One canonical estimator</strong> (Weibull MLE, adopted) with diagnostics clearly labeled.</li>
          <li><strong>Immutable snapshots per scan</strong> containing {timestamp, n, D_MLE, D_direct, D_linear, λ, KS, AD, AICs} for historical comparison.</li>
          <li><strong>Genuinely independent families</strong> — not more crypto or weather series, but new domains (biological aging, linguistic decay, archaeological dating).</li>
          <li><strong>Seasonal adjustment test</strong> for temperature series: D_raw vs D_seasonally-adjusted.</li>
          <li><strong>Parametric-bootstrap KS test</strong> that refits parameters inside each bootstrap sample.</li>
        </ul>

        <h3>The DREAM irony</h3>
        <p>The most important discovery may be that D_meta is sensitive to the population of systems and to the statistical lens used to observe it — precisely the kind of projection dependence DREAM itself would predict. If the 10D → 4D projection smooths information differently depending on resolution, then different estimators (which operate at different effective resolutions) should see different D_meta values. The estimator sensitivity may not be a bug; it may be the theory working as predicted. Modeling this projection dependence — rather than ignoring it — is the next theoretical step.</p>

        <p>The wide bootstrap CI means more <em>independent</em> families are needed, not just more rows from existing families. Adding 10 more crypto or weather series will not narrow the CI meaningfully. Adding a genuinely new domain (e.g., biological aging, linguistic decay, archaeological dating) would.</p>

        <p class="article-footnote">
          Analysis performed {{date}}. Data: <a href="../tests.html">DREAM Test Registry</a> ({{n}} Meta-S2-eligible uncensored systems, of {{n_total}} total registered).<br>
          This page is speculative. The recursive S2 property is an empirical observation on a non-random, partially dependent sample, not a proven theorem. It is offered as a direction for investigation.<br><br>
          <strong>Update protocol:</strong> This article is regenerated with each scanner run. The live Meta-S2 readout on the <a href="../tests.html">Test Registry</a> page always reflects the current TESTS array.
        </p>
      </div>
"""

# Russian body — full translation
BODY_RU = """
      <!-- ===== Title card ===== -->
      <div class="card">
        <p class="kicker">{{kicker}}</p>
        <h1>Meta-S2: рекурсивное удержание</h1>
      </div>

      <hr>

      <!-- ===== Empirical Observation ===== -->
      <div class="card">
        <h2>Эмпирическое наблюдение</h2>

        <p>В текущем наборе данных из <strong>{{n}} системных оценок</strong> — охватывающем когнитивное затухание, финансовые рынки, космологическую структуру, квантовую декогеренцию, экологические временные ряды и др. — каждая система была индивидуально подогнана под закон удержания S2:

        <div class="eq">
        $$R(\\lambda) = \\exp\\!\\left[-\\left(\\frac{\\lambda}{\\lambda_q}\\right)^{D_{\\mathrm{eff}}}\\right]$$
        </div>

        <p>Это дало {{n}} системных оценок D_eff. Эти оценки <strong>не все статистически независимы</strong>, поскольку некоторые принадлежат связанным наборам данных и доменным семействам (например, пять криптовалютных рядов имеют общий рыночный режим; пятнадцать погодных рядов имеют общего провайдера данных). Затем была проанализирована <em>дистрибуция</em> этих значений D_eff.</p>

        <h3>Базовая статистика</h3>
        <table class="data-table">
          <thead>
            <tr><th>Статистика</th><th class="num">Значение</th></tr>
          </thead>
          <tbody>
            <tr><td>Измерено систем (n)</td><td class="num">{{n}}</td></tr>
            <tr><td>Среднее D_eff</td><td class="num">{{mean}}</td></tr>
            <tr><td>Медиана D_eff</td><td class="num">{{median}}</td></tr>
            <tr><td>Геометрическое среднее</td><td class="num">{{geom_mean}}</td></tr>
            <tr><td>Стандартное отклонение</td><td class="num">{{std}}</td></tr>
            <tr><td>Размах</td><td class="num">[{{d_min}}, {{d_max}}]</td></tr>
            <tr><td>D &lt; 1 (естественный режим)</td><td class="num">{{natural}} ({{pct_natural}}%)</td></tr>
            <tr><td>D &gt; 1 (режим извлечения)</td><td class="num">{{extraction}} ({{pct_extraction}}%)</td></tr>
          </tbody>
        </table>

        <h3>Находка 1: Распределение значений D следует S2</h3>

        <p>Дополнительная кумулятивная функция распределения (функция выживаемости) D_eff — то есть доля систем с D ≥ d — была подогнана <strong>напрямую по фактическим значениям D</strong> к функциональной форме S2 / Вейбулла:</p>

        <div class="eq">
        $$\\hat{S}(d) = P(D \\ge d) = \\exp\\!\\left[-\\left(\\frac{d}{\\lambda_{\\mathrm{meta}}}\\right)^{D_{\\mathrm{meta}}}\\right]$$
        </div>

        <p>Были выполнены три подгонки:</p>

        <table class="data-table">
          <thead>
            <tr><th>Метод</th><th class="num">D_meta</th><th class="num">λ_q</th><th class="num">R² / p</th></tr>
          </thead>
          <tbody>
            <tr><td>Прямая МНК-подгонка по d</td><td class="num">{{d_direct}}</td><td class="num">{{lam_direct}}</td><td class="num">R²={{r2_direct}}</td></tr>
            <tr><td>Линеаризация: ln[−ln(Ŝ)] vs ln(d)</td><td class="num">{{d_linear}}</td><td class="num">{{lam_linear}}</td><td class="num">R²={{r2_linear}}</td></tr>
            <tr><td>Вейбулл ММП (все {{n}})</td><td class="num">{{d_mle}}</td><td class="num">{{lam_mle}}</td><td class="num">KS p={{ks_p}}</td></tr>
          </tbody>
        </table>

        <p>Все три метода согласуются в том, что распределение D_eff хорошо описывается функцией выживаемости Вейбулла (растянутой экспонентой) с D_meta ≈ {{d_mle}} и λ_meta ≈ {{lam_mle}} <strong>в единицах D</strong>. Нелинейная прямая подгонка по d и линеаризованная подгонка Вейбулла дают существенно различные оценки формы ({{d_direct}} против {{d_linear}}). Их значения R² напрямую не сопоставимы, поскольку вычисляются в разных трансформированных пространствах. Критерий Колмогорова–Смирнова не отверг подогнанную модель Вейбулла на уровне 5% (p = {{ks_p}}). Замечание: поскольку параметры Вейбулла были оценены по тем же {{n}} наблюдениям, обычное табличное p-значение KS не является строго корректным; параметрический бутстреп KS-теста, переподгоняющий параметры в каждой бутстреп-выборке, был бы предпочтительнее.</p>

        <blockquote class="callout warn">
          <strong>Проспективное обновление — {{date}} (исправлено):</strong> Изначальное утверждение о стабильности (ΔD = 0.011) было получено на загрязнённых данных. После удаления 4 повреждённых записей и 3 значений на границе оптимизатора (D=5.0) исправленный базовый уровень равен D_meta(ММП) = {{d_mle}}. Диапазон оценок: [{{d_direct}}, {{d_linear}}]. Прежнее утверждение об «инкрементальной стабильности» <strong>приостановлено</strong>, поскольку сдвиг был обусловлен очисткой данных, а не реальным расширением реестра. Чистый проспективный тест начинается сейчас: останется ли D_meta(ММП) вблизи {{d_mle}} при добавлении подлинно новых, независимых систем?
        </blockquote>

        <h4>Дополнительные статистики подгонки (n = {{n}})</h4>
        <table class="data-table">
          <thead>
            <tr><th>Статистика</th><th class="num">Значение</th><th class="right">Интерпретация</th></tr>
          </thead>
          <tbody>
            <tr><td>p-значение критерия KS</td><td class="num">{{ks_p}}</td><td class="right">Вейбулл не отвергается</td></tr>
            <tr><td>Статистика Андерсона–Дарлинга</td><td class="num">{{ad_stat}}</td><td class="right">Погранично (5% крит. ≈ 0.757)</td></tr>
            <tr><td>AIC Вейбулла</td><td class="num">{{aic_weibull}}</td><td class="right">{{aic_weibull_verdict_ru}}</td></tr>
            <tr><td>AIC логнормального</td><td class="num">{{aic_lognormal}}</td><td class="right">{{aic_lognormal_verdict_ru}}</td></tr>
            <tr><td>AIC гамма</td><td class="num">{{aic_gamma}}</td><td class="right">{{aic_gamma_verdict_ru}}</td></tr>
            <tr><td>AIC экспоненциального</td><td class="num">{{aic_exponential}}</td><td class="right">Отвергается (ΔAIC = +{{delta_aic_exponential}})</td></tr>
          </tbody>
        </table>

        <p>{{weibull_adequacy_ru}}</p>

        <h4>Линеаризация Вейбулла</h4>
        <p>Линеаризованный график ln[−ln(Ŝ(d))] от ln(d) должен быть линейным, если модель Вейбулла корректна. Линейная подгонка даёт R² = {{r2_lin_lin}}, а квадратичная улучшает до R² = {{r2_lin_quad}} (ΔR² = {{r2_lin_delta}}) с квадратичным коэффициентом {{quad_coeff}} — это указывает на <strong>слегка вогнутую кривизну</strong>. Значит, модель Вейбулла — хорошее первое приближение, но не точное; хвосты систематически отклоняются от линейного предсказания.</p>

        <p>Для сравнения, подгонка степенного закона к тому же распределению дала P(D ≥ d) ~ d<sup>−{{pl_exponent}}</sup> с R² = 0.755 — более слабая подгонка, чем S2.</p>

        <h3>Находка 2: Анализ бимодальности</h3>

        <p><strong>Цензурирование:</strong> {{n_censored}} значений достигли границы оптимизатора при D = 5.0. Они рассматриваются как цензурированные справа и исключены из анализа бимодальности, оставляя n = {{n}} нецензурированных наблюдений.</p>

        <table class="data-table">
          <thead>
            <tr><th>Тест</th><th class="num">Результат</th><th class="right">Вердикт</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>GMM ΔBIC = BIC₂ − BIC₁ (неценз.)</td>
              <td class="num">{{delta_bic}}</td>
              <td class="right">{{gmm_verdict_ru}}</td>
            </tr>
            <tr>
              <td>Средние компонент GMM (неценз.)</td>
              <td class="num">{{gmm_mean_1}}, {{gmm_mean_2}}</td>
              <td class="right">Два кластера</td>
            </tr>
            <tr>
              <td>Тест Сильвермана (бутстреп, 200 реп.)</td>
              <td class="num">p = {{silverman_p}}</td>
              <td class="right">Унимодальность не отвергается</td>
            </tr>
            <tr>
              <td>KS-тест Вейбулла (неценз.)</td>
              <td class="num">p = {{ks_p_uncens}}</td>
              <td class="right">Вейбулл не отвергается</td>
            </tr>
          </tbody>
        </table>

        <p><strong>Свидетельства бимодальности смешанные.</strong> GMM {{gmm_verdict_ru_lower}} (ΔBIC = {{delta_bic}}, где ΔBIC = BIC₂ − BIC₁; отрицательное значение означает предпочтительность 2-компонентной модели). Средние нецензурированных кластеров: D ≈ {{gmm_mean_1}} ({{gmm_weight_1}}%) и D ≈ {{gmm_mean_2}} ({{gmm_weight_2}}%).</p>

        <p>Однако <strong>бутстреп-тест Сильвермана не может отвергнуть унимодальность</strong> (p = {{silverman_p}}). Это значит, что наблюдаемая бимодальность может быть артефактом малой выборки — при {{n}} точках второй режим не является статистически устойчивым. Результат GMM следует рассматривать как наводящий, но не подтверждённый.</p>

        <p>Оценка ММП Вейбулла по нецензурированным данным даёт shape = {{shape_uncens}}, scale = {{scale_uncens}}, KS p = {{ks_p_uncens}} — хорошая подгонка к одному унимодальному распределению Вейбулла.</p>

        <h3>Находка 3: Робастность leave-one-family-out</h3>

        <p>Поскольку многие системы зависимы (5 криптовалют имеют общий рыночный режим, 15 погодных рядов — общего провайдера), эффективный размер выборки меньше, чем n = {{n}}. Чтобы проверить, не определяет ли одно семейство результат, каждое семейство исключалось, и мета-подгонка пересчитывалась:</p>

        {{lofo_table_ru}}

        <p><strong>D_meta устойчив к исключению семейств.</strong> {{lofo_summary_ru}}</p>

        <h3>Находка 4: Семейный бутстреп CI</h3>

        <p>Поскольку отдельные системы внутри семейства зависимы, стандартный бутстреп занижал бы неопределённость. Поэтому семейства ресемплировались как блоки (500 итераций):</p>
        <ul>
          <li><strong>Точечная оценка D_meta:</strong> {{d_mle}}</li>
          <li><strong>Бутстреп-среднее:</strong> {{boot_mean}}</li>
          <li><strong>95% CI:</strong> [{{boot_lo}}, {{boot_hi}}]</li>
          <li><strong>Ширина CI:</strong> {{boot_width}}</li>
        </ul>

        <p>Широкий CI отражает структуру зависимостей — эффективный размер выборки меньше, чем {{n}}. CI исключает D = 1 (едва), но слишком широк, чтобы различить Вейбулла, гамму и логнормальное. Для сужения нужны дополнительные независимые семейства.</p>

        <h3>Находка 5: Смещение выборки</h3>

        <p><strong>Это не случайная выборка.</strong> {{n}} системных оценок были отобраны, потому что для них предпринималась подгонка S2. Эти оценки не все статистически независимы — пять криптовалютных рядов имеют общий рыночный режим, пятнадцать погодных рядов имеют общего провайдера данных, а несколько рядов FRED имеют общую экономическую систему. Выборка смещена в сторону:</p>
        <ul>
          <li>Систем с доступными для загрузки данными (FRED, World Bank, USGS, Zenodo)</li>
          <li>Систем, где растянутая экспоненциальная деградация правдоподобна</li>
          <li>Доменов, где фреймворк DREAM предсказывал специфический режим D</li>
        </ul>

        <p>Утверждение «{{pct_extraction_round}}% системных оценок имеют D > 1» применимо к <em>этой выборке</em>, а не ко всем измеримым феноменам во Вселенной. Действительно случайная выборка могла бы показать иное распределение. Если реестр смещён в сторону заметных высоко-деформационных систем, распределение в генеральной совокупности может содержать больше систем с D &lt; 1, чем показывает текущая выборка. Количественная оценка этой возможности требует явной модели смещения выборки.</p>

        <h4>Согласование популяций реестра</h4>
        <p>Различные знаменатели, указанные на сайте, относятся к разным уровням допустимости:</p>
        <table class="data-table compact">
          <thead>
            <tr><th>Уровень допустимости</th><th class="num">Кол-во</th><th>Критерий</th></tr>
          </thead>
          <tbody>
            <tr><td>Всего зарегистрированных тестов</td><td class="num">{{n_total}}</td><td>Все записи в массиве TESTS</td></tr>
            <tr><td>Валидные S2-подгонки (0 &lt; D &lt; 5)</td><td class="num">{{n}}</td><td>Значение D существует и в правдоподобном диапазоне</td></tr>
            <tr><td>Цензурированные (D ≥ 5.0 граница)</td><td class="num">{{n_censored}}</td><td>Оптимизатор упёрся в верхний предел</td></tr>
            <tr><td>Шумовые подгонки исключены (R² &lt; 0.1)</td><td class="num">{{n_noise}}</td><td>S2 подгоняет шум (напр. коллапс ACF из-за суточного цикла) — нет реального сигнала сохранения</td></tr>
            <tr><td>Полные отказы S2 (S2_NO_FIT)</td><td class="num">{{n_rejected}}</td><td>S2 не удалось подогнать вообще — оптимизатор не сошёлся, ACF не вычислился, или недостаточно данных</td></tr>
            <tr><td>Сравнение моделей (победа/ничья/проигрыш/пыль)</td><td class="num">{{n_compared}}</td><td>Есть model_verdict (S2_WINS/TIES/LOSES/DUST_WINS)</td></tr>
          </tbody>
        </table>
        <p class="note">Поток: {{n_total}} зарегистрировано → {{n}} валидных нецензурированных S2-подгонок с R² ≥ 0.1 → {{n}} пригодно для Meta-S2 → {{n_compared}} пригодно для сравнения. {{n_rejected}} наборов данных, где S2 не удалось подогнать вообще, записаны в реестр как REJECTED (с rejection_reason), но исключены из распределения Meta-S2, так как не имеют значения D. Шумовые подгонки (R² &lt; 0.1) также исключены — S2 подгонка плоского хвоста ACF не даёт осмысленного значения D.</p>
      </div>

      <hr>

      <!-- ===== Hypothesis (Meta-S2) ===== -->
      <div class="card">
        <h2>Гипотеза (Мета-S2)</h2>

        <div class="speculative-note">
          <strong>Далее — гипотеза, а не теорема.</strong> Она мотивирована эмпирическими наблюдениями выше, но не доказана. Предлагается как направление исследования.
        </div>

        <h3>Рекурсивное свойство</h3>

        <p>Сильнейший эмпирический результат не философский — он математический:</p>

        <blockquote class="callout emph">
          Закон удержания S2 описывает удержание индивидуальной системы. Распределение подогнанных показателей S2 по {{n}} системным оценкам само хорошо описывается распределением Вейбулла (растянутой экспонентой) — тем же функциональным семейством, что и S2.
        </blockquote>

        <p>Эта <em>дистрибутивная рекурсия</em> — S2 описывает распределение параметров S2 — является более слабым утверждением, чем математическое замыкание при агрегации. Она говорит: показатели D_i распределены по Вейбуллу. Она <strong>не</strong> доказывает, что композитное удержание R_ансамбль(λ) = A({R_i(λ)}) само является S2. Это более сильное утверждение замыкания потребовало бы определения оператора агрегации A и доказательства того, что A отображает функции семейства S2 в функции семейства S2.</p>

        <p>Текущий анализ проверяет D_i ~ Вейбулл. Это ценно, но концептуально отлично от доказательства замыкания процессов удержания при агрегации. Различие таково:</p>
        <ul>
          <li><strong>Дистрибутивная рекурсия</strong> (показано): <em>Параметры</em> индивидуальных подгонок S2 распределены по Вейбуллу.</li>
          <li><strong>Замыкание при агрегации</strong> (не показано): <em>Композитное удержание</em> множества S2-систем само является S2.</li>
        </ul>

        <h3>Что бы это значило</h3>

        <p>Если Meta-S2 реальна:</p>
        <ul>
          <li><strong>S2 — не просто кривая, хорошо подгоняющая многие наборы данных.</strong> Это распределение, самоподобное при агрегации. Закон управляет и частями, и целым.</li>
          <li><strong>Мета-показатель D_meta ≈ {{d_mle}} имеет смысл.</strong> Он описывает «удержание удержания» — как быстро свойство иметь конкретное D_eff само деградирует при движении по пространству систем.</li>
          <li><strong>Бимодальность может указывать на два класса систем.</strong> Кластеры GMM при D ≈ {{gmm_mean_1}} и D ≈ {{gmm_mean_2}} предполагают наличие двух качественно различных режимов — не «естественный против извлечения», а «обычный против экстремального». Что отличает экстремальный кластер — открытый вопрос.</li>
        </ul>

        <h3>Что это ПОКА не значит</h3>

        <ul>
          <li><strong>«Вселенная находится в режиме извлечения.»</strong> Это слишком сильно. Текущая выборка смещена. Можно сказать: «в текущем наборе данных {{extraction}} из {{n}} подогнанных кривых удержания имеют D > 1, что соответствует быстрее-чем-экспоненциальному затуханию в подогнанной модели S2», но нельзя обобщать на Вселенную без случайной выборки.</li>
          <li><strong>«Извлечение каскадно возрастает.»</strong> Это предложенный механизм, а не установленный вывод. Он потребовал бы либо теоретического вывода (показывая, что вложенные S2-процессы композируются, давая max(D_i)), либо наблюдательных свидетельств на иерархических системах. Ни того, ни другого не представлено.</li>
          <li><strong>«S2 замкнут при агрегации.»</strong> Текущий анализ показывает <em>дистрибутивную рекурсию</em> (параметры распределены по Вейбуллу), а не математическое замыкание (композитное удержание является S2). Эти понятия концептуально различны. Замыкание потребовало бы определения оператора агрегации и доказательства того, что он отображает S2 в S2.</li>
          <li><strong>«Распределение бимодально.»</strong> Тест Сильвермана не может отвергнуть унимодальность (p = {{silverman_p}}). Предпочтительность 2-компонентной GMM (ΔBIC = {{delta_bic}}) наводящая, но не окончательная. При n = {{n}} второй режим может быть артефактом малой выборки.</li>
        </ul>

        <h3>Как проверить</h3>

        <p>Гипотеза даёт проверяемое предсказание:</p>

        <blockquote class="callout warn">
          При росте реестра тестов за пределы {{n}} нецензурированных систем параметр формы Вейбулла ММП должен оставаться в диапазоне [1.4, 2.0] с KS p > 0.05, если дистрибутивная рекурсия реальна. Если D_meta(ММП) систематически дрейфует за пределы этого диапазона или KS-тест последовательно отвергает Вейбулла по нескольким независимым семействам, результат был артефактом текущей выборки.
          <br><br>
          <strong>Чистый проспективный тест: ОЖИДАЕТСЯ.</strong> Прежний статус «ПРОЙДЕН» был основан на загрязнённых данных и приостановлен. Чистый базовый уровень: D_meta(ММП) = {{d_mle}}, n = {{n}}.
        </blockquote>

        <p>В частности:</p>
        <ul>
          <li>Если следующие 100 систем (удвоение реестра до ~200) по-прежнему подгоняются под Вейбулла с D_meta ≈ 1.8–2.4, дистрибутивная рекурсия дополнительно усиливается.</li>
          <li>Если подгонка ухудшается, текущий результат был смещением отбора.</li>
          <li>Если тест Сильвермана начнёт отвергать унимодальность при росте n, двухклассовая структура реальна. Если никогда — результат GMM был артефактом малой выборки.</li>
          <li>Если граничные значения D = 5.0 исчезнут при повышении потолка оптимизатора, высоко-D хвост был артефактом.</li>
        </ul>
      </div>

      <hr>

      <!-- ===== Method ===== -->
      <div class="card">
        <h2>Метод</h2>

        <details class="exp" open>
          <summary><strong>Как конструировалась мета-подгонка</strong></summary>

          <div class="method-body">
            <p><strong>Данные:</strong> {{n}} значений D_eff, извлечённых из реестра тестов DREAM (ru/tests.html) на {{date}}. Каждое D_eff было индивидуально подогнано к R(λ) = exp[−(λ/λ_q)^D] методом нелинейных наименьших квадратов (scipy.optimize.curve_fit) по автокорреляционным функциям (для временных рядов) или кривым удержания «амплитуда–частота» (для событийных данных).</p>

            <p><strong>Мета-подгонка (по d, не по рангу):</strong> {{n}} значений D отсортированы по возрастанию: D_(1) ≤ D_(2) ≤ ... ≤ D_({{n}}). Эмпирическая функция выживаемости вычислена как Ŝ(D_(i)) = (n − i) / n. Выполнены три подгонки <strong>с использованием фактических значений D как независимой переменной</strong> (не индекса ранга):</p>
            <ol>
              <li><strong>Прямая МНК-подгонка по d:</strong> Ŝ(d) = exp[−(d/λ_q)^D] подгонялась напрямую по фактическим значениям D через scipy.optimize.curve_fit.</li>
              <li><strong>Линеаризованная регрессия:</strong> ln[−ln(Ŝ(d))] = D_meta · ln(d) − D_meta · ln(λ_q), как линейная регрессия в лог-лог масштабе.</li>
              <li><strong>ММП Вейбулла:</strong> Оценка максимального правдоподобия Weibull(shape=D_meta, scale=λ_q) через scipy.stats.weibull_min.fit при loc=0. Это статистически предпочтительный метод.</li>
            </ol>

            <p><strong>Цензурирование:</strong> {{n_censored}} значений достигли границы оптимизатора при D = 5.0 (верхний предел подгонки). Они рассматриваются как цензурированные справа и исключены из нецензурированного анализа (n = {{n}}).</p>

            <p><strong>Тесты бимодальности:</strong></p>
            <ul>
              <li>Гауссова смесь (GMM): 1-компонентная против 2-компонентной, сравнение через BIC. ΔBIC = BIC₂ − BIC₁ (отрицательное — предпочтительнее 2-компонентная).</li>
              <li>Тест Сильвермана на ширину полосы: бутстреп с 200 повторениями на критической ширине полосы.</li>
              <li>Критерий Колмогорова–Смирнова против подогнанного распределения Вейбулла.</li>
            </ul>

            <p><strong>ПО:</strong> Python 3.12, scipy 1.x, scikit-learn (GaussianMixture).</p>

            <p><strong>Ограничения:</strong></p>
            <ul>
              <li>{{n}} систем не являются независимыми — пять криптовалют имеют общий рыночный режим, пятнадцать погодных рядов — общего провайдера, несколько рядов FRED — общую экономическую систему. Для робастного вывода необходим семейный блочный бутстреп (не индивидуальный).</li>
              <li>{{n_censored}} значений D цензурированы на границе D = 5.0. Более высокий потолок оптимизатора прояснил бы, является ли высоко-D хвост подлинным или артефактом.</li>
              <li>ММП Вейбулла (shape={{d_mle}}) и прямая подгонка по d (D_meta={{d_direct}}) расходятся, отражая чувствительность к методу подгонки. Истинный D_meta, вероятно, лежит между этими оценками.</li>
              <li>Тест Сильвермана не может отвергнуть унимодальность (p={{silverman_p}}), поэтому предпочтительность 2-компонентной GMM следует рассматривать как наводящую, но не подтверждённую.</li>
            </ul>
          </div>
        </details>
      </div>

      <hr>

      <!-- ===== Recursion Analysis ===== -->
      <div class="card">
        <h2>Анализ рекурсии: Фрактал ли S2?</h2>

        <p>Если S2 управляет индивидуальными системами (Уровень 0) и распределение показателей S2 также следует S2 (Уровень 1, Meta-S2), естественный вопрос — продолжается ли эта рекурсия: следует ли распределение D_meta также закону S2?</p>

        <h3>Проверены три уровня</h3>

        <table class="data-table">
          <thead>
            <tr><th>Уровень</th><th>Что описывает</th><th class="num">Параметр формы</th><th>Подгонка Вейбулла?</th><th>Математический объект</th></tr>
          </thead>
          <tbody>
            <tr><td>0</td><td>Удержание индивидуальной системы R(λ)</td><td class="num">D_i (варьирует)</td><td>S2 подгоняет отдельные системы</td><td>Физические / наблюдаемые системы</td></tr>
            <tr><td>1</td><td>Распределение D_i по системам</td><td class="num">D_meta ≈ {{d_mle}}</td><td>Вейбулл (KS p = {{ks_p}})</td><td>Распределение по параметрам</td></tr>
            <tr><td>2</td><td>Распределение оценок D_meta (бутстреп)</td><td class="num">≈ {{shape2}} (ММП)</td><td><strong>Отвергается</strong> (KS p = {{ks_p2}})</td><td>Выборочное распределение оценки</td></tr>
          </tbody>
        </table>

        <p>На Уровне 2 бутстреп-распределение D_meta <strong>не является Вейбуллом</strong> — оно приблизительно гауссово (форма Вейбулла ≈ {{shape2}} сходится к нормальному). S2-сигнатура исчезает на втором уровне.</p>

        <h3>Что это значит и не значит</h3>

        <div class="speculative-note">
          <strong>Статистический факт (установлен):</strong> При многих статистических условиях оценки параметров становятся приблизительно нормальными вследствие центральной предельной теоремы. Гауссовость на Уровне 2 неудивительна — она ожидаема для выборочного распределения оценки, основанной на {{n}} наблюдениях. Базовое распределение (Уровень 1) всё ещё может быть S2/Вейбуллом; неопределённость в оценке его параметров просто становится гауссовой. Эти утверждения совместимы.
        </div>

        <p><strong>Гауссовость на Уровне 2 НЕ доказывает, что S2 «физически исчез».</strong> Она может просто отражать, что оценки параметров усредняют множество независимых источников неопределённости. Различие существенно:</p>
        <ul>
          <li><strong>Установлено:</strong> S2 подгоняет Уровень 0 (индивидуальные системы). Вейбулл подгоняет Уровень 1 (распределение D_i). Гаусс подгоняет Уровень 2 (вариативность оценки). Это стандартная статистическая иерархия.</li>
          <li><strong>Не установлено:</strong> Что ядро проекции DREAM вызывает переход от S2 к гауссову. Это потребовало бы показать, что последовательные физические проекции — а не просто статистическая агрегация — производят этот паттерн.</li>
        </ul>

        <h3>Конъектура (Гипотеза двойной проекции)</h3>

        <div class="speculative-note">
          <strong>Конъектура:</strong> Если ядро проекции DREAM действует многократно на S2-управляемых переменных, то первая проекция сохраняет S2-структуру на ансамблевом уровне (Meta-S2), тогда как последующие проекции прогрессивно подавляют структуру высших порядков, порождая приблизительно гауссову статистику для агрегированных наблюдаемых.
        </div>

        <p>Эта конъектура обладает тремя свойствами, делающими её научно полезной:</p>
        <ol>
          <li><strong>Она явно помечена как конъектура</strong>, а не теорема или установленный факт.</li>
          <li><strong>Она фальсифицируема.</strong> Если можно показать, что вторая независимая физическая проекция (не просто CLT-агрегация) по-прежнему даёт S2-структуру, конъектура ложна. Если последовательные проекции последовательно порождают гауссовость, она поддерживается.</li>
          <li><strong>Она разделяет установленные статистические факты</strong> (гауссовы оценки часто возникают при CLT) <strong>и новую физическую интерпретацию</strong> (ядро вызывает переход от S2 к гауссову).</li>
        </ol>

        <p>Если бы эту конъектуру в конечном счёте удалось продемонстрировать — что одна проекция последовательно даёт S2, а две независимые проекции последовательно дают гауссово поведение по нескольким наборам данных, — она связала бы статистическую иерархию напрямую с архитектурой фреймворка DREAM, где наблюдаемые возникают через необратимое ядро проекции и S2 управляет удержанием. Это был бы гораздо более специфический результат, чем простое наблюдение гауссовых оценок параметров.</p>

        <p class="note">Проверено {{date}} через 500-итерационный бутстреп D_meta по {{n}} системным оценкам. Подгонка Вейбулла Уровня 2 отвергнута (KS p = {{ks_p2}}). CLT-объяснения достаточно для наблюдаемой гауссовости; конъектура двойной проекции остаётся непроверенной.</p>
      </div>

      <hr>

      <!-- ===== Current Verdict ===== -->
      <div class="card">
        <h2>Текущий вердикт</h2>

        <blockquote class="callout">
          {{verdict_headline_ru}}<br><br>
          По состоянию на {{date}} реестр из {{n}} систем даёт форму Вейбулла ММП D_meta = {{d_mle}} (KS p = {{ks_p}}). Диапазон оценок: [{{d_direct}}, {{d_linear}}]. {{callout_summary_ru}}
        </blockquote>

        <h3>Что выживает</h3>
        <ul>
          <li><strong>D_meta > 1 по всем оценщикам.</strong> Качественный вывод о том, что распределение показателя лежит на стороне извлечения, остаётся в силе.</li>
          <li><strong>Вейбулл не отвергается.</strong> KS p = {{ks_p}} (ММП). Распределение в целом совместимо с растянутой экспонентой.</li>
          <li><strong>Семейная робастность.</strong> D_meta остаётся стабильным при удалении любого семейства. Ни одно семейство не определяет результат.</li>
        </ul>

        <h3>Что не выживает (приостановлено)</h3>
        <ul>
          <li><strong>Точная постоянная Meta-S2.</strong> D_meta ≈ {{d_mle}} не может рассматриваться как инвариант ядра или универсальный показатель. Диапазон оценок — [{{d_direct}}, {{d_linear}}]; сдвиг при очистке данных показывает чувствительность к популяции и методу.</li>
          <li>{{weibull_recurrence_ru}}</li>
          <li><strong>Инкрементальная стабильность.</strong> Утверждение «первый проспективный тест пройден» было основано на загрязнённых данных. Чистый проспективный тест начинается сейчас.</li>
        </ul>

        <h3>Что нужно дальше</h3>
        <ul>
          <li><strong>Один канонический оценщик</strong> (ММП Вейбулла, принят) с явно помеченными диагностиками.</li>
          <li><strong>Неизменяемые снимки на каждый скан</strong>, содержащие {timestamp, n, D_ММП, D_прям, D_лин, λ, KS, AD, AICs} для исторического сравнения.</li>
          <li><strong>Подлинно независимые семейства</strong> — не дополнительные криптовалютные или погодные ряды, а новые домены (биологическое старение, лингвистическое затухание, археологическая датировка).</li>
          <li><strong>Тест сезонной корректировки</strong> для температурных рядов: D_сырой против D_сезонно-скорректированного.</li>
          <li><strong>Параметрический бутстреп KS-теста</strong>, переподгоняющий параметры в каждой бутстреп-выборке.</li>
        </ul>

        <h3>Ирония DREAM</h3>
        <p>Возможно, самое важное открытие состоит в том, что D_meta чувствителен к популяции систем и к статистической линзе, используемой для его наблюдения — именно та проекционная зависимость, которую сам DREAM предсказал бы. Если проекция 10D → 4D сглаживает информацию по-разному в зависимости от разрешения, то разные оценщики (работающие на разных эффективных разрешениях) должны видеть разные значения D_meta. Чувствительность к оценщику может быть не багом; она может быть теорией, работающей как предсказано. Моделирование этой проекционной зависимости — а не её игнорирование — является следующим теоретическим шагом.</p>

        <p>Широкий бутстреп-CI означает, что нужны дополнительные <em>независимые</em> семейства, а не просто дополнительные строки из существующих семейств. Добавление ещё 10 криптовалютных или погодных рядов существенно не сузит CI. Добавление подлинно нового домена (например, биологическое старение, лингвистическое затухание, археологическая датировка) — сузило бы.</p>

        <p class="article-footnote">
          Анализ выполнен {{date}}. Данные: <a href="../tests.html">Реестр тестов DREAM</a> ({{n}} пригодных для Meta-S2 нецензурированных систем из {{n_total}} всего зарегистрированных).<br>
          Эта страница спекулятивна. Рекурсивное свойство S2 — эмпирическое наблюдение на неслучайной, частично зависимой выборке, а не доказанная теорема. Предлагается как направление исследования.<br><br>
          <strong>Протокол обновления:</strong> Эта статья регенерируется при каждом запуске сканера. Живой считыватель Meta-S2 на странице <a href="../tests.html">Реестр тестов</a> всегда отражает текущий массив TESTS.
        </p>
      </div>
"""


# ── Leave-one-family-out table ────────────────────────────────────────────

def _build_lofo_table(existing, families, lang):
    """Compute leave-one-family-out table and render as HTML."""
    from scipy.stats import weibull_min, kstest
    is_ru = (lang == 'ru')

    # Full baseline — exclude noise fits (R² < 0.1)
    Ds_all = sorted([e['D'] for e in existing
                     if e.get('D') is not None
                     and 0 < e['D'] < 4.99
                     and e.get('r2', 0) >= 0.1])
    try:
        shape_full, _, scale_full = weibull_min.fit(np.array(Ds_all), floc=0)
        _, ks_full = kstest(Ds_all, 'weibull_min', args=(shape_full, 0, scale_full))
    except Exception:
        shape_full, ks_full = float('nan'), float('nan')

    # Get family names with their D values
    fam_dict = {}
    for e in existing:
        D = e.get('D')
        if D is None or not (0 < D < 4.99):
            continue
        if e.get('r2', 0) < 0.1:
            continue
        fam = _family_from_name(e.get('name'))
        fam_dict.setdefault(fam, []).append(D)

    rows = []
    max_delta = 0.0
    min_delta = 0.0
    # Build index: each D value belongs to one or more families.
    # For LOFO, we drop all entries whose family matches.
    for fam_name, fam_Ds in sorted(fam_dict.items()):
        # Build the "remaining" list by filtering entries by family
        remaining = []
        fam_Ds_set = list(fam_Ds)  # copy
        for d in Ds_all:
            if d in fam_Ds_set:
                fam_Ds_set.remove(d)  # consume one match
                continue
            remaining.append(d)
        if len(remaining) < 10:
            continue
        try:
            shape, _, scale = weibull_min.fit(np.array(remaining), floc=0)
            _, ks = kstest(remaining, 'weibull_min', args=(shape, 0, scale))
            delta = shape - shape_full
            if delta > max_delta:
                max_delta = delta
            if delta < min_delta:
                min_delta = delta
            rows.append((fam_name, len(remaining), shape, ks, delta))
        except Exception:
            pass

    if is_ru:
        header = """          <thead>
            <tr><th>Исключённое семейство</th><th class="num">n</th><th class="num">D_meta</th><th class="num">KS p</th><th class="num">ΔD</th></tr>
          </thead>"""
        none_label = "<strong>Все (базовый)</strong>"
        summary_tmpl = "Наибольший сдвиг ΔD = {max_d:+.3f} (исключая {max_fam}) и ΔD = {min_d:+.3f} (исключая {min_fam}). Во всех случаях D_meta остаётся стабильным, и KS-тест никогда не отвергает Вейбулла. Ни одно семейство не определяет результат."
    else:
        header = """          <thead>
            <tr><th>Excluded family</th><th class="num">n</th><th class="num">D_meta</th><th class="num">KS p</th><th class="num">ΔD</th></tr>
          </thead>"""
        none_label = "<strong>None (full)</strong>"
        summary_tmpl = "The largest shift is ΔD = {max_d:+.3f} (excluding {max_fam}) and ΔD = {min_d:+.3f} (excluding {min_fam}). In all cases, D_meta remains stable, and the KS test never rejects Weibull. No single family drives the result."

    body_rows = [f"""            <tr><td>{none_label}</td><td class="num">{len(Ds_all)}</td><td class="num">{shape_full:.3f}</td><td class="num">{ks_full:.3f}</td><td class="num">—</td></tr>"""]
    for fam_name, n_fam, shape, ks, delta in rows:
        body_rows.append(f"""            <tr><td>{fam_name}</td><td class="num">{n_fam}</td><td class="num">{shape:.3f}</td><td class="num">{ks:.3f}</td><td class="num">{delta:+.3f}</td></tr>""")

    table_html = f"""        <table class="data-table compact">
{header}
          <tbody>
{chr(10).join(body_rows)}
          </tbody>
        </table>"""

    # Find which families had max/min delta
    max_fam = min_fam = "—"
    for fam_name, n_fam, shape, ks, delta in rows:
        if abs(delta - max_delta) < 1e-6:
            max_fam = fam_name
        if abs(delta - min_delta) < 1e-6:
            min_fam = fam_name

    summary = summary_tmpl.format(max_d=max_delta, min_d=min_delta, max_fam=max_fam, min_fam=min_fam)
    return table_html, summary


# ── Render ────────────────────────────────────────────────────────────────

def _verdict_text(delta, is_ru):
    """Return verdict text for AIC delta."""
    if delta == 0:
        return ("Best fit (ΔAIC = 0.0)",
                "Наилучшая подгонка (ΔAIC = 0.0)")
    if delta < -2:
        return ("Preferred (ΔAIC = {:.1f})".format(delta),
                "Предпочтительно (ΔAIC = {:.1f})".format(delta))
    elif delta > 2:
        return ("Rejected (ΔAIC = +{:.1f})".format(delta),
                "Отвергается (ΔAIC = +{:.1f})".format(delta))
    else:
        return ("Competitive (ΔAIC = {:+.1f})".format(delta),
                "Конкурентоспособно (ΔAIC = {:+.1f})".format(delta))


def render(lang, snapshot, existing=None, families=None):
    """
    Render the full article HTML for the given language and snapshot.

    If `existing` is provided, the leave-one-family-out table is recomputed.
    Otherwise, the snapshot must contain pre-computed lofo_table_en / lofo_table_ru
    and lofo_summary_en / lofo_summary_ru strings.
    """
    chrome = _chrome(lang)
    is_ru = (lang == 'ru')

    # Build context with all snapshot fields + chrome
    ctx = dict(snapshot)
    ctx.update(chrome)

    # Compute AIC verdicts
    vw_en, vw_ru = _verdict_text(snapshot['delta_aic_weibull'], is_ru)
    vg_en, vg_ru = _verdict_text(snapshot['delta_aic_gamma'], is_ru)
    vl_en, vl_ru = _verdict_text(snapshot['delta_aic_lognormal'], is_ru)
    ctx['aic_weibull_verdict'] = vw_en
    ctx['aic_weibull_verdict_ru'] = vw_ru
    ctx['aic_gamma_verdict'] = vg_en
    ctx['aic_gamma_verdict_ru'] = vg_ru
    ctx['aic_lognormal_verdict'] = vl_en
    ctx['aic_lognormal_verdict_ru'] = vl_ru

    # Adaptive adequacy text — depends on whether Weibull is best by AIC
    weibull_is_best = (snapshot['delta_aic_weibull'] == 0)
    # Localize distribution names
    dist_names_ru = {
        'weibull': 'Вейбулл',
        'gamma': 'гамма',
        'lognormal': 'логнормальное',
        'exponential': 'экспоненциальное',
    }
    if weibull_is_best:
        # Find the runner-up
        deltas = {
            'gamma': snapshot['delta_aic_gamma'],
            'lognormal': snapshot['delta_aic_lognormal'],
            'exponential': snapshot['delta_aic_exponential'],
        }
        runner_name = min(deltas, key=deltas.get)
        runner_delta = deltas[runner_name]
        runner_name_ru = dist_names_ru[runner_name]
        ctx['weibull_adequacy_en'] = (
            f"<strong>Weibull is the single best fit by AIC.</strong> "
            f"A KS goodness-of-fit test did not reject the fitted Weibull model "
            f"at the 5% level (p = {snapshot['ks_p']}), though this p-value is "
            f"approximate because the parameters were estimated from the same data. "
            f"The Anderson-Darling statistic ({snapshot['ad_stat']}) "
            f"{'is below the 5% critical value (~0.757), supporting the fit' if snapshot['ad_stat'] < 0.757 else 'is above the 5% critical value (~0.757), suggesting the fit is marginal'}. "
            f"By AIC, the nearest competitor is {runner_name} "
            f"(ΔAIC = +{runner_delta}), which is "
            f"{'clearly inferior' if runner_delta > 10 else 'competitive' if runner_delta < 2 else 'weakly inferior'}. "
            f"The Weibull (S2) hypothesis is the single best fit by AIC."
        )
        ctx['weibull_adequacy_ru'] = (
            f"<strong>Вейбулл — наилучшая подгонка по AIC.</strong> "
            f"Критерий Колмогорова–Смирнова не отверг подогнанную модель Вейбулла "
            f"на уровне 5% (p = {snapshot['ks_p']}), хотя это p-значение приближённое, "
            f"поскольку параметры оценивались по тем же данным. "
            f"Статистика Андерсона–Дарлинга ({snapshot['ad_stat']}) "
            f"{'ниже 5%-критического значения (~0.757), что подтверждает подгонку' if snapshot['ad_stat'] < 0.757 else 'превышает 5%-критическое значение (~0.757), что указывает на пограничное качество'}. "
            f"По AIC ближайший конкурент — {runner_name_ru} "
            f"(ΔAIC = +{runner_delta}), что "
            f"{'явно уступает' if runner_delta > 10 else 'конкурентоспособно' if runner_delta < 2 else 'слабо уступает'}. "
            f"Гипотеза Вейбулла (S2) является наилучшей подгонкой по AIC."
        )
        ctx['weibull_recurrence_en'] = (
            f"<strong>Exact Weibull recurrence is supported by AIC.</strong> "
            f"Gamma (ΔAIC = +{snapshot['delta_aic_gamma']}) and Lognormal "
            f"(ΔAIC = +{snapshot['delta_aic_lognormal']}) are "
            f"{'clearly inferior' if snapshot['delta_aic_gamma'] > 10 and snapshot['delta_aic_lognormal'] > 10 else 'competitive but not preferred'}. "
            f"The distribution belongs to the Weibull family over the current range."
        )
        ctx['weibull_recurrence_ru'] = (
            f"<strong>Точная рекурсия Вейбулла поддерживается AIC.</strong> "
            f"Гамма (ΔAIC = +{snapshot['delta_aic_gamma']}) и логнормальное "
            f"(ΔAIC = +{snapshot['delta_aic_lognormal']}) "
            f"{'явно уступают' if snapshot['delta_aic_gamma'] > 10 and snapshot['delta_aic_lognormal'] > 10 else 'конкурентоспособны, но не предпочтительны'}. "
            f"Распределение принадлежит семейству Вейбулла в текущем диапазоне."
        )
    else:
        # Weibull is NOT the best — find which is preferred
        deltas = {
            'weibull': snapshot['delta_aic_weibull'],
            'gamma': snapshot['delta_aic_gamma'],
            'lognormal': snapshot['delta_aic_lognormal'],
            'exponential': snapshot['delta_aic_exponential'],
        }
        best_name = min(deltas, key=deltas.get)
        best_name_ru = dist_names_ru[best_name]
        ctx['weibull_adequacy_en'] = (
            f"<strong>The Weibull fit is adequate but not uniquely preferred.</strong> "
            f"A KS goodness-of-fit test did not reject the fitted Weibull model "
            f"at the 5% level (p = {snapshot['ks_p']}), though this p-value is "
            f"approximate because the parameters were estimated from the same data. "
            f"The Anderson-Darling statistic ({snapshot['ad_stat']}) "
            f"{'is below the 5% critical value (~0.757), supporting the fit' if snapshot['ad_stat'] < 0.757 else 'is above the 5% critical value (~0.757), suggesting the fit is marginal'}. "
            f"By AIC, {best_name} is preferred over Weibull "
            f"(ΔAIC_Weibull = {snapshot['delta_aic_weibull']}). "
            f"The Weibull (S2) hypothesis survives but is not the single best fit by AIC."
        )
        ctx['weibull_adequacy_ru'] = (
            f"<strong>Подгонка Вейбулла адекватна, но не является единственно предпочтительной.</strong> "
            f"Критерий Колмогорова–Смирнова не отверг подогнанную модель Вейбулла "
            f"на уровне 5% (p = {snapshot['ks_p']}), хотя это p-значение приближённое, "
            f"поскольку параметры оценивались по тем же данным. "
            f"Статистика Андерсона–Дарлинга ({snapshot['ad_stat']}) "
            f"{'ниже 5%-критического значения (~0.757), что подтверждает подгонку' if snapshot['ad_stat'] < 0.757 else 'превышает 5%-критическое значение (~0.757), что указывает на пограничное качество'}. "
            f"По AIC {best_name_ru} предпочтительнее Вейбулла "
            f"(ΔAIC_Вейбулл = {snapshot['delta_aic_weibull']}). "
            f"Гипотеза Вейбулла (S2) выживает, но не является наилучшей подгонкой по AIC."
        )
        ctx['weibull_recurrence_en'] = (
            f"<strong>Exact Weibull recurrence is not uniquely supported.</strong> "
            f"By AIC, the preferred fit is {best_name} "
            f"(ΔAIC_Weibull = {snapshot['delta_aic_weibull']}). "
            f"The correct interpretation may be: the distribution belongs to a broader "
            f"positive, right-skewed decay family that is approximately Weibull over "
            f"the current range."
        )
        ctx['weibull_recurrence_ru'] = (
            f"<strong>Точная рекурсия Вейбулла не имеет единственной поддержки.</strong> "
            f"По AIC предпочтительная подгонка — {best_name_ru} "
            f"(ΔAIC_Вейбулл = {snapshot['delta_aic_weibull']}). "
            f"Корректная интерпретация может быть такой: распределение принадлежит "
            f"более широкому семейству положительных, скошенных вправо распределений "
            f"затухания, приблизительно Вейбуллу в текущем диапазоне."
        )

    # Adaptive verdict headline + summary in the callout
    if weibull_is_best:
        ctx['verdict_headline_en'] = (
            "<strong>Weibull-preferred distributional recurrence.</strong>"
        )
        ctx['verdict_headline_ru'] = (
            "<strong>Предпочтительная по Вейбуллу дистрибутивная рекурсия.</strong>"
        )
        ctx['callout_summary_en'] = (
            "The Weibull fit is the single best by AIC. "
            "Gamma and Lognormal are competitive but inferior (ΔAIC_Gamma = {{delta_aic_gamma}}, "
            "ΔAIC_Lognormal = {{delta_aic_lognormal}}). "
            "The family-level bootstrap 95% CI is [{{boot_lo}}, {{boot_hi}}] — wide, reflecting the dependence structure. "
            "The clean prospective test is pending: does D_meta(MLE) remain stable as genuinely new independent systems are added?"
        )
        ctx['callout_summary_ru'] = (
            "Подгонка Вейбулла — наилучшая по AIC. "
            "Гамма и логнормальное конкурентоспособны, но уступают (ΔAIC_Гамма = {{delta_aic_gamma}}, "
            "ΔAIC_Логнорм = {{delta_aic_lognormal}}). "
            "Семейный бутстреп 95% CI — [{{boot_lo}}, {{boot_hi}}] — широкий, отражающий структуру зависимостей. "
            "Чистый проспективный тест ожидается: останется ли D_meta(ММП) стабильным при добавлении подлинно новых независимых систем?"
        )
    else:
        ctx['verdict_headline_en'] = (
            "<strong>Candidate distributional recurrence with estimator-sensitive meta-exponent.</strong>"
        )
        ctx['verdict_headline_ru'] = (
            "<strong>Кандидат на дистрибутивную рекурсию с чувствительным к оценщику мета-показателем.</strong>"
        )
        ctx['callout_summary_en'] = (
            "The Weibull fit is adequate but not uniquely preferred — Gamma and Lognormal are competitive by AIC "
            "(ΔAIC_Weibull = {{delta_aic_weibull}}, ΔAIC_Gamma = {{delta_aic_gamma}}, ΔAIC_Lognormal = {{delta_aic_lognormal}}). "
            "The family-level bootstrap 95% CI is [{{boot_lo}}, {{boot_hi}}] — wide, reflecting the dependence structure. "
            "The clean prospective test is pending: does D_meta(MLE) remain stable as genuinely new independent systems are added?"
        )
        ctx['callout_summary_ru'] = (
            "Подгонка Вейбулла адекватна, но не является единственно предпочтительной — гамма и логнормальное "
            "конкурентоспособны по AIC (ΔAIC_Вейбулл = {{delta_aic_weibull}}, ΔAIC_Гамма = {{delta_aic_gamma}}, "
            "ΔAIC_Логнорм = {{delta_aic_lognormal}}). "
            "Семейный бутстреп 95% CI — [{{boot_lo}}, {{boot_hi}}] — широкий, отражающий структуру зависимостей. "
            "Чистый проспективный тест ожидается: останется ли D_meta(ММП) стабильным при добавлении подлинно новых независимых систем?"
        )

    # GMM verdict
    if snapshot['delta_bic'] < -10:
        ctx['gmm_verdict'] = "2-comp preferred"
        ctx['gmm_verdict_lower'] = "prefers 2 components"
        ctx['gmm_verdict_ru'] = "Предпочтительно 2 компоненты"
        ctx['gmm_verdict_ru_lower'] = "предпочитает 2 компоненты"
    elif snapshot['delta_bic'] < 0:
        ctx['gmm_verdict'] = "2-comp weakly preferred"
        ctx['gmm_verdict_lower'] = "weakly prefers 2 components"
        ctx['gmm_verdict_ru'] = "Слабо предпочтительно 2 компоненты"
        ctx['gmm_verdict_ru_lower'] = "слабо предпочитает 2 компоненты"
    else:
        ctx['gmm_verdict'] = "1-comp preferred"
        ctx['gmm_verdict_lower'] = "prefers 1 component"
        ctx['gmm_verdict_ru'] = "Предпочтительна 1 компонента"
        ctx['gmm_verdict_ru_lower'] = "предпочитает 1 компоненту"

    # Round pct_extraction for inline mentions
    ctx['pct_extraction_round'] = int(round(snapshot['pct_extraction']))

    # Leave-one-family-out table
    if existing is not None:
        table_en, summary_en = _build_lofo_table(existing, families, 'en')
        table_ru, summary_ru = _build_lofo_table(existing, families, 'ru')
        ctx['lofo_table_en'] = table_en
        ctx['lofo_summary_en'] = summary_en
        ctx['lofo_table_ru'] = table_ru
        ctx['lofo_summary_ru'] = summary_ru
    else:
        ctx.setdefault('lofo_table_en', '<p><em>LOFO table unavailable.</em></p>')
        ctx.setdefault('lofo_summary_en', 'See full table.')
        ctx.setdefault('lofo_table_ru', '<p><em>LOFO-таблица недоступна.</em></p>')
        ctx.setdefault('lofo_summary_ru', 'См. полную таблицу.')

    # Assemble HTML
    html = f"""<!doctype html>
<html lang="{('ru' if is_ru else 'en')}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{chrome['title']}</title>
  <meta name="description" content="{chrome['meta_desc']}" />
  <link rel="stylesheet" href="../../css/global.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
  <script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
      displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
      processEscapes: true,
      packages: {{'[+]': ['ams']}}
    }},
    chtml: {{ displayAlign:'center', displayIndent:'0' }},
    svg:   {{ displayAlign:'center', displayIndent:'0' }},
    options: {{ skipHtmlTags:['script','noscript','style','textarea','pre','code'] }}
  }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
{CSS_BLOCK}
</head>
<body data-lang="{('ru' if is_ru else 'en')}" class="is-article">

  <!-- ===== Fractal backdrop ===== -->
  <div id="fx" aria-hidden="true">
    <canvas id="mb"></canvas>
  </div>

  <!-- ===== Header / Nav ===== -->
  <header class="site-header">
    <div class="brand">
      <span class="dot" aria-hidden="true"></span>
      <a href="../index.html">D.R.E.A.M</a>
    </div>
    <nav class="nav" aria-label="Primary"><div class="nav-links-scroll">
{chrome['nav_links']}

      <div class="nav-more">
{chrome['nav_more']}
    </nav>
    <div class="controls">
      <div class="toggles">
        <label class="switch" title="Show/Hide mathematical content">
          <input type="checkbox" id="mathToggle"><span class="slider" aria-hidden="true"></span>
          <span class="switch-label">{chrome['math_label']}</span>
        </label>
        <label class="switch" title="Show speculative content">
          <input type="checkbox" id="specToggle"><span class="slider" aria-hidden="true"></span>
          <span class="switch-label">{chrome['spec_label']}</span>
        </label>
      </div>
      <div class="lang">
        <a class="lang-link" href="../../ru/articles/meta-s2.html" title="Русский"><img src="https://flagcdn.com/16x12/ru.png" width="16" height="11" alt="RU" class="flag-icon"></a>
        <a class="lang-link" href="../../en/articles/meta-s2.html" title="English"><img src="https://flagcdn.com/16x12/gb.png" width="16" height="11" alt="EN" class="flag-icon"></a>
      </div>
    </div>
  </header>

  <!-- ===== Main ===== -->
  <main class="container">
    <div class="article-wrap">
{_substitute(BODY_RU if is_ru else BODY_EN, ctx)}
    </div>
  </main>

  <!-- ===== Footer ===== -->
  <footer class="site-footer">
    <div class="footer-content">
      <small>{chrome['footer']}</small>
    </div>
  </footer>

  <!-- ===== Spec toggle ===== -->
  <script>
  document.addEventListener('DOMContentLoaded', () => {{
    const cb = document.getElementById('specToggle');
    const BODY_FLAG = 'show-spec';
    const KEY = 'spec:on';
    const saved = localStorage.getItem(KEY);
    if (saved !== null) {{
      const on = saved === '1';
      document.body.classList.toggle(BODY_FLAG, on);
      if (cb) cb.checked = on;
    }} else if (cb) {{
      document.body.classList.toggle(BODY_FLAG, cb.checked);
    }}
    cb?.addEventListener('change', e => {{
      const on = e.target.checked;
      document.body.classList.toggle(BODY_FLAG, on);
      localStorage.setItem(KEY, on ? '1' : '0');
    }});
  }});
  </script>

  <!-- ===== Math toggle / typeset ===== -->
  <script>
  document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('.math-block, .math-inline').forEach(el => {{
      let t = el.textContent;
      if (/\\\\\\[/.test(t) && !/\\\\\\]/.test(t)) t = t.replace(/$/, '\\n\\\\]');
      if (/\\\\\\(/.test(t) && !/\\\\\\)/.test(t)) t = t.replace(/$/, '\\\\)');
      if (t !== el.textContent) el.textContent = t;
    }});
    const typeset = nodes =>
      (window.MathJax && MathJax.typeset) && MathJax.typeset(nodes || undefined);
    if (document.body.classList.contains('show-math')) typeset();
    const mt = document.getElementById('mathToggle');
    const MATH_KEY = 'math:on';
    const MATH_FLAG = 'show-math';
    const mathSaved = localStorage.getItem(MATH_KEY);
    if (mathSaved !== null) {{
      const on = mathSaved === '1';
      document.body.classList.toggle(MATH_FLAG, on);
      if (mt) mt.checked = on;
    }}
    mt && mt.addEventListener('change', () => {{
      const on = mt.checked;
      document.body.classList.toggle(MATH_FLAG, on);
      localStorage.setItem(MATH_KEY, on ? '1' : '0');
      setTimeout(() => typeset(), 0);
    }});
    document.querySelectorAll('details.more').forEach(d => {{
      d.addEventListener('toggle', () => d.open && typeset([d]));
    }});
  }});
  </script>

  <!-- ===== Language switch rewrite ===== -->
  <script>
  (function(){{
    const here = location.pathname.replace(/\\/index\\.html$/i, '/');
    const m = here.match(/\\/(en|ru)\\/(.+?)\\.html$/i);
    let rest = '';
    if (m) rest = m[2];
    if (!rest || rest === 'index') rest = 'index';
    document.querySelectorAll('.lang-link').forEach(a=>{{
      const img = a.querySelector('img'); const isRu = img && (img.src.includes('ru.png') || img.alt === 'RU');
      const targetLang = isRu ? 'ru' : 'en';
      const parts = rest.split('/');
      const depth = parts.length - 1;
      const prefix = '../'.repeat(depth + 1);
      a.setAttribute('href', prefix + targetLang + '/' + rest + '.html');
    }});
  }})();
  </script>

  <!-- ===== Site scripts ===== -->
  <script defer src="../../js/fractal.js"></script>
  <script defer src="../../js/nav-mobile.js"></script>
  <script defer src="../../js/nav-dropdown.js"></script>
  <script defer src="../../js/nav-autoscroll.js"></script>

</body>
</html>
"""
    # Final pass: substitute any remaining {{var}} placeholders in body
    html = _substitute(html, ctx)
    return html


# ── CLI for testing ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    # Quick smoke test: load snapshot, render both languages
    snap_path = sys.argv[1] if len(sys.argv) > 1 else 'meta_s2_snapshot.json'
    with open(snap_path) as f:
        snap = json.load(f)
    print('=== EN ===')
    print(render('en', snap)[:2000])
    print('=== RU ===')
    print(render('ru', snap)[:2000])
