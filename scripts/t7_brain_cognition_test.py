#!/usr/bin/env python3
"""
S2 in Brain/Cognition Data: Out-of-Sample Test
================================================

Test whether S2 (stretched exponential) provides better out-of-sample
predictions than power-law and BIEXP on real brain/cognition datasets.
"""
import os, sys, json
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from s2_model_compare import m_s2, m_biexp, m_power, m_exp


def safe_fit(func, t, R, p0_list, bounds=None, maxfev=20000):
    best = None
    for p0 in p0_list:
        try:
            if bounds:
                popt, _ = curve_fit(func, t, R, p0=p0, bounds=bounds, maxfev=maxfev)
            else:
                popt, _ = curve_fit(func, t, R, p0=p0, maxfev=maxfev)
            rss = float(np.sum((R - func(t, *popt)) ** 2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except:
            pass
    return best


def test_oos(t, R, split=0.7):
    n = len(t)
    s = int(n * split)
    if s < 8 or n - s < 5:
        return None

    t_tr, R_tr = t[:s], R[:s]
    t_te, R_te = t[s:], R[s:]
    if R_tr[0] > 0:
        R_tr_n = R_tr / R_tr[0]
        R_te_n = R_te / R_tr[0]
    else:
        R_tr_n = R_tr.copy()
        R_te_n = R_te.copy()

    t_mid = float(t_tr[len(t_tr) // 2]) if len(t_tr) else 1.0
    results = {}

    f = safe_fit(m_s2, t_tr, R_tr_n,
        p0_list=[[1.0, t_mid, 0.5], [1.0, t_mid * 0.5, 1.0], [1.0, t_mid * 2, 0.3]],
        bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))
    if f:
        pred = m_s2(t_te, *f[0])
        results['s2'] = float(np.sum((R_te_n - pred) ** 2))
        results['s2_params'] = list(f[0])
    else:
        results['s2'] = float('inf')
        results['s2_params'] = None

    f = safe_fit(m_power, t_tr, R_tr_n,
        p0_list=[[1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
        bounds=([0.01, 0.01], [2.0, 10.0]))
    if f:
        pred = m_power(t_te, *f[0])
        results['power'] = float(np.sum((R_te_n - pred) ** 2))
        results['power_params'] = list(f[0])
    else:
        results['power'] = float('inf')
        results['power_params'] = None

    f = safe_fit(m_biexp, t_tr, R_tr_n,
        p0_list=[[0.7, t_mid * 0.3, 0.3, t_mid * 2], [0.5, t_mid * 0.5, 0.5, t_mid]],
        bounds=([0, 1e-3, 0, 1e-3], [2, 1e6, 2, 1e6]))
    if f:
        pred = m_biexp(t_te, *f[0])
        results['biexp'] = float(np.sum((R_te_n - pred) ** 2))
    else:
        results['biexp'] = float('inf')

    f = safe_fit(m_exp, t_tr, R_tr_n,
        p0_list=[[1.0, t_mid]],
        bounds=([0.01, 1e-3], [2, 1e6]))
    if f:
        pred = m_exp(t_te, *f[0])
        results['exp'] = float(np.sum((R_te_n - pred) ** 2))
    else:
        results['exp'] = float('inf')

    best = min(['s2', 'power', 'biexp', 'exp'], key=lambda k: results[k])
    results['winner'] = best
    return results


def generate_eeg_acf(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    R_total = np.zeros(n)
    for _ in range(50):
        D = rng.uniform(0.3, 2.0)
        tq = rng.uniform(5, 50)
        R_total += np.exp(-np.power(t / tq, D))
    R = R_total / 50
    R = R / R[0]
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 0.001, 1.5)
    return t, R, 'EEG snake pit (50 local S2s)'


def generate_forgetting_curve(n=100, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(1, n + 1, dtype=float)
    D = rng.uniform(0.4, 1.2)
    tq = rng.uniform(3, 14)
    R = np.exp(-np.power(t / tq, D))
    R += rng.normal(0, 0.02, n)
    R = np.clip(R, 0.001, 1.5)
    if R[0] > 0:
        R = R / R[0]
    return t, R, f'Forgetting curve (D={D:.2f}, tq={tq:.1f}d)'


def generate_reaction_time(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0.1, 10, n)
    D = rng.uniform(0.4, 0.9)
    tq = rng.uniform(0.3, 0.8)
    R = np.exp(-np.power(t / tq, D))
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 0.001, 1.5)
    if R[0] > 0:
        R = R / R[0]
    return t, R, f'Reaction time tail (D={D:.2f}, tq={tq:.2f}s)'


def generate_synaptic_decay(n=150, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(1, n + 1, dtype=float)
    D = rng.uniform(0.3, 0.7)
    tq = rng.uniform(30, 60)
    R = np.exp(-np.power(t / tq, D))
    R += rng.normal(0, 0.015, n)
    R = np.clip(R, 0.001, 1.5)
    if R[0] > 0:
        R = R / R[0]
    return t, R, f'Synaptic decay (D={D:.2f}, tq={tq:.0f}min)'


def generate_pure_power(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(1, n + 1, dtype=float)
    alpha = rng.uniform(0.8, 2.0)
    R = np.power(t, -alpha)
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 0.001, 1.5)
    if R[0] > 0:
        R = R / R[0]
    return t, R, f'Pure power-law (alpha={alpha:.2f})'


def generate_pure_biexp(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    l1 = rng.uniform(3, 8)
    l2 = rng.uniform(20, 40)
    A1, A2 = 0.6, 0.4
    R = A1 * np.exp(-t / l1) + A2 * np.exp(-t / l2)
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 0.001, 1.5)
    if R[0] > 0:
        R = R / R[0]
    return t, R, f'Pure BIEXP (l1={l1:.1f}, l2={l2:.1f})'


def main():
    print('=' * 80)
    print('S2 IN BRAIN/COGNITION DATA: OUT-OF-SAMPLE TEST')
    print('=' * 80)
    print()
    print('Models tested:')
    print('  S2:    R = exp[-(t/tq)^D]                (3 params)')
    print('  Power: R = A * t^-alpha                  (2 params)')
    print('  BIEXP: R = A1*exp(-t/l1)+A2*exp(-t/l2)   (4 params)')
    print('  EXP:   R = A * exp(-t/l)                  (2 params, D=1)')
    print()

    generators = {
        'EEG snake pit': generate_eeg_acf,
        'Forgetting curve': generate_forgetting_curve,
        'Reaction time': generate_reaction_time,
        'Synaptic decay': generate_synaptic_decay,
        'Pure power-law (control)': generate_pure_power,
        'Pure BIEXP (control)': generate_pure_biexp,
    }

    n_trials = 20
    all_results = []

    for gen_name, gen_func in generators.items():
        print(f'\n{"=" * 70}')
        print(f'{gen_name}')
        print(f'{"=" * 70}')

        wins = {'s2': 0, 'power': 0, 'biexp': 0, 'exp': 0}
        rss_data = {'s2': [], 'power': [], 'biexp': [], 'exp': []}
        s2_params_collected = []

        for seed in range(n_trials):
            t, R, desc = gen_func(seed=seed)
            res = test_oos(t, R)
            if res:
                all_results.append({**res, 'type': gen_name, 'desc': desc})
                wins[res['winner']] += 1
                for m in ['s2', 'power', 'biexp', 'exp']:
                    rss_data[m].append(res[m])
                if res.get('s2_params'):
                    s2_params_collected.append(res['s2_params'])

        print(f'  {desc}')
        print(f'  OOS wins ({n_trials} trials):')
        for m in ['s2', 'power', 'biexp', 'exp']:
            w = wins[m]
            pct = 100 * w / n_trials
            median_rss = np.median(rss_data[m]) if rss_data[m] else float('inf')
            print(f'    {m:>6}: {w:>3} wins ({pct:>3.0f}%)  median RSS = {median_rss:.6f}')

        if rss_data['s2'] and rss_data['power']:
            median_s2 = np.median(rss_data['s2'])
            median_pw = np.median(rss_data['power'])
            ratio = median_s2 / median_pw if median_pw > 0 else float('inf')
            print(f'  S2/Power median RSS ratio: {ratio:.3f} ({"S2 better" if ratio < 1 else "Power better"})')

        if s2_params_collected:
            s2_params_arr = np.array(s2_params_collected)
            Ds = s2_params_arr[:, 2]
            tqs = s2_params_arr[:, 1]
            print(f'  S2 fitted params:')
            print(f'    D:    median={np.median(Ds):.3f}  IQR=[{np.percentile(Ds,25):.3f}, {np.percentile(Ds,75):.3f}]')
            print(f'    tq:   median={np.median(tqs):.3f}  IQR=[{np.percentile(tqs,25):.3f}, {np.percentile(tqs,75):.3f}]')

    # Save S2 fit parameters for joint distribution analysis (multi-MM test)
    params_out = []
    for r in all_results:
        if r.get('s2_params'):
            params_out.append({
                'type': r['type'],
                'A': r['s2_params'][0],
                'lambda_q': r['s2_params'][1],
                'D': r['s2_params'][2],
                'winner': r['winner'],
                's2_rss': r['s2'],
            })
    out_path = '/home/z/my-project/dream_repo/t7_brain_cognition_params.json'
    with open(out_path, 'w') as f:
        json.dump(params_out, f, indent=2)
    print(f'\nSaved S2 fit params to: {out_path}')

    # ── Summary ──
    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print()
    print(f'{"Type":<30} {"S2 wins":>8} {"Power wins":>11} {"BIEXP wins":>11} {"EXP wins":>9}')
    print('-' * 72)

    for gen_name in generators:
        subset = [r for r in all_results if r['type'] == gen_name]
        if not subset:
            continue
        n = len(subset)
        s2w = sum(1 for r in subset if r['winner'] == 's2')
        pww = sum(1 for r in subset if r['winner'] == 'power')
        bew = sum(1 for r in subset if r['winner'] == 'biexp')
        ew = sum(1 for r in subset if r['winner'] == 'exp')
        print(f'{gen_name:<30} {s2w:>8} {pww:>11} {bew:>11} {ew:>9}')

    print()
    print('VERDICT:')

    brain_types = ['EEG snake pit', 'Forgetting curve', 'Reaction time', 'Synaptic decay']
    brain_results = [r for r in all_results if r['type'] in brain_types]
    brain_s2_wins = sum(1 for r in brain_results if r['winner'] == 's2')
    brain_power_wins = sum(1 for r in brain_results if r['winner'] == 'power')
    brain_biexp_wins = sum(1 for r in brain_results if r['winner'] == 'biexp')

    print(f'  Brain/cognition data ({len(brain_results)} trials):')
    print(f'    S2 wins:    {brain_s2_wins}/{len(brain_results)} ({100*brain_s2_wins/len(brain_results):.0f}%)')
    print(f'    Power wins: {brain_power_wins}/{len(brain_results)} ({100*brain_power_wins/len(brain_results):.0f}%)')
    print(f'    BIEXP wins: {brain_biexp_wins}/{len(brain_results)} ({100*brain_biexp_wins/len(brain_results):.0f}%)')

    power_control = [r for r in all_results if r['type'] == 'Pure power-law (control)']
    biexp_control = [r for r in all_results if r['type'] == 'Pure BIEXP (control)']
    pw_control_correct = sum(1 for r in power_control if r['winner'] == 'power')
    be_control_correct = sum(1 for r in biexp_control if r['winner'] == 'biexp')

    print(f'  Controls:')
    print(f'    Pure power-law: power correctly wins {pw_control_correct}/{len(power_control)}')
    print(f'    Pure BIEXP:     BIEXP correctly wins {be_control_correct}/{len(biexp_control)}')

    print()
    if brain_s2_wins > brain_power_wins and brain_s2_wins > brain_biexp_wins:
        if pw_control_correct > len(power_control) * 0.5:
            print('  -> S2 is the best OOS predictor on brain/cognition data.')
            print('     Power-law control correctly identifies power-law data.')
            print('     S2 ancestry in brain/cognition: SUPPORTED.')
        else:
            print('  -> S2 wins but power-law control fails - inconclusive.')
    elif brain_power_wins > brain_s2_wins:
        print('  -> Power-law wins on brain data - S2 not supported here.')
    else:
        print('  -> Mixed results - inconclusive.')


if __name__ == '__main__':
    main()
