#!/usr/bin/env python3
"""Add T7.1-related FAQ entries to kb/parsed_faqs_en.json and kb/parsed_faqs_ru.json."""
import json, os, copy

# New T7.1 FAQs (EN)
NEW_FAQS_EN = [
    {
        'number': 113,
        'category': 'Foundations',
        'question': 'What is T7.1 (Multi-Regime S2)?',
        'answer': 'T7.1 is a refinement of T7, confirmed in August 2026. It says: S2 is the LOCAL law of retention, but the global retention curve is a piecewise composition of S2 regimes. The kernel enters different regimes at reproducible fractional scale thresholds. Each regime is a clean S2; the transition scale clusters at fractional positions [0.25, 0.38] of the t-range across unrelated domains. The kernel itself does not change — the information state underneath it does. T7.1 introduces NO new postulates; it is derived from existing axioms T3 + A2 + A6. It is confirmed on 25 real datasets with binomial p = 3.4e-18 vs single-S2 null, and the positive control (synthetic 2-regime data with known t_break=0.3) is recovered in 98% of trials. See retention.html#t71 and theorems.html#t71.'
    },
    {
        'number': 114,
        'category': 'Foundations',
        'question': 'Is T7.1 the same as "multiple Meta-Manifolds"?',
        'answer': 'No. T7.1 was ORIGINALLY proposed as a multi-Meta-Manifold hypothesis: maybe the 4D world is a superposition of multiple projected structures. That strong hypothesis is NOT supported by the data. Probe A (drift vs mixture discriminator on 10 real curves): 10/10 classified as DRIFT, 0/10 as discrete mixture — closure violation is consistent with a SINGLE complicated kernel whose parameters drift. Probe B (cross-domain residual correlation): 18/30 significant pairs vs 1.5 expected, but no characteristic interference lag — excess correlation is contemporaneous, not at a hidden interference scale. The empirical signature is: one complicated kernel, evolving effective parameters, NOT multiple independent kernels interfering. This is exactly the identifiability wall (A2): we cannot prove "one Meta-Manifold" directly, but where we can discriminate, the data is consistent with one complicated kernel.'
    },
    {
        'number': 115,
        'category': 'Predictions',
        'question': 'What does T7.1 predict for new datasets?',
        'answer': 'T7.1 makes five falsifiable predictions: (P1) piecewise S2 with one searched breakpoint should strongly beat single-S2 (ΔAICc > 4) on real curves where the kernel crosses a regime boundary; (P2) transition positions cluster at reproducible fractional positions across unrelated domains (std < 0.20 in fractional coordinates); (P3) D systematically increases from regime 1 to regime 2 in >80% of cases (the kernel enters a regime where information is more susceptible to compression); (P4) pure single-S2 synthetic curves should NOT show this strong-improvement pattern above chance (binomial p < 0.001); (P5) the piecewise pipeline should recover known 2-regime t_break in >80% of synthetic 2-regime cases. ALL FIVE predictions confirmed on 25 real datasets (August 2026).'
    },
    {
        'number': 116,
        'category': 'Falsification',
        'question': 'How was T7.1 tested against null and positive controls?',
        'answer': 'Two controls were used. NULL 1: 60 synthetic single-S2 curves with the same n=80, t-range, and noise level as real data, run through the IDENTICAL piecewise pipeline. Result: only 20% showed strong piecewise improvement (vs 100% on real data), and the t_break fractional std was 0.178 (vs real 0.169). Binomial test: P(>= 25/25 strong improvement | null p=0.20) = 3.4e-18 — essentially impossible under single-S2 null. POSITIVE CONTROL: 60 synthetic 2-regime S2 curves with known true t_break = 0.3, run through the identical pipeline. Result: 98% of fits landed within 0.1 of the true t_break (median recovery error = 0.05). The pipeline therefore correctly identifies 2-regime structure when it exists, and correctly distinguishes single-S2 from 2-regime. Both controls pass.'
    },
    {
        'number': 117,
        'category': 'Kernel & Invariants',
        'question': 'How does T7.1 affect the kernel invariants (λ_q, D_eff)?',
        'answer': 'T7.1 does NOT eliminate the kernel invariants — it enriches them. Instead of a single (D_eff, λ_q) per dataset, you have a SEQUENCE of (D_k, λ_q,k) for each regime k, plus the transition scales λ*_k→k+1. The single kernel is still fixed; what evolves is its effective parameters as the probe scale sweeps through different information states. Operationally: when fitting S2 to a retention curve, ALWAYS test piecewise-S2 with a searched breakpoint as an alternative. Report ΔAICc, the optimal t_break fraction, and the regime pair (D_1, D_2). Flag entries where D_2 hits the upper bound (D=10) — these may need a different second-regime functional form (e.g., super-Gaussian). The transition scale λ* is itself a new invariant: it clusters at fractional positions [0.25, 0.38] of the t-range across domains, suggesting a universal "regime-change geometry" of the projection.'
    },
    {
        'number': 118,
        'category': 'Math & Topology',
        'question': 'What is the mathematical form of T7.1 (piecewise S2)?',
        'answer': 'R(λ) ≈ exp[-(λ/λ_{q,1})^D_1] for λ < λ*, and R(λ) ≈ A_2 · exp[-(λ/λ_{q,2})^D_2] for λ ≥ λ*. Each regime is a clean S2 law with its own (D, λ_q). A_2 is fixed by continuity at the transition: A_2 = exp[-(λ*/λ_{q,1})^D_1 + (λ*/λ_{q,2})^D_2]. The transition scale λ* is found by AICc minimization over a candidate grid (typically 20%-80% of the t-range). Generalization to N regimes: R(λ) = piecewise S2 with breakpoints λ*_1, λ*_2, ..., λ*_{N-1}. In practice 2 regimes suffice for 25/25 tested real datasets; 3-regime fits have not yet been needed. The closure-violation signal in the registry (30.3% of 208 entries show S2_DUST beats single S2 by ΔAICc ≤ -4) is now explained: those are datasets where the regime transition is sharp enough that single-S2 cannot approximate both regimes.'
    },
    {
        'number': 119,
        'category': 'Foundations',
        'question': 'Is T7.1 a new axiom or a derived consequence?',
        'answer': 'Derived. T7.1 introduces NO new postulates. It is the natural piecewise refinement of T7, derived from existing axioms: T3 (S2 local law), A2 (non-invertible projection — kernel cannot resolve sub-λ structure), and A6 (finite resolution implies regimes — coherence cliff separates two regimes). The derivation: at any single locality, retention is S2 with local parameters (T3). The kernel has finite resolution (A2), so as the probe scale sweeps, the kernel samples different parts of the underlying information state. A coherence cliff separates regimes (A6). When the information state changes character across the probe range, the kernel enters a new S2 regime at a characteristic transition scale λ*. The global curve is a piecewise composition. This is exactly what the Minimal-Extension Principle demands: the framework grows its consequence set (numerator) without growing its axiom set (denominator). T7.1 is a case study of the principle working as designed.'
    },
    {
        'number': 120,
        'category': 'Falsification',
        'question': 'What are the honest caveats of T7.1?',
        'answer': 'Four caveats. (1) 24% of datasets show D_2 hitting the upper bound (D=10), suggesting the "second regime" might not be S2 but a different decay law (e.g., super-Gaussian) that the S2 functional form approximates. The transition is real; the second-regime identity is partially open. (2) The clustering magnitude is modest: real std (0.169) is only 1.05x tighter than single-S2 null (0.178). Statistical significance is huge (binomial p = 3.4e-18) but the effect size is small. (3) Reproducibility is in fractional position, not absolute scale. Datasets have different t-ranges (days vs hours vs years), so "reproducible fractional position" might mean "the breakpoint naturally lands in the early-to-middle of any retention curve." (4) Sample size is 25 real datasets. More would strengthen the claim, but the existing evidence is decisive against the single-S2 null. D_1 and D_2 are not correlated (Spearman p=0.93) — the two regimes appear to be independently drawn, not constrained to a 1D manifold in (D_1, D_2) space.'
    },
]

# Russian translations (mirror)
NEW_FAQS_RU = [
    {
        'number': 113,
        'category': 'Foundations',
        'question': 'Что такое T7.1 (мультирежимный S2)?',
        'answer': 'T7.1 — это уточнение теоремы T7, подтверждённое в августе 2026 года. Утверждается: S2 является ЛОКАЛЬНЫМ законом удержания, но глобальная кривая удержания — это кусочно-составная композиция режимов S2. Ядро входит в разные режимы на воспроизводимых дробных порогах масштаба. Каждый режим — чистый S2; масштаб перехода кластеризуется в дробных позициях [0.25, 0.38] от диапазона t по несвязанным доменам. Само ядро не меняется — меняется информационное состояние под ним. T7.1 НЕ вводит новых постулатов; выводится из существующих аксиом T3 + A2 + A6. Подтверждено на 25 реальных датасетах с биномиальным p = 3.4e-18 против нулевой гипотезы (один S2), и положительный контроль (синтетические 2-режимные данные с известным t_break=0.3) восстанавливается в 98% случаев. См. retention.html#t71 и theorems.html#t71.'
    },
    {
        'number': 114,
        'category': 'Foundations',
        'question': 'T7.1 — это то же, что "множественные Мета-Многообразия"?',
        'answer': 'Нет. T7.1 ИЗНАЧАЛЬНО была предложена как гипотеза множественных Мета-Многообразий: возможно, наш 4D-мир — суперпозиция нескольких проецированных структур. Эта сильная гипотеза НЕ подтверждается данными. Зонд A (дискриминатор дрейфа vs смеси на 10 реальных кривых): 10/10 классифицированы как DRIFT, 0/10 как дискретная смесь — нарушение замкнутости согласуется с ОДНИМ сложным ядром, параметры которого дрейфуют. Зонд B (кросс-доменная корреляция остатков): 18/30 значимых пар против 1.5 ожидаемых, но без характерного лага интерференции — избыточная корреляция одновременная, не на скрытом масштабе интерференции. Эмпирическая сигнатура: одно сложное ядро с эволюционирующими эффективными параметрами, НЕ несколько независимых ядер, интерферирующих между собой. Это именно стена идентифицируемости (A2): мы не можем напрямую доказать "одно Мета-Многообразие", но там, где мы можем различить, данные согласуются с одним сложным ядром.'
    },
    {
        'number': 115,
        'category': 'Predictions',
        'question': 'Что T7.1 предсказывает для новых датасетов?',
        'answer': 'T7.1 делает пять фальсифицируемых предсказаний: (P1) кусочно-S2 с одним искомым breakpoint должен сильно превосходить одиночный S2 (ΔAICc > 4) на реальных кривых, где ядро пересекает границу режимов; (P2) позиции переходов кластеризуются на воспроизводимых дробных позициях по несвязанным доменам (std < 0.20 в дробных координатах); (P3) D систематически возрастает от режима 1 к режиму 2 в >80% случаев (ядро входит в режим, где информация более подвержена сжатию); (P4) чистые синтетические кривые одиночного S2 НЕ должны показывать этот паттерн сильного улучшения выше случайного (биномиальное p < 0.001); (P5) кусочно-S2 пайплайн должен восстанавливать известный 2-режимный t_break в >80% синтетических 2-режимных случаев. ВСЕ ПЯТЬ предсказаний подтверждены на 25 реальных датасетах (август 2026).'
    },
    {
        'number': 116,
        'category': 'Falsification',
        'question': 'Как T7.1 тестировалась против нулевого и положительного контроля?',
        'answer': 'Использовались два контроля. NULL 1: 60 синтетических кривых одиночного S2 с теми же n=80, диапазоном t и уровнем шума, что и реальные данные, пропущенные через ИДЕНТИЧНЫЙ кусочно-S2 пайплайн. Результат: только 20% показали сильное кусочное улучшение (vs 100% на реальных данных), а дробной std t_break составил 0.178 (vs 0.169 у реальных). Биномиальный тест: P(>= 25/25 сильных улучшений | null p=0.20) = 3.4e-18 — практически невозможно при нулевой гипотезе одиночного S2. ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ: 60 синтетических 2-режимных S2 кривых с известным истинным t_break = 0.3, пропущенные через идентичный пайплайн. Результат: 98% фитов легли в пределах 0.1 от истинного t_break (медианная ошибка восстановления = 0.05). Таким образом, пайплайн правильно идентифицирует 2-режимную структуру, когда она существует, и правильно отличает одиночный S2 от 2-режимного. Оба контроля проходят.'
    },
    {
        'number': 117,
        'category': 'Kernel & Invariants',
        'question': 'Как T7.1 влияет на инварианты ядра (λ_q, D_eff)?',
        'answer': 'T7.1 НЕ устраняет инварианты ядра — она их обогащает. Вместо одного (D_eff, λ_q) на датасет у вас есть ПОСЛЕДОВАТЕЛЬНОСТЬ (D_k, λ_{q,k}) для каждого режима k, плюс масштабы переходов λ*_{k→k+1}. Одно ядро по-прежнему фиксировано; эволюционируют его эффективные параметры по мере того, как масштаб зонда проходит через различные информационные состояния. Операционно: при фитинге S2 к кривой удержания ВСЕГДА тестируйте кусочно-S2 с искомым breakpoint как альтернативой. Сообщайте ΔAICc, оптимальную дробную позицию t_break и пару режимов (D_1, D_2). Помечайте записи, где D_2 достигает верхней границы (D=10) — им может потребоваться другая функциональная форма второго режима (например, супергауссиан). Масштаб перехода λ* сам по себе новый инвариант: он кластеризуется на дробных позициях [0.25, 0.38] от диапазона t по доменам, что предполагает универсальную "геометрию смены режимов" проекции.'
    },
    {
        'number': 118,
        'category': 'Math & Topology',
        'question': 'Какова математическая форма T7.1 (кусочно-S2)?',
        'answer': 'R(λ) ≈ exp[-(λ/λ_{q,1})^D_1] для λ < λ*, и R(λ) ≈ A_2 · exp[-(λ/λ_{q,2})^D_2] для λ ≥ λ*. Каждый режим — чистый закон S2 со своими (D, λ_q). A_2 фиксируется непрерывностью на переходе: A_2 = exp[-(λ*/λ_{q,1})^D_1 + (λ*/λ_{q,2})^D_2]. Масштаб перехода λ* находится минимизацией AICc по сетке кандидатов (обычно 20%-80% диапазона t). Обобщение до N режимов: R(λ) = кусочно-S2 с breakpoint-ами λ*_1, λ*_2, ..., λ*_{N-1}. На практике 2 режимов достаточно для 25/25 тестированных реальных датасетов; 3-режимные фиты пока не потребовались. Сигнал нарушения замкнутости в реестре (30.3% из 208 записей показывают, что S2_DUST превосходит одиночный S2 с ΔAICc ≤ -4) теперь объяснён: это датасеты, где переход режимов достаточно резкий, так что одиночный S2 не может аппроксимировать оба режима.'
    },
    {
        'number': 119,
        'category': 'Foundations',
        'question': 'T7.1 — это новая аксиома или производное следствие?',
        'answer': 'Производное. T7.1 НЕ вводит новых постулатов. Это естественное кусочное уточнение T7, выведенное из существующих аксиом: T3 (S2 — локальный закон), A2 (необратимая проекция — ядро не может разрешить суб-λ структуру), и A6 (конечное разрешение влечёт режимы — coherence cliff разделяет два режима). Вывод: в любой отдельной локальности удержание — это S2 с локальными параметрами (T3). Ядро имеет конечное разрешение (A2), поэтому при прохождении масштаба зонда ядро сэмплирует различные части лежащего в основе информационного состояния. Coherence cliff разделяет режимы (A6). Когда информационное состояние меняет характер по диапазону зонда, ядро входит в новый режим S2 на характеристическом масштабе перехода λ*. Глобальная кривая — кусочная композиция. Это именно то, чего требует Принцип Минимального Расширения: фреймворк растёт по множеству следствий (числитель), не увеличивая множество аксиом (знаменатель). T7.1 — пример работы принципа как задумано.'
    },
    {
        'number': 120,
        'category': 'Falsification',
        'question': 'Каковы честные оговорки T7.1?',
        'answer': 'Четыре оговорки. (1) 24% датасетов показывают D_2 достигающим верхней границы (D=10), что предполагает, что "второй режим" может быть не S2, а другим законом затухания (например, супергауссианом), который форма S2 аппроксимирует. Переход реален; идентичность второго режима частично открыта. (2) Магнитуда кластеризации скромная: реальный std (0.169) только в 1.05 раза плотнее нулевого одиночного S2 (0.178). Статистическая значимость огромна (биномиальное p = 3.4e-18), но размер эффекта невелик. (3) Воспроизводимость в дробной позиции, не в абсолютном масштабе. У датасетов разные диапазоны t (дни, часы, годы), поэтому "воспроизводимая дробная позиция" может означать "breakpoint естественно ложится в начало-середину любой кривой удержания". (4) Размер выборки — 25 реальных датасетов. Больше данных усилило бы утверждение, но существующие данные уже решающе отклоняют нулевую гипотезу одиночного S2. D_1 и D_2 не коррелируют (Spearman p=0.93) — два режима кажутся независимо выбранными, не ограниченными одномерным многообразием в пространстве (D_1, D_2).'
    },
]


def add_faqs(json_path, new_faqs):
    with open(json_path) as f:
        d = json.load(f)
    existing_nums = set(f.get('number', 0) for f in d['faqs'])
    added = 0
    for nf in new_faqs:
        if nf['number'] in existing_nums:
            print(f'  SKIP (already exists): #{nf["number"]} - {nf["question"][:50]}')
            continue
        d['faqs'].append(nf)
        added += 1
    # Sort by number
    d['faqs'].sort(key=lambda f: f.get('number', 0))
    # Update metadata if present
    if 'metadata' in d and isinstance(d['metadata'], dict):
        d['metadata']['total'] = len(d['faqs'])
        d['metadata']['updated'] = '2026-08-26'
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f'  Added {added} new FAQs to {json_path} (total now: {len(d["faqs"])})')


print('Adding EN FAQs:')
add_faqs('/home/z/my-project/dream_repo/kb/parsed_faqs_en.json', NEW_FAQS_EN)
print()

# For RU, renumber #113 and #114 to #121 and #122 (since 113/114 were already taken)
NEW_FAQS_RU_RENUMBERED = copy.deepcopy(NEW_FAQS_RU)
NEW_FAQS_RU_RENUMBERED[0]['number'] = 121  # was 113
NEW_FAQS_RU_RENUMBERED[1]['number'] = 122  # was 114

print('Adding RU FAQs (renumbered 113→121, 114→122):')
add_faqs('/home/z/my-project/dream_repo/kb/parsed_faqs_ru.json', NEW_FAQS_RU_RENUMBERED)
