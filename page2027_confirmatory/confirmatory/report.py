"""Question-paired statistics, measured costs, and complete-result validation."""
import csv
import json
import math
import sqlite3
from pathlib import Path
from .core import MODELS, POLICIES, atomic_json, digest
from .workflow import load_run, calibrated_inputs


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center - half, center + half


def paired_bootstrap(delta, seed=2027, repetitions=2000):
    import numpy as np
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for i in range(repetitions):
        means[i] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(x) for x in np.quantile(means, [.025, .975])]


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, previous = [0.] * len(order), 0.
    for j, i in enumerate(order):
        previous = max(previous, min(1., p_values[i] * (len(order) - j)))
        adjusted[i] = previous
    return adjusted


def report(args):
    import numpy as np
    from scipy.stats import binomtest
    out = Path(args.out)
    manifest, splits = load_run(out, strict_runtime=False)
    router, cal = calibrated_inputs(out, manifest)
    all_rows, keys = [], set()
    expected_ids = {(r['dataset'], r['id']) for r in splits['evaluation']}
    for path in sorted(out.glob('eval-shard-*.sqlite')):
        db = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        if (identity['manifest'] != digest(manifest) or identity['router'] != digest(router)
                or identity['calibration'] != digest(cal) or identity['stage'] != 'evaluation'):
            raise RuntimeError(f'Incompatible shard {path}')
        for (payload,) in db.execute('SELECT payload FROM episodes'):
            row = json.loads(payload)
            key = (row['policy'], row['dataset'], row['question_id'])
            if key in keys or row['policy'] not in POLICIES:
                raise RuntimeError(f'Duplicate or unexpected policy {key}; do not mix shard layouts')
            if (row['dataset'], row['question_id']) not in expected_ids:
                raise RuntimeError('Evaluation contains an unexpected question')
            keys.add(key)
            all_rows.append(row)
        db.close()
    expected = {(p, ds, qid) for p in POLICIES for ds, qid in expected_ids}
    if keys != expected:
        raise RuntimeError(f'Incomplete evaluation: {len(keys)}/{len(expected)}. '
                           'Finish/resume all shards before publishing a report.')
    all_rows.sort(key=lambda r: (r['dataset'], r['policy'], r['question_id']))
    destination = out / 'report'
    destination.mkdir(exist_ok=True)
    flat, calls, step_rows = [], [], []
    for r in all_rows:
        base = {k: v for k, v in r.items() if k not in ['steps', 'calls']}
        for tier in MODELS:
            selected = [c for c in r['calls'] if c['tier'] == tier]
            base[f'{tier}_calls'] = len(selected)
            base[f'{tier}_input_tokens'] = sum(c['input_tokens'] for c in selected)
            base[f'{tier}_output_tokens'] = sum(c['output_tokens'] for c in selected)
        base['invalid_steps'] = sum(s['invalid'] for s in r['steps'])
        flat.append(base)
        key = {k: r[k] for k in ['policy', 'dataset', 'question_id']}
        for c in r['calls']:
            calls.append({**key, **{k: v for k, v in c.items() if k != 'features'},
                          'features_json': json.dumps(c['features'], sort_keys=True)})
        for s in r['steps']:
            step_rows.append({**key, **{k: v for k, v in s.items() if k != 'features'},
                              'features_json': json.dumps(s['features'], sort_keys=True)})
    write_csv(destination / 'episodes.csv', flat)
    write_csv(destination / 'calls.csv', calls)
    write_csv(destination / 'steps.csv', step_rows)
    summary, paired, random_means = [], [], []
    by_group = {}
    for ds in ['hotpotqa', 'musique']:
        for p in POLICIES:
            rows = [r for r in flat if r['dataset'] == ds and r['policy'] == p]
            by_group[ds, p] = rows
            n, k = len(rows), sum(r['success'] for r in rows)
            ci = wilson(k, n)
            s = {'dataset': ds, 'policy': p, 'n': n, 'successes': k,
                 'success': k/n, 'success_ci_low': ci[0], 'success_ci_high': ci[1]}
            for col in ['alias_success', 'llm_call_ratio', 'n_steps', 'call_wall_s',
                        'cuda_elapsed_s', 'episode_wall_s', 'input_tokens', 'output_tokens',
                        'parameter_b_tokens', 'small_calls', 'mid_calls', 'large_calls', 'invalid_steps']:
                s['mean_' + col] = sum(r[col] for r in rows) / n
            s['mean_total_tokens'] = s['mean_input_tokens'] + s['mean_output_tokens']
            s['peak_allocated_gib'] = max(r['peak_allocated_bytes'] for r in rows) / 1024**3
            summary.append(s)
        for p in POLICIES:
            if p == 'always_large':
                continue
            reference = ('always_mid' if p.startswith('cascade') else
                         'entropy_2.0' if p.startswith('random') or p == 'query' else 'always_large')
            rows = by_group[ds, p]
            ref = {r['question_id']: r for r in by_group[ds, reference]}
            delta = [int(r['success']) - int(ref[r['question_id']]['success']) for r in rows]
            wins, losses = delta.count(1), delta.count(-1)
            pval = float(binomtest(wins, wins+losses, .5).pvalue) if wins+losses else 1.
            ci = paired_bootstrap(delta)
            paired.append({'dataset': ds, 'policy': p, 'reference': reference, 'n_questions': len(rows),
                           'accuracy_delta': float(np.mean(delta)), 'delta_ci_low': ci[0],
                           'delta_ci_high': ci[1], 'discordant_wins': wins,
                           'discordant_losses': losses, 'mcnemar_exact_p': pval})
        # Average the seeds WITHIN question. Never count the 3 seeds as 3000 questions.
        seeds = [{r['question_id']: r for r in by_group[ds, f'random_{seed}']} for seed in [17,29,43]]
        reference = by_group[ds, 'entropy_2.0']
        means = [sum(s[r['question_id']]['success'] for s in seeds)/3 for r in reference]
        delta = [v - r['success'] for v, r in zip(means, reference)]
        ci = paired_bootstrap(delta)
        random_means.append({'dataset': ds, 'n_questions': len(reference), 'seeds': 3,
                             'seed_averaged_accuracy': float(np.mean(means)),
                             'delta_vs_entropy_2.0': float(np.mean(delta)),
                             'delta_ci_low': ci[0], 'delta_ci_high': ci[1]})
    # Family-wise correction across all displayed exact paired tests, both datasets.
    for row, pval in zip(paired, holm([r['mcnemar_exact_p'] for r in paired])):
        row['holm_p_all_comparisons'] = pval
    for row in summary:
        peers = [s for s in summary if s['dataset'] == row['dataset']]
        base = next(s for s in peers if s['policy'] == 'always_large')
        target = next(s for s in peers if s['policy'] == 'entropy_2.0')
        row['call_time_ratio_vs_always_large'] = row['mean_call_wall_s'] / base['mean_call_wall_s']
        row['call_time_mismatch_vs_entropy_2.0'] = row['mean_call_wall_s'] / target['mean_call_wall_s'] - 1
        # This is a parameter-billion-times-token proxy, not measured FLOPs or dollars.
        row['latency_savings_vs_always_large'] = 1 - row['mean_call_wall_s'] / base['mean_call_wall_s']
        row['episode_latency_savings_vs_always_large'] = 1 - row['mean_episode_wall_s'] / base['mean_episode_wall_s']
        row['token_savings_vs_always_large'] = 1 - row['mean_total_tokens'] / base['mean_total_tokens']
        row['compute_proxy_savings_vs_always_large'] = 1 - row['mean_parameter_b_tokens'] / base['mean_parameter_b_tokens']
        row['measured_cost_per_question_seconds'] = row['mean_call_wall_s']
        row['compute_proxy_per_question'] = row['mean_parameter_b_tokens']
        for metric in ['mean_call_wall_s', 'mean_llm_call_ratio']:
            row['non_dominated_' + metric] = not any(
                p[metric] <= row[metric] and p['success'] >= row['success'] and
                (p[metric] < row[metric] or p['success'] > row['success']) for p in peers)
    write_csv(destination / 'summary.csv', summary)
    write_csv(destination / 'paired_tests.csv', paired)
    write_csv(destination / 'random_seed_average.csv', random_means)
    atomic_json(destination / 'provenance.json', {'manifest': manifest, 'router': router, 'calibration': cal,
                 'complete_episodes': len(flat), 'mock': manifest['config']['mock'],
                 'ci_note': 'Wilson marginal accuracy CIs; paired question bootstrap differences. '
                            'Intervals are descriptive and not multiplicity-adjusted. '
                            'Holm adjusts all exact McNemar tests across both datasets. '
                            'No equivalence claim follows from non-significance.',
                 'cost_note': 'Per-call synchronized wall time includes tokenization, generation, '
                              'feature extraction and decoding. CUDA event elapsed time covers generate(), '
                              'not GPU active cycles/FLOPs/energy. Setup and interrupted incomplete '
                              'episodes are excluded from completed-episode totals.'})
    from .statistics_artifacts import render_artifacts
    render_artifacts(destination, summary, manifest['config']['mock'])
    print(f'Complete report: {destination} ({len(flat)} episodes; MOCK={manifest["config"]["mock"]})', flush=True)
