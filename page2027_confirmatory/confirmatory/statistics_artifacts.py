"""CPU-only publication tables and Wilson error-bar plots."""
from pathlib import Path
import importlib.metadata
from .core import atomic_json


def render_artifacts(destination, summary, mock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .report import write_csv
    destination = Path(destination)
    families = [('always', 'Direct baselines', '#222222', 'x'),
                ('entropy', 'Entropy', '#c63138', 's'),
                ('step', 'Step gate', '#ba7018', '^'),
                ('cascade', 'Cascade', '#228c45', 'D'),
                ('learned', 'Learned', '#8463a6', 'P'),
                ('confidence', 'Confidence', '#2675a6', 'o'),
                ('query', 'Query level', '#477c78', '*'),
                ('random', 'Random seeds', '#777777', '.')]
    for ds, title in [('hotpotqa', 'HotpotQA'), ('musique', 'MuSiQue')]:
        rows = [r for r in summary if r['dataset'] == ds]
        for metric, label, stem in [
            ('call_time_ratio_vs_always_large', 'Mean measured call time / always-8B', 'measured_cost'),
            ('mean_llm_call_ratio', 'Mean executed 8B-step fraction', 'tier_usage')]:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for prefix, name, color, marker in families:
                points = sorted([r for r in rows if r['policy'].startswith(prefix)], key=lambda r: r[metric])
                if not points:
                    continue
                ax.errorbar([r[metric] for r in points], [r['success'] for r in points],
                            yerr=[[max(0., r['success']-r['success_ci_low']) for r in points],
                                  [max(0., r['success_ci_high']-r['success']) for r in points]],
                            fmt=marker, color=color, capsize=2, markersize=6,
                            elinewidth=.9, label=name)
            ax.set(xlabel=label, ylabel='Exact-match success (Wilson 95% CI)',
                   title=('MOCK - NOT PAPER RESULTS: ' if mock else '') + title)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=.2)
            fig.tight_layout()
            for extension in ['png', 'pdf']:
                fig.savefig(destination / f'{ds}_{stem}_with_ci.{extension}', bbox_inches='tight', dpi=200)
            plt.close(fig)
    table = []
    for r in summary:
        table.append({'dataset': r['dataset'], 'policy': r['policy'], 'questions': r['n'],
                      'exact_match': f'{r["success"]:.3f}',
                      'wilson_95_ci': f'[{r["success_ci_low"]:.3f}, {r["success_ci_high"]:.3f}]',
                      'mean_8b_step_fraction': f'{r["mean_llm_call_ratio"]:.3f}',
                      'mean_call_seconds': f'{r["mean_call_wall_s"]:.3f}',
                      'latency_savings_vs_8b': f'{100*r["latency_savings_vs_always_large"]:.1f}%',
                      'token_savings_vs_8b': f'{100*r["token_savings_vs_always_large"]:.1f}%',
                      'compute_proxy_savings_vs_8b': f'{100*r["compute_proxy_savings_vs_always_large"]:.1f}%',
                      'time_ratio_vs_8b': f'{r["call_time_ratio_vs_always_large"]:.3f}'})
    write_csv(destination / 'confirmatory_results_table.csv', table)
    tex = ['% Generated from complete question-level results; do not edit numerical cells.',
           '% MOCK - NOT PAPER RESULTS' if mock else '% Wilson 95% confidence intervals.',
           '\\begin{table*}[t]', '\\centering', '\\small',
           '\\caption{' + ('MOCK - NOT PAPER RESULTS. ' if mock else '') +
           'Confirmatory exact-match accuracy and measured generation-call time. '
           'Intervals are Wilson 95\\% intervals. Time includes candidate and routing calls.}',
           '\\label{tab:confirmatory}', '\\begin{tabular}{llrrrr}', '\\toprule',
           'Dataset & Policy & EM [95\\% CI] & 8B-step fraction & Call time (s) & Time / 8B \\\\',
           '\\midrule']
    for r in table:
        escaped = r['policy'].replace('_', '\\_')
        tex.append(f'{r["dataset"]} & {escaped} & {r["exact_match"]} {r["wilson_95_ci"]} & '
                   f'{r["mean_8b_step_fraction"]} & {r["mean_call_seconds"]} & '
                   f'{r["time_ratio_vs_8b"]} \\\\')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    (destination / 'confirmatory_results_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    atomic_json(destination / 'analysis_versions.json',
                {name: importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib']})
