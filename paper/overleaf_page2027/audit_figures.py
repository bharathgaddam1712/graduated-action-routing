"""Rebuild paper plots and audit results from the existing CSVs; no GPU calls."""
from pathlib import Path
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / 'results' / 'exploratory'
episodes = pd.concat([pd.read_csv(RESULTS / (n + '.csv')) for n in
                     ['baseline_episodes', 'routing_episodes', 'learned_router_episodes']])
assert not episodes.duplicated(['dataset', 'policy', 'question_id']).any()
summary = episodes.groupby(['dataset', 'policy']).agg(
    success=('success', 'mean'), ratio=('llm_call_ratio', 'mean'),
    n=('success', 'size')).reset_index()
assert len(episodes) == 20500 and len(summary) == 82
styles = [('conf_gate', 'Confidence', '#2675a6', 'o'),
          ('entropy_gate', 'Entropy', '#c63138', 's'),
          ('step_gate', 'Step index', '#ba7018', '^'),
          ('cascade', 'Three-tier cascade', '#228c45', 'D'),
          ('learned', 'Learned router', '#8463a6', 'P')]
plt.rcParams.update({'font.size': 10, 'figure.dpi': 180})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / name, bbox_inches='tight')
    plt.close(fig)

def curves(ds, families, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sub = summary[summary.dataset == ds]
    for prefix, label, color, marker in styles:
        if prefix not in families:
            continue
        pts = sub[sub.policy.str.startswith(prefix)].sort_values('ratio')
        ax.plot(pts.ratio, pts.success, marker=marker, color=color,
                label=label, linewidth=1.2, markersize=5)
    for policy, label, marker in [('always_slm_1b', 'Always 1.5B', 'x'),
                                 ('always_slm_3b', 'Always 3B', '+'),
                                 ('always_llm', 'Always 8B', '*')]:
        row = sub[sub.policy == policy].iloc[0]
        ax.scatter(row.ratio, row.success, marker=marker, color='black',
                   s=95, zorder=5, label=label)
    ax.set(xlabel='Mean per-episode 8B-step fraction',
           ylabel='Exact-match success rate', title=title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=.2)
    save(fig, filename)

curves('hotpotqa', [s[0] for s in styles], 'HotpotQA policy sweep', 'fig1_pareto_frontier.png')
curves('hotpotqa', ['entropy_gate', 'cascade'], 'HotpotQA: two-tier vs. three-tier', 'fig2_3level_vs_2level.png')
curves('musique', [s[0] for s in styles], 'MuSiQue policy sweep', 'fig5_cross_task.png')
steps = pd.read_csv(RESULTS / 'baseline_steps.csv')
small = steps[steps.slm_name == 'Qwen2.5-1.5B']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, ds, title in zip(axes, ['hotpotqa', 'musique'], ['HotpotQA', 'MuSiQue']):
    sub = small[small.dataset == ds].groupby('step_idx').agg(
        entropy=('first_token_entropy', 'mean'), n=('question_id', 'size'))
    ax.bar(sub.index, sub.entropy, color='#2675a6', alpha=.7)
    twin = ax.twinx()
    twin.plot(sub.index, sub.n / sub.n.max(), color='#c63138', marker='o')
    twin.set(ylim=(0, 1.05), ylabel='Fraction of episodes still active')
    ax.set(title=title, xlabel='Step index', ylabel='Mean first-token entropy (nats)')
save(fig, 'fig3_step_escalation.png')
labels = (~steps.episode_success).astype(int)
aucs = [roc_auc_score(labels, steps.first_token_entropy),
        roc_auc_score(labels, 1 - steps.top1_prob)]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(['First-token entropy', '1 - top-1 probability'], aucs, color=['#2675a6', '#c63138'])
ax.axhline(.5, color='gray', linestyle='--')
ax.set(ylim=(0, 1), ylabel='AUC for episode failure', title='Pooled baseline signals (all three models)')
save(fig, 'fig4_signal_quality.png')
features = ['first_token_entropy', 'mean_entropy', 'top1_prob', 'margin_top1_top2', 'mean_top5_prob']
training_auc = []
for cols in [features, features + ['step_idx']]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500, class_weight='balanced'))
    y = (~small.episode_success).astype(int)
    clf.fit(small[cols], y)
    training_auc.append(roc_auc_score(y, clf.predict_proba(small[cols])[:, 1]))
for ds in ['hotpotqa', 'musique']:
    pts = summary[summary.dataset == ds]
    oracle = pts.loc[pts.policy == 'always_llm', 'success'].iloc[0]
    assert not ((pts.ratio < .6) & (pts.success >= .9 * oracle)).any()
    for _, p in pts[pts.policy.str.startswith('learned')].iterrows():
        assert ((pts.ratio <= p.ratio) & (pts.success >= p.success) &
                ((pts.ratio < p.ratio) | (pts.success > p.success))).any()
tex = (HERE / 'main.tex').read_text(encoding='utf-8')
bib = (HERE / 'references.bib').read_text(encoding='utf-8')
cited = {k for group in re.findall(r'\\cite\{([^}]+)\}', tex) for k in group.split(',')}
defined = set(re.findall(r'@\w+\{([^,]+),', bib))
assert cited == defined, (cited - defined, defined - cited)
print(json.dumps({'episodes': len(episodes), 'configurations': len(summary),
                  'baseline_steps': len(steps), 'training_steps': len(small),
                  'training_auc': training_auc, 'pooled_signal_auc': aucs,
                  'resolved_citations': len(cited), 'per_dataset_dominance': 'passed'}, indent=2))
