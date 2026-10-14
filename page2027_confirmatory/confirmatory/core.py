"""CPU-testable routing, environment, persistence, and scoring."""
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
from pathlib import Path

MODELS = {
    'small': ('Qwen/Qwen2.5-1.5B-Instruct', 1.5),
    'mid': ('Qwen/Qwen2.5-3B-Instruct', 3.0),
    'large': ('meta-llama/Meta-Llama-3.1-8B-Instruct', 8.0),
}
FEATURES = ['first_token_entropy', 'mean_entropy', 'top1_prob',
            'margin_top1_top2', 'mean_top5_prob']
POLICIES = ['always_small', 'always_mid', 'always_large',
            'entropy_1.0', 'entropy_1.5', 'entropy_2.0', 'entropy_2.5',
            'step_1', 'step_2', 'cascade_0.5_2.0', 'cascade_1.0_2.5',
            'learned_0.3', 'learned_0.4', 'learned_0.5', 'confidence_0.9',
            'query', 'random_17', 'random_29', 'random_43']
SYSTEM = '''You are a reasoning agent that solves multi-hop questions by alternating between thinking and using tools.
Available tools:
- Search[query]: Search for relevant information. Returns the most relevant passages.
- Finish[answer]: Provide your final answer and stop.
Format each step exactly as:
Thought: <your reasoning>
Action: Search[<query>] or Finish[<answer>]
Do not include anything else after an Action line.'''


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as out:
        json.dump(obj, out, indent=2, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def normalize(text):
    text = re.sub(r'\b(a|an|the)\b', ' ', text.lower().strip())
    return ' '.join(re.sub(r'[^\w\s]', '', text).split())


def question_key(text):
    return ' '.join(text.lower().split())


def parse_action(text):
    # Parse bracketed action input, not the Thought text as in the pilot.
    m = re.search(r'^\s*Action\s*:\s*(Search|Finish)\s*\[(.*?)\]',
                  text, re.I | re.M | re.S)
    if m:
        return m.group(1).lower(), m.group(2).strip(), False
    m = re.search(r'^\s*Action\s*:\s*(Search|Finish)\s*$', text, re.I | re.M)
    v = re.search(r'^\s*Action Input\s*:\s*(.+)$', text, re.I | re.M)
    if m and v:
        return m.group(1).lower(), v.group(1).strip(), False
    m = re.search(r'^\s*Final Answer\s*:\s*(.+)$', text, re.I | re.M)
    if m:
        return 'finish', m.group(1).strip().rstrip('.'), False
    return 'invalid', '', True


def parse_confidence(text):
    m = re.search(r'Confidence\s*:\s*(\d+(?:\.\d+)?)', text, re.I)
    if not m:
        return None
    return max(0., min(100., float(m.group(1)))) / 100.


def router_probability(artifact, features):
    x = [(features[f] - m) / s for f, m, s in
         zip(FEATURES, artifact['mean'], artifact['scale'])]
    z = artifact['intercept'] + sum(a * b for a, b in zip(x, artifact['coef']))
    return 1. / (1. + math.exp(-max(-700., min(700., z))))


def selected_samples(samples, shard, num_shards):
    if not 0 <= shard < num_shards:
        raise ValueError('shard must be in [0, num_shards)')
    return samples[shard::num_shards]


class Store:
    """One writer per shard; each complete episode and its calls commit together."""
    def __init__(self, path, identity):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_suffix('.lock').open('a+')
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.lock.close()
                raise RuntimeError(f'Another worker owns {path}') from exc
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS episodes '
                        '(policy TEXT, dataset TEXT, question_id TEXT, payload TEXT, '
                        'PRIMARY KEY(policy,dataset,question_id))')
        previous = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        encoded = json.dumps(identity, sort_keys=True)
        if previous and previous[0] != encoded:
            self.close()
            raise RuntimeError('Resume identity changed. Use a new --out directory.')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (encoded,))
        self.db.commit()

    def get(self, policy, sample):
        row = self.db.execute('SELECT payload FROM episodes WHERE policy=? AND dataset=? AND question_id=?',
                              (policy, sample['dataset'], sample['id'])).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, row):
        with self.db:
            self.db.execute('INSERT INTO episodes VALUES (?,?,?,?)',
                            (row['policy'], row['dataset'], row['question_id'],
                             json.dumps(row, allow_nan=False)))

    def rows(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM episodes')]

    def close(self):
        self.db.close()
        self.lock.close()


def run_episode(sample, policy, backend, config, router=None, parameter=None):
    from rank_bm25 import BM25Okapi
    began = time.perf_counter()
    backend.begin_episode(sample)
    passages = sample['passages']
    index = BM25Okapi([re.findall(r'\w+', p.lower()) for p in passages])
    scratch = ''
    calls, steps = [], []
    prediction, stop = '', 'max_steps'
    fixed_tier = None
    rng = random.Random(int(digest([config['seed'], policy, sample['dataset'], sample['id']])[:16], 16))

    def call(tier, messages, purpose, feature_mode='none', limit=None):
        out = backend.generate(tier, messages, feature_mode, limit or config['max_new_tokens'])
        out.update(tier=tier, purpose=purpose, step_idx=step, call_idx=len(calls))
        calls.append(out)
        return out

    for step in range(config['max_steps']):
        user = f"Question: {sample['question']}"
        if scratch:
            user += '\n\n' + scratch.strip()
        prompt = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
        feature_mode = 'all' if policy == 'train_small' or policy.startswith('learned') else 'first'
        if policy.startswith('always_') or policy == 'train_small':
            tier = policy.split('_', 1)[1]
            out = call(tier, prompt, 'executed', 'all' if policy == 'train_small' else 'none')
        elif policy == 'query' and fixed_tier is not None:
            tier = fixed_tier
            out = call(tier, prompt, 'executed')
        else:
            out = call('small', prompt, 'candidate', feature_mode)
            tier = 'small'
            if policy.startswith('entropy_'):
                escalate = out['features']['first_token_entropy'] > float(policy.split('_')[1])
            elif policy.startswith('step_'):
                escalate = step >= int(policy.split('_')[1])
            elif policy.startswith('learned_'):
                if router is None:
                    raise ValueError('Learned policy requires a frozen router')
                escalate = router_probability(router, out['features']) >= float(policy.split('_')[1])
            elif policy.startswith('confidence_'):
                conf_prompt = [{'role': 'system', 'content': 'You are a calibrated confidence estimator.'},
                               {'role': 'user', 'content': user + '\nYou proposed:\n' + out['response'] +
                                '\nRate confidence that this action is correct from 0 to 100. '
                                'Respond with exactly: Confidence: <number>'}]
                conf = parse_confidence(call('small', conf_prompt, 'confidence', limit=16)['response'])
                escalate = conf is None or conf < .9
            elif policy == 'query':
                if parameter is None:
                    raise ValueError('Query policy requires calibrated threshold')
                escalate = out['features']['first_token_entropy'] > parameter
                fixed_tier = 'large' if escalate else 'small'
            elif policy.startswith('random_'):
                if parameter is None or not 0 <= parameter <= 1:
                    raise ValueError('Random policy requires probability in [0,1]')
                escalate = rng.random() < parameter
            elif policy.startswith('cascade_'):
                t1, t2 = map(float, policy.split('_')[1:])
                escalate = False
                if out['features']['first_token_entropy'] > t1:
                    out = call('mid', prompt, 'candidate', 'first')
                    tier = 'mid'
                    escalate = out['features']['first_token_entropy'] > t2
            else:
                raise ValueError(f'Unknown policy {policy}')
            if escalate:
                tier = 'large'
                out = call(tier, prompt, 'executed')
            out['purpose'] = 'executed'

        action, value, invalid = parse_action(out['response'])
        steps.append({'step_idx': step, 'tier': tier, 'action': action,
                      'action_input': value, 'invalid': invalid,
                      'features': out['features']})
        if action == 'finish':
            prediction, stop = value, 'finish'
            break
        if invalid:
            observation = 'Invalid action. Use Action: Search[query] or Action: Finish[answer].'
        else:
            scores = index.get_scores(re.findall(r'\w+', value.lower()))
            best = sorted(range(len(passages)), key=lambda j: (-float(scores[j]), j))[:2]
            observation = '\n\n'.join(passages[j] for j in best)[:config['observation_chars']]
        scratch += '\n' + out['response'].strip() + '\nObservation: ' + observation + '\n'

    row = {'policy': policy, 'dataset': sample['dataset'], 'question_id': sample['id'],
           'success': normalize(prediction) == normalize(sample['answer']),
           'alias_success': any(normalize(prediction) == normalize(a)
                                for a in [sample['answer']] + sample.get('aliases', [])),
           'predicted': prediction, 'gold_answer': sample['answer'], 'stop_reason': stop,
           'n_steps': len(steps), 'steps': steps, 'calls': calls,
           'llm_call_ratio': sum(s['tier'] == 'large' for s in steps) / len(steps),
           'episode_wall_s': time.perf_counter() - began,
           'peak_allocated_bytes': backend.peak_memory(),
           'call_wall_s': sum(c['wall_s'] for c in calls),
           'cuda_elapsed_s': sum(c['cuda_s'] for c in calls),
           'input_tokens': sum(c['input_tokens'] for c in calls),
           'output_tokens': sum(c['output_tokens'] for c in calls),
           'parameter_b_tokens': sum(MODELS[c['tier']][1] *
                                     (c['input_tokens'] + c['output_tokens']) for c in calls)}
    return row
