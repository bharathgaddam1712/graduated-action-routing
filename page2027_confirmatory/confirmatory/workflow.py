"""Frozen experiment stages. Importing this module does not load CUDA."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from .core import (FEATURES, MODELS, POLICIES, Store, atomic_json, digest,
                   read_json, router_probability, run_episode, selected_samples)
from .data import SOURCES, build_splits, mock_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def code_hash():
    files = sorted(list((ROOT / 'confirmatory').glob('*.py')) +
                   list((ROOT / 'scripts').glob('*.py')) + [ROOT / 'run_server.py'])
    return digest({str(p.relative_to(ROOT)): p.read_text(encoding='utf-8') for p in files})


def versions(mock):
    names = ['numpy', 'scikit-learn', 'scipy', 'rank-bm25']
    if not mock:
        names += ['torch', 'transformers', 'huggingface-hub', 'pyarrow', 'accelerate']
    return {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {'seed': args.seed, 'train_n': args.train_n, 'calibration_n': args.calibration_n,
           'eval_n': args.eval_n, 'max_steps': 8, 'max_input_tokens': 2048,
           'max_new_tokens': 256, 'observation_chars': 500, 'mock': args.mock,
           'policies': POLICIES, 'code_hash': code_hash(), 'versions': versions(args.mock),
           'sources': SOURCES, 'protocol': 'corrected_parser_fp32_features_v1'}
    for key in ['train_n', 'calibration_n', 'eval_n']:
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    manifest = out / 'manifest.json'
    if manifest.exists():
        previous = read_json(manifest)
        if previous['config'] != cfg:
            raise RuntimeError('Prepared configuration or code changed. Use a new --out.')
        load_run(out)
        print('Prepared inputs already verified.', flush=True)
        return
    splits = mock_splits(cfg) if args.mock else build_splits(cfg)
    snapshots, revisions = {}, {}
    if not args.mock:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        for tier, (repo, _) in MODELS.items():
            sha = api.model_info(repo).sha
            revisions[tier] = {'repo': repo, 'revision': sha}
            snapshots[tier] = snapshot_download(repo, revision=sha,
                allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken'])
    atomic_json(out / 'splits.json', splits)
    atomic_json(manifest, {'config': cfg, 'data_hash': digest(splits),
                           'snapshots': snapshots, 'model_revisions': revisions})
    print(f'Prepared {sum(map(len, splits.values()))} unique questions. MOCK={args.mock}', flush=True)


def load_run(out, strict_runtime=True):
    out = Path(out)
    manifest, splits = read_json(out / 'manifest.json'), read_json(out / 'splits.json')
    if digest(splits) != manifest['data_hash']:
        raise RuntimeError('Input data changed after preparation')
    cfg = manifest['config']
    if code_hash() != cfg['code_hash']:
        raise RuntimeError('Code changed; use the original bundle with these results')
    if strict_runtime and versions(cfg['mock']) != cfg['versions']:
        raise RuntimeError('Code or dependency versions changed; use a new output directory')
    validate_splits(splits, cfg)
    return manifest, splits


def backend_for(manifest):
    from .backend import HFBackend, MockBackend
    cls = MockBackend if manifest['config']['mock'] else HFBackend
    return cls(manifest['config'], manifest['snapshots'])


def execute(store, samples, policy, backend, cfg, router=None, parameter=None, store_key=None):
    rows = []
    store_key = store_key or policy
    for i, sample in enumerate(samples):
        row = store.get(store_key, sample)
        if row is None:
            row = run_episode(sample, policy, backend, cfg, router, parameter)
            row['policy'] = store_key
            store.put(row)
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            print(f'{store_key}: {i+1}/{len(samples)} completed (includes resumed)', flush=True)
    return rows


def fit(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    out = Path(args.out)
    manifest, splits = load_run(out)
    identity = {'manifest': digest(manifest), 'stage': 'fit'}
    artifact_path = out / 'router.json'
    if artifact_path.exists():
        if read_json(artifact_path)['identity'] != identity:
            raise RuntimeError('Router identity mismatch')
        print('Frozen router already exists.', flush=True)
        return
    store = Store(out / 'fit.sqlite', identity)
    try:
        backend = backend_for(manifest)
        atomic_json(out / 'fit_device.json', backend.metadata)
        cfg = manifest['config']
        train = execute(store, splits['train'], 'train_small', backend, cfg)
        cal = execute(store, splits['calibration'], 'train_small', backend, cfg)
        def arrays(rows):
            x, y, weights = [], [], []
            for row in rows:
                for step in row['steps']:
                    x.append([step['features'][f] for f in FEATURES])
                    y.append(int(not row['success']))
                    weights.append(1. / row['n_steps'])
            return np.array(x), np.array(y), np.array(weights)
        x, y, w = arrays(train)
        if len(np.unique(y)) < 2:
            raise RuntimeError('Training labels have one class. Do not invent AUC or continue; '
                               'inspect baseline outputs and increase a NEW run training split.')
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=1., max_iter=2000, class_weight='balanced', random_state=cfg['seed']))
        clf.fit(x, y, logisticregression__sample_weight=w)
        scaler, lr = clf.steps[0][1], clf.steps[1][1]
        if int(lr.n_iter_.max()) >= 2000:
            raise RuntimeError('Router did not converge')
        artifact = {'identity': identity, 'features': FEATURES,
                    'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                    'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
                    'train_episodes': len(train), 'train_steps': len(x),
                    'training_auc': float(roc_auc_score(y, clf.predict_proba(x)[:, 1])),
                    'episode_weighting': 'each episode contributes total sample weight 1'}
        xc, yc, _ = arrays(cal)
        artifact['calibration_auc'] = (float(roc_auc_score(yc, clf.predict_proba(xc)[:, 1]))
                                       if len(np.unique(yc)) == 2 else None)
        # Check JSON-only deployment math against sklearn before freezing it.
        deployed = [router_probability(artifact, dict(zip(FEATURES, v))) for v in xc]
        if not np.allclose(deployed, clf.predict_proba(xc)[:, 1], atol=1e-10):
            raise RuntimeError('Exported router does not match sklearn')
        atomic_json(artifact_path, artifact)
        print(f'Router frozen; calibration AUC={artifact["calibration_auc"]}', flush=True)
    finally:
        store.close()


def calibrated_inputs(out, manifest):
    router = read_json(Path(out) / 'router.json')
    calibration = read_json(Path(out) / 'calibration.json')
    if router['identity'] != {'manifest': digest(manifest), 'stage': 'fit'}:
        raise RuntimeError('Router provenance mismatch')
    if calibration['identity'] != {'manifest': digest(manifest), 'router': digest(router), 'stage': 'calibration'}:
        raise RuntimeError('Calibration provenance mismatch')
    return router, calibration


def calibrate(args):
    import numpy as np
    out = Path(args.out)
    manifest, splits = load_run(out)
    router = read_json(out / 'router.json')
    identity = {'manifest': digest(manifest), 'router': digest(router), 'stage': 'calibration'}
    if (out / 'calibration.json').exists():
        calibrated_inputs(out, manifest)
        print('Calibration already frozen.', flush=True)
        return
    store = Store(out / 'calibration.sqlite', identity)
    try:
        backend = backend_for(manifest)
        atomic_json(out / 'calibration_device.json', backend.metadata)
        cfg = manifest['config']
        result = {'identity': identity, 'metric': 'mean synchronized call_wall_s per episode',
                  'target_policy': 'entropy_2.0', 'datasets': {}}
        for ds in SOURCES:
            samples = [s for s in splits['calibration'] if s['dataset'] == ds]
            target_rows = execute(store, samples, 'entropy_2.0', backend, cfg)
            target = float(np.mean([r['call_wall_s'] for r in target_rows]))
            entropies = [r['calls'][0]['features']['first_token_entropy'] for r in target_rows]
            query_grid = [min(entropies) - 1.] + list(np.quantile(entropies, [.25, .5, .75])) + [max(entropies) + 1.]
            query_trials = []
            for t in sorted(set(float(x) for x in query_grid)):
                rows = execute(store, samples, 'query', backend, cfg, parameter=t,
                               store_key=f'cal_query_{t:.12g}')
                query_trials.append({'value': t, 'cost': float(np.mean([r['call_wall_s'] for r in rows]))})
            random_trials, lo, hi = [], 0., 1.
            # Endpoints and four bisection points; calibration randomness is
            # independent of the three evaluation seeds. No evaluation feedback.
            for trial in range(6):
                p = float(trial) if trial < 2 else (lo + hi) / 2
                rows = execute(store, samples, 'random_7001', backend, cfg, parameter=p,
                               store_key=f'cal_random_{p:.12g}')
                cost = float(np.mean([r['call_wall_s'] for r in rows]))
                random_trials.append({'value': p, 'cost': cost})
                if trial >= 2:
                    if cost < target:
                        lo = p
                    else:
                        hi = p
            choose = lambda trials: min(trials, key=lambda t: (abs(t['cost'] - target), t['value']))
            chosen_q, chosen_r = choose(query_trials), choose(random_trials)
            result['datasets'][ds] = {
                'target_call_wall_s': target, 'query_threshold': chosen_q['value'],
                'random_probability': chosen_r['value'],
                'query_relative_mismatch': chosen_q['cost'] / target - 1,
                'random_relative_mismatch': chosen_r['cost'] / target - 1,
                'query_trials': query_trials, 'random_trials': random_trials}
            print(f'{ds} calibration: {result["datasets"][ds]}', flush=True)
        atomic_json(out / 'calibration.json', result)
    finally:
        store.close()


def run(args):
    out = Path(args.out)
    manifest, splits = load_run(out)
    router, cal = calibrated_inputs(out, manifest)
    smoke = args.stage == 'smoke'
    policies = POLICIES
    if not smoke:
        freeze_layout(out, args.num_shards)
    identity = {'manifest': digest(manifest), 'router': digest(router), 'calibration': digest(cal),
                'stage': 'smoke' if smoke else 'evaluation',
                'shard': args.shard, 'num_shards': args.num_shards,
                'partition': 'questions_within_dataset',
                'physical_device_selection': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    stem = 'smoke' if smoke else f'eval-shard-{args.shard}-of-{args.num_shards}'
    store = Store(out / (stem + '.sqlite'), identity)
    try:
        backend = backend_for(manifest)
        atomic_json(out / (stem + '-device.json'), backend.metadata)
        # Permute policy order deterministically to reduce a fixed family/time bias.
        import random
        policies = list(policies)
        random.Random(manifest['config']['seed'] + args.shard).shuffle(policies)
        for policy in policies:
            for ds in SOURCES:
                samples = [s for s in splits['calibration' if smoke else 'evaluation'] if s['dataset'] == ds]
                if smoke or args.limit:
                    samples = samples[:1 if smoke else args.limit]
                if not smoke:
                    samples = selected_samples(samples, args.shard, args.num_shards)
                params = cal['datasets'][ds]
                parameter = (params['query_threshold'] if policy == 'query' else
                             params['random_probability'] if policy.startswith('random') else None)
                execute(store, samples, policy, backend, manifest['config'], router, parameter)
    finally:
        store.close()


def freeze_layout(out, num_shards):
    # A short shared lock prevents new workers from reading a partially
    # initialized sibling database while all workers start simultaneously.
    with (Path(out) / 'layout.lock').open('a+') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        path = Path(out) / 'layout.json'
        layout = {'num_shards': num_shards, 'partition': 'questions_within_dataset'}
        if path.exists():
            if read_json(path) != layout:
                raise RuntimeError('Keep the same --gpus/shard count when resuming this output directory.')
        else:
            atomic_json(path, layout)


def estimate(args):
    """Project from completed evaluation episodes, not an invented GPU throughput."""
    import sqlite3
    out = Path(args.out)
    manifest, _ = load_run(out)
    grouped = {}
    for path in out.glob('eval-shard-*.sqlite'):
        db = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
        for (payload,) in db.execute('SELECT payload FROM episodes'):
            row = json.loads(payload)
            grouped.setdefault((row['dataset'], row['policy']), []).append(row['episode_wall_s'])
        db.close()
    if len(grouped) != len(POLICIES) * 2:
        raise RuntimeError('Run the pilot for every shard before estimating')
    projected = sum(sum(v) / len(v) * manifest['config']['eval_n'] for v in grouped.values())
    completed = sum(sum(v) for v in grouped.values())
    report = {'projected_evaluation_gpu_hours': projected / 3600,
              'remaining_evaluation_gpu_hours': max(0, projected - completed) / 3600,
              'completed_episodes': sum(map(len, grouped.values())),
              'warning': 'Projection only; small pilot, episode lengths and GPU contention affect it. '
                         'Excludes model loading and training/calibration already completed. '
                         'Two GPUs reduce wall time only if shards and contention are balanced.'}
    atomic_json(out / 'runtime_estimate.json', report)
    print(json.dumps(report, indent=2), flush=True)
