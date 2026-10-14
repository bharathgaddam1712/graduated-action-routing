"""Revision-pinned source files; no remote dataset execution."""
import json
import random
from .core import atomic_json, digest, question_key

SOURCES = {
    'hotpotqa': {'repo': 'hotpotqa/hotpot_qa',
                'revision': '1908d6afbbead072334abe2965f91bd2709910ab',
                'train': ['distractor/train-00000-of-00002.parquet',
                          'distractor/train-00001-of-00002.parquet'],
                'validation': ['distractor/validation-00000-of-00001.parquet'],
                'exclude_first': 300},
    'musique': {'repo': 'dgslibisey/musique',
               'revision': 'c8f4f8c9465fb69d31a8eae894c3fd509c4ca321',
               'train': ['musique_ans_v1.0_train.jsonl'],
               'validation': ['musique_ans_v1.0_dev.jsonl'],
               'exclude_first': 200},
}


def source_rows(ds, split):
    from huggingface_hub import hf_hub_download
    spec = SOURCES[ds]
    for filename in spec[split]:
        path = hf_hub_download(spec['repo'], filename, repo_type='dataset', revision=spec['revision'])
        if filename.endswith('.parquet'):
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                yield from batch.to_pylist()
        else:
            with open(path, encoding='utf-8') as src:
                for line in src:
                    if line.strip():
                        yield json.loads(line)


def normalize_sample(ds, row):
    if ds == 'hotpotqa':
        context = row['context']
        if isinstance(context, dict):
            pairs = zip(context['title'], context['sentences'])
        else:
            pairs = context
        passages = [title + ': ' + ' '.join(sentences) for title, sentences in pairs]
    else:
        passages = [p['title'] + ': ' + p['paragraph_text'] for p in row['paragraphs']]
    item = {'dataset': ds, 'id': str(row['id']), 'question': row['question'],
            'answer': row['answer'], 'aliases': row.get('answer_aliases', []), 'passages': passages}
    if not item['question'].strip() or not item['answer'].strip() or not passages:
        raise ValueError(f'Empty required field: {ds}/{item["id"]}')
    return item


def build_splits(config):
    splits = {s: [] for s in ['train', 'calibration', 'evaluation']}
    # Remove pilot questions from the original loader-order prefixes. Also
    # exclude identical question text across datasets and all new splits.
    seen_text, seen_ids = set(), set()
    dev_by_ds = {}
    for ds, spec in SOURCES.items():
        dev = [normalize_sample(ds, r) for r in source_rows(ds, 'validation')]
        for row in dev[:spec['exclude_first']]:
            seen_text.add(question_key(row['question']))
            seen_ids.add((ds, row['id']))
        dev_by_ds[ds] = dev[spec['exclude_first']:]
    for ds in SOURCES:
        # Reservoir sampling keeps memory bounded while sampling unique training
        # questions uniformly rather than materializing the full passage corpus.
        rng = random.Random(config['seed'])
        rows, local_seen, count = [], set(), 0
        for raw in source_rows(ds, 'train'):
            row = normalize_sample(ds, raw)
            key, qid = question_key(row['question']), (ds, row['id'])
            if key in seen_text or qid in seen_ids or key in local_seen:
                continue
            local_seen.add(key)
            count += 1
            if len(rows) < config['train_n']:
                rows.append(row)
            else:
                j = rng.randrange(count)
                if j < config['train_n']:
                    rows[j] = row
        rng.shuffle(rows)
        take_unique(rows, config['train_n'], splits['train'], seen_text, seen_ids)
    for ds in SOURCES:
        rows = dev_by_ds[ds]
        random.Random(config['seed'] + 1).shuffle(rows)
        available = [r for r in rows if question_key(r['question']) not in seen_text
                     and (ds, r['id']) not in seen_ids]
        used = take_unique(available, config['calibration_n'], splits['calibration'], seen_text, seen_ids)
        take_unique(available[used:], config['eval_n'], splits['evaluation'], seen_text, seen_ids)
    validate_splits(splits, config)
    return splits


def take_unique(rows, n, destination, seen_text, seen_ids):
    count = 0
    for idx, row in enumerate(rows):
        key, qid = question_key(row['question']), (row['dataset'], row['id'])
        if key in seen_text or qid in seen_ids:
            continue
        destination.append(row)
        seen_text.add(key)
        seen_ids.add(qid)
        count += 1
        if count == n:
            return idx + 1
    raise ValueError(f'Only {count} unique examples available, need {n}')


def validate_splits(splits, config):
    texts, ids = set(), set()
    for split, n in [('train', config['train_n']), ('calibration', config['calibration_n']),
                     ('evaluation', config['eval_n'])]:
        for ds in SOURCES:
            if sum(r['dataset'] == ds for r in splits[split]) != n:
                raise ValueError(f'Incorrect split size: {split}/{ds}')
        for row in splits[split]:
            key, qid = question_key(row['question']), (row['dataset'], row['id'])
            if key in texts or qid in ids:
                raise ValueError(f'Overlapping question: {qid}')
            texts.add(key)
            ids.add(qid)


def mock_splits(config):
    out = {}
    for split, n in [('train', config['train_n']), ('calibration', config['calibration_n']),
                     ('evaluation', config['eval_n'])]:
        out[split] = [{'dataset': ds, 'id': f'{ds}-{split}-{i}',
                       'question': f'{ds} {split} synthetic question {i}?',
                       'answer': f'answer {i}', 'aliases': [],
                       'passages': [f'evidence answer {i}', 'another unrelated passage']}
                      for ds in SOURCES for i in range(n)]
    validate_splits(out, config)
    return out
