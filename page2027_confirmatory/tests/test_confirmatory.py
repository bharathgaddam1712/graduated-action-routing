import argparse
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from confirmatory.backend import MockBackend
from confirmatory.core import (POLICIES, Store, parse_action, parse_confidence,
                               router_probability, run_episode, selected_samples)
from confirmatory.data import mock_splits, normalize_sample, validate_splits
from confirmatory.report import holm, paired_bootstrap, report, wilson
from confirmatory import workflow

CFG = {'seed': 2027, 'train_n': 12, 'calibration_n': 6, 'eval_n': 3,
       'max_steps': 8, 'max_new_tokens': 256, 'max_input_tokens': 2048,
       'observation_chars': 500}


class CoreTests(unittest.TestCase):
    def sample(self):
        return mock_splits(CFG)['evaluation'][0]

    def test_action_input_is_query_not_thought(self):
        self.assertEqual(parse_action('Thought: wrong query\nAction: Search[correct query]'),
                         ('search', 'correct query', False))
        self.assertEqual(parse_action('Thought: x\nAction: Finish[answer]'), ('finish','answer',False))
        self.assertEqual(parse_action('Action: Search\nAction Input: specific'), ('search','specific',False))
        self.assertEqual(parse_action('unstructured')[0], 'invalid')

    def test_confidence(self):
        self.assertEqual(parse_confidence('Confidence: 90'), .9)
        self.assertIsNone(parse_confidence('42 tokens but no confidence'))

    def test_source_schemas(self):
        common = {'id': 'x', 'question': 'Q?', 'answer': 'A'}
        hp = normalize_sample('hotpotqa', {**common, 'context': {'title': ['T'], 'sentences': [['A']]}})
        self.assertEqual(hp['passages'], ['T: A'])
        mq = normalize_sample('musique', {**common, 'paragraphs': [{'title':'T','paragraph_text':'A'}]})
        self.assertEqual(mq['passages'], ['T: A'])

    def test_splits_reject_leakage(self):
        splits = mock_splits(CFG)
        validate_splits(splits, CFG)
        splits['evaluation'][0]['question'] = splits['train'][0]['question']
        with self.assertRaises(ValueError):
            validate_splits(splits, CFG)

    def test_shard_partition(self):
        samples = list(range(101))
        a, b = selected_samples(samples,0,2), selected_samples(samples,1,2)
        self.assertEqual(set(a+b), set(samples))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(POLICIES), 19)

    def test_direct_baseline_has_no_discarded_generation(self):
        row = run_episode(self.sample(), 'always_large', MockBackend(CFG), CFG)
        self.assertEqual(len(row['calls']), row['n_steps'])
        self.assertTrue(all(c['tier'] == 'large' for c in row['calls']))

    def test_cascade_counts_all_tiers(self):
        row = run_episode(self.sample(), 'cascade_-1_-1', MockBackend(CFG), CFG)
        self.assertEqual(len(row['calls']), 3 * row['n_steps'])
        self.assertEqual(row['llm_call_ratio'], 1)
        self.assertTrue(all(c['purpose'] == 'candidate' for c in row['calls'] if c['tier'] != 'large'))

    def test_confidence_overhead_is_logged(self):
        row = run_episode(self.sample(), 'confidence_0.9', MockBackend(CFG), CFG)
        self.assertEqual(len(row['calls']), 3 * row['n_steps'])
        self.assertEqual(sum(c['purpose'] == 'confidence' for c in row['calls']), row['n_steps'])

    def test_query_routes_once(self):
        row = run_episode(self.sample(), 'query', MockBackend(CFG), CFG, parameter=-1)
        self.assertEqual(len(row['calls']), row['n_steps'] + 1)
        self.assertEqual(sum(c['tier'] == 'small' for c in row['calls']), 1)

    def test_random_reproducibility_and_endpoints(self):
        a = run_episode(self.sample(), 'random_17', MockBackend(CFG), CFG, parameter=.5)
        b = run_episode(self.sample(), 'random_17', MockBackend(CFG), CFG, parameter=.5)
        self.assertEqual(a['steps'], b['steps'])
        for p in [0,1]:
            r = run_episode(self.sample(), 'random_17', MockBackend(CFG), CFG, parameter=p)
            self.assertEqual(r['llm_call_ratio'], p)

    def test_invalid_output_cannot_be_scored_from_observation(self):
        class Invalid(MockBackend):
            def generate(self, *a, **kw):
                out = super().generate(*a, **kw)
                out['response'] = 'malformed'
                return out
        row = run_episode(self.sample(), 'always_small', Invalid(CFG), CFG)
        self.assertEqual(row['predicted'], '')
        self.assertFalse(row['success'])
        self.assertEqual(row['stop_reason'], 'max_steps')

    def test_store_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.sqlite'
            sample = self.sample()
            row = run_episode(sample, 'always_small', MockBackend(CFG), CFG)
            store = Store(path, {'v':1})
            store.put(row)
            with self.assertRaises(sqlite3.IntegrityError):
                store.put(row)
            store.close()
            store = Store(path, {'v':1})
            self.assertEqual(store.get('always_small',sample)['steps'], row['steps'])
            store.close()
            with self.assertRaises(RuntimeError):
                Store(path, {'v':2})

    def test_statistics(self):
        lo, hi = wilson(0,100)
        self.assertAlmostEqual(lo,0)
        self.assertGreater(hi,0)
        self.assertEqual(paired_bootstrap([0]*10), [0,0])
        self.assertEqual(holm([.01,.04,.03]), [.03,.06,.06])

    def test_layout_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow.freeze_layout(tmp, 2)
            workflow.freeze_layout(tmp, 2)
            with self.assertRaises(RuntimeError):
                workflow.freeze_layout(tmp, 1)


class IntegrationTests(unittest.TestCase):
    def test_complete_mock_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(out=tmp, seed=2027, train_n=12, calibration_n=6,
                                      eval_n=3, mock=True, shard=0, num_shards=2, limit=1,
                                      stage='prepare')
            workflow.prepare(args)
            workflow.fit(args)
            workflow.calibrate(args)
            args.stage = 'run'
            workflow.run(args)
            with self.assertRaises(RuntimeError):
                report(args)
            args.limit = 0
            workflow.run(args)
            args.shard = 1
            workflow.run(args)
            workflow.run(args)  # resumed run cannot duplicate rows
            report(args)
            provenance = json.loads((Path(tmp)/'report/provenance.json').read_text())
            self.assertEqual(provenance['complete_episodes'], 19*6)
            self.assertTrue(provenance['mock'])
            self.assertTrue((Path(tmp)/'report/calls.csv').stat().st_size > 0)
            self.assertTrue((Path(tmp)/'report/confirmatory_results_table.tex').stat().st_size > 0)
            self.assertTrue((Path(tmp)/'report/hotpotqa_measured_cost_with_ci.png').stat().st_size > 0)
            self.assertEqual(len(list(__import__('csv').DictReader(
                (Path(tmp)/'report/summary.csv').read_text().splitlines()))), 38)


if __name__ == '__main__':
    unittest.main()
