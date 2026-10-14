"""Standalone entry point. Run --help for the stage interface."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'calibrate', 'smoke', 'run', 'estimate', 'report'], required=True)
    parser.add_argument('--out', default='runs/confirmatory')
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--train-n', type=int, default=300, help='Per dataset; from official training split')
    parser.add_argument('--calibration-n', type=int, default=150, help='Per dataset; disjoint validation subset')
    parser.add_argument('--eval-n', type=int, default=1000, help='Per dataset; disjoint validation subset')
    parser.add_argument('--mock', action='store_true', help='Synthetic backend: NEVER scientific results')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='Pilot: first N evaluation questions per dataset; 0=all')
    args = parser.parse_args()
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error('Invalid limit or shard configuration')
    from confirmatory import workflow
    if args.stage == 'report':
        from confirmatory.report import report
        report(args)
    elif args.stage in ['run', 'smoke']:
        workflow.run(args)
    else:
        getattr(workflow, args.stage)(args)


if __name__ == '__main__':
    main()
