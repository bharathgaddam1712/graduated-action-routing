"""CPU-only Wilson intervals, paired bootstrap/McNemar tests, figures and table."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/confirmatory')
    args = parser.parse_args()
    from confirmatory.report import report
    report(args)


if __name__ == '__main__':
    main()
