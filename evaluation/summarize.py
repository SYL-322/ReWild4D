"""Aggregate saved per-scene metrics under the real or Artemis protocol."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    import runpy
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('kind', choices=['real', 'artemis'])
    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module('evaluation.summaries.' + args.kind, run_name='__main__')


if __name__ == '__main__':
    main()
