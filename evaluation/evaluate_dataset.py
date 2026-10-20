"""Evaluate every scene in a released dataset, then summarize."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from evaluation.protocols.datasets import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--data_root', required=True, type=Path, help='folder containing the prepared datasets')
    parser.add_argument('--output_root', required=True, type=Path, help='folder containing dataset/scene outputs')
    parser.add_argument('--iteration', type=int, help='default: 10000 for APTv2, 20000 for DAVIS/Artemis')
    parser.add_argument('--render', action='store_true', help='generate predictions from existing checkpoints before computing metrics')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    dataset = DATASETS[args.dataset]
    data_name, kind = dataset.data_name, dataset.kind
    scenes = dataset.scenes
    train_root = data_root / dataset.train_root
    iteration = args.iteration if args.iteration is not None else dataset.iteration
    if iteration <= 0:
        parser.error('--iteration must be positive')
    for scene in scenes:
        source = train_root / scene / 'cropped'
        model = output_root / data_name / scene
        if not source.is_dir():
            parser.error(f'Missing prepared sequence: {source}')
        required = model if args.render else model / f'eval_{iteration}'
        if not required.is_dir():
            parser.error(f'Missing {"model output" if args.render else "saved predictions"}: {required}')

    def run(command, cwd):
        print(f'[{cwd.name}] {shlex.join(command)}', flush=True)
        subprocess.run(command, cwd=cwd, check=True)

    for scene in scenes:
        model = str(output_root / data_name / scene)
        source = str(train_root / scene / 'cropped')
        common = ['-m', model, '-s', source, '--iteration', str(iteration)]
        if args.render:
            run([sys.executable, 'render_eval.py', *common], ROOT / kind)
        run([sys.executable, '-m', 'evaluation.evaluate', kind, *common], ROOT)
    run([sys.executable, '-m', 'evaluation.summarize', kind,
         '--output_root', str(output_root), '--data_name', data_name,
         '--iteration', str(iteration), '--scenes', *scenes], ROOT)


if __name__ == '__main__':
    main()
