"""Compute metrics from saved predictions without importing the reconstruction model."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from evaluation.protocols.paths import default_gt_path, default_chamfer_root, real_novel_path


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + '\n')


def source_path(args, eval_dir):
    manifest_file = eval_dir / 'prediction_manifest.json'
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    if manifest and manifest.get('kind') != args.kind:
        raise ValueError(f"Prediction kind {manifest.get('kind')} does not match evaluator {args.kind}")
    source = args.source_path or manifest.get('source_path')
    if not source:
        raise ValueError('Supply --source_path for predictions without prediction_manifest.json')
    return str(Path(source).absolute())


def record_environment(args, eval_dir):
    from evaluation.protocols.environment import metric_environment
    prediction_metadata = eval_dir / 'prediction_metadata.json'
    metadata = json.loads(prediction_metadata.read_text()) if prediction_metadata.exists() else {}
    metadata['metric_environment'] = metric_environment()
    metadata['metric_arguments'] = vars(args)
    save_json(eval_dir / 'evaluation_metadata.json', metadata)


def evaluate_real(args):
    model_dir = Path(args.model_path)
    eval_dir = model_dir / f'eval_{args.iteration}'
    has_source = args.source_path or (eval_dir / 'prediction_manifest.json').is_file()
    source = source_path(args, eval_dir) if has_source or not args.skip_novel else None
    if not args.skip_input:
        from evaluation.metrics.image import eval_input_views
        result = eval_input_views(str(eval_dir), source_path=source)
        save_json(eval_dir / f'metrics_{args.iteration}_wGT.json', result)
        print('input views:', result)
    if not args.skip_novel:
        from evaluation.metrics.video import eval_kid, eval_fvd, N_SPIN
        views = real_novel_path(model_dir, args.iteration, f'novel_views_{N_SPIN}')
        kid = eval_kid(source, str(views), eval_n_spin=args.eval_n_spin, seed=args.rand_seed)
        kid_path = eval_dir / f'kid_results_{args.eval_n_spin}.json'
        if kid is not None:
            save_json(kid_path, {'kid': kid})
        else:
            kid_path.unlink(missing_ok=True)
        fvd = eval_fvd(source, str(views), seed=args.rand_seed)
        save_json(eval_dir / f'fvd_results_{N_SPIN}.json', fvd)
        print(f'novel views: kid {kid}, {fvd}')
    record_environment(args, eval_dir)


def evaluate_artemis(args):
    from evaluation.metrics.geometry import gt_for_scene, eval_prediction_file, source_frame_ids
    eval_dir = Path(args.model_path) / f'eval_{args.iteration}'
    source = source_path(args, eval_dir)
    if not args.skip_views:
        from evaluation.metrics.image import eval_multiview
        with open(Path(source) / 'artemis/val_info.json') as file:
            val_names = json.load(file)['val_names']
        result = eval_multiview(str(eval_dir), args.gt_path or default_gt_path(source), val_names=val_names)
        save_json(eval_dir / f'metrics_{args.iteration}_wGT.json', result)
        print('novel views:', result)
    if not args.skip_chamfer:
        import numpy as np
        root = args.chamfer_root or default_chamfer_root(source)
        gt, diagonal = gt_for_scene(root, source)
        if gt is None:
            raise FileNotFoundError(f'No geometry GT in {root} for {source}; supply --chamfer_root or explicitly use --skip_chamfer')
        prediction = args.prediction or str(eval_dir / 'predicted_points.npz')
        with np.load(prediction, allow_pickle=False) as archive:
            if not np.array_equal(archive['frame_ids'], source_frame_ids(source)):
                raise ValueError('Prediction frame_ids do not match transforms_artemis.json')
        result = eval_prediction_file(prediction, gt, diagonal)
        save_json(eval_dir / f'chamfer_{args.iteration}.json', result)
        print('geometry:', result)
    record_environment(args, eval_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profiles = parser.add_subparsers(dest='kind', required=True)
    for kind in ['real', 'artemis']:
        sub = profiles.add_parser(kind, help=f'Compute {kind} metrics from saved predictions')
        sub.add_argument('-m', '--model_path', required=True, help='prediction/output scene folder')
        sub.add_argument('--iteration', type=int, required=True)
        sub.add_argument('-s', '--source_path', help='prepared sequence; default: prediction_manifest.json')
        if kind == 'real':
            sub.add_argument('--skip_input', action='store_true')
            sub.add_argument('--skip_novel', action='store_true')
            sub.add_argument('--eval_n_spin', type=int, default=16)
            sub.add_argument('--rand_seed', type=int, default=0)
        else:
            sub.add_argument('--skip_views', action='store_true')
            sub.add_argument('--skip_chamfer', action='store_true')
            sub.add_argument('--gt_path', help='scene multi-view GT folder containing images/')
            sub.add_argument('--chamfer_root', help='geometry GT folder containing gt/ and metadata.json')
            sub.add_argument('--prediction', help='world-space point NPZ; default: eval_<iteration>/predicted_points.npz')
    sub = profiles.add_parser('geometry', help='Compute geometry metrics directly from point clouds on CPU')
    sub.add_argument('--prediction', required=True)
    sub.add_argument('--gt_file', required=True)
    sub.add_argument('--diag', type=float, required=True, help='Artemis rest-pose bounding-box diagonal')
    sub.add_argument('--output', required=True, help='output metric JSON')
    args = parser.parse_args()
    if args.kind == 'real':
        evaluate_real(args)
    elif args.kind == 'artemis':
        evaluate_artemis(args)
    else:
        from evaluation.metrics.geometry import eval_prediction_file
        result = eval_prediction_file(args.prediction, args.gt_file, args.diag)
        save_json(Path(args.output), result)
        print('geometry:', result)


if __name__ == '__main__':
    main()
