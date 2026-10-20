import os
import json
from glob import glob
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.summaries.validation import IMAGE_METRICS, read_metrics
from evaluation.protocols.paths import real_novel_path


def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def find_scenes(data_root, iteration):
    """scenes are the sub folders (depth 1 or 2, e.g. APTv2 <animal>/<clip>) that contain eval_<iteration>/"""
    scenes = glob(os.path.join(data_root, "*", f"eval_{iteration}")) + glob(os.path.join(data_root, "*", "*", f"eval_{iteration}"))
    return sorted(os.path.relpath(os.path.dirname(s), data_root) for s in scenes)


def main(output_root, data_name, name_list, iteration, eval_n_spin=16, n_spin=64, allow_partial=False):
    if not name_list:
        raise ValueError("No scenes found for evaluation summary")
    if len(name_list) != len(set(name_list)):
        raise ValueError("Scene names must be unique")
    data_root = os.path.join(output_root, data_name)
    summary = defaultdict(lambda: defaultdict(list))
    used_names = []
    for name in name_list:
        scene_dir = os.path.join(data_root, name)
        try:
            # Input-view and novel-view metrics.
            metrics_wGT = read_metrics(f"{scene_dir}/eval_{iteration}/metrics_{iteration}_wGT.json", IMAGE_METRICS)
            kid = read_metrics(real_novel_path(scene_dir, iteration, f"kid_results_{eval_n_spin}.json"), ('kid',))
            fvd = read_metrics(real_novel_path(scene_dir, iteration, f"fvd_results_{n_spin}.json"), ('fvd_f', 'fvd_diag'))
        except (FileNotFoundError, ValueError) as e:
            if not allow_partial:
                error_type = FileNotFoundError if isinstance(e, FileNotFoundError) else ValueError
                raise error_type(f"Invalid or incomplete evaluation for {name}: {e}. Use --allow_partial to skip this scene.") from e
            print(f"Skip {name}: {e}")
            continue
        used_names.append(name)
        for metric, value in metrics_wGT.items():
            summary['original'][metric].append(value)
        for metric, value in {**kid, **fvd}.items():
            summary['novel'][metric].append(value)
    
    if not used_names:
        raise ValueError("No complete scene results found for evaluation summary")
    print(f"{data_name} {iteration} evaluation summary ({len(used_names)} scenes):")
    for view in summary.keys():
        for metric, values in summary[view].items():
            mean = sum(values) / len(values)
            min_name = used_names[np.argmin(values)]
            max_name = used_names[np.argmax(values)]
            print(f"{view} {metric}: mean {mean:.4f}, std {np.std(values):.4f}, min_name [{min_name} | {min(values):.4f}], max_name [{max_name} | {max(values):.4f}]")
    
    for view in ['original', 'novel']:
        summary[view]['average'] = {metric: sum(values) / len(values) for metric, values in summary[view].items()}
    summary['scenes'] = used_names
        
    save_path = f"{output_root}/summary_{data_name}_{iteration}.json"
    save_json(save_path, summary)
    print(f"Saved to {save_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Average the per-scene metrics of one dataset")
    parser.add_argument("--output_root", type=str, required=True, help="folder that contains <data_name>/<scene>/")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--scenes", nargs="+", type=str, default=None, help="default: all scenes with eval_<iteration>/")
    parser.add_argument("--allow_partial", action="store_true", help="skip scenes with missing or invalid metric results")
    args = parser.parse_args()
    
    name_list = args.scenes or find_scenes(os.path.join(args.output_root, args.data_name), args.iteration)
    main(args.output_root, args.data_name, name_list, args.iteration, allow_partial=args.allow_partial)
