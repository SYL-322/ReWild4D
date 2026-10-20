import os
import json
from glob import glob
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.summaries.validation import IMAGE_METRICS, GEOMETRY_METRICS, read_metrics


def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def find_scenes(data_root, iteration):
    """scenes are the <animal>/<motion> sub folders that contain eval_<iteration>/"""
    scenes = glob(os.path.join(data_root, "*", "*", f"eval_{iteration}"))
    return sorted(os.path.relpath(os.path.dirname(s), data_root) for s in scenes)


def main(output_root, data_name, name_list, iteration, allow_partial=False):
    """
    per-scene evaluation metrics, averaged over the scenes:
      novel_view: eval_<iteration>/metrics_<iteration>_wGT.json   (Artemis views not used for training)
      geometry:   eval_<iteration>/chamfer_<iteration>.json       (all requested scenes)
    """
    if not name_list:
        raise ValueError("No scenes found for evaluation summary")
    if len(name_list) != len(set(name_list)):
        raise ValueError("Scene names must be unique")
    data_root = os.path.join(output_root, data_name)
    files = {'novel_view': f"metrics_{iteration}_wGT.json", 'geometry': f"chamfer_{iteration}.json"}
    required = {'novel_view': IMAGE_METRICS, 'geometry': GEOMETRY_METRICS}
    summary = defaultdict(lambda: defaultdict(list))
    used_names = defaultdict(list)
    for name in name_list:
        for group, fname in files.items():
            path = os.path.join(data_root, name, f"eval_{iteration}", fname)
            try:
                metrics = read_metrics(path, required[group])
            except (FileNotFoundError, ValueError) as e:
                if not allow_partial:
                    error_type = FileNotFoundError if isinstance(e, FileNotFoundError) else ValueError
                    raise error_type(f"Invalid or incomplete {group} result for {name}: {e}. Use --allow_partial to skip this result.") from e
                print(f"Skip {group} of {name}: {e}")
                continue
            used_names[group].append(name)
            for metric, value in metrics.items():
                if metric != 'n_frames':
                    summary[group][metric].append(value)

    if not used_names:
        raise ValueError("No scene results found for evaluation summary")
    print(f"{data_name} {iteration} evaluation summary:")
    for group in summary.keys():
        names = used_names[group]
        for metric, values in summary[group].items():
            mean = sum(values) / len(values)
            min_name = names[np.argmin(values)]
            max_name = names[np.argmax(values)]
            print(f"{group} {metric} ({len(values)} scenes): mean {mean:.4f}, std {np.std(values):.4f}, min_name [{min_name} | {min(values):.4f}], max_name [{max_name} | {max(values):.4f}]")

    for group in list(summary.keys()):
        summary[group]['average'] = {metric: sum(values) / len(values) for metric, values in summary[group].items()}
        summary[group]['scenes'] = used_names[group]

    save_path = f"{output_root}/summary_{data_name}_{iteration}.json"
    save_json(save_path, summary)
    print(f"Saved to {save_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Average the per-scene metrics of one dataset")
    parser.add_argument("--output_root", type=str, required=True, help="folder that contains <data_name>/<animal>/<motion>/")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--scenes", nargs="+", type=str, default=None, help="default: all scenes with eval_<iteration>/")
    parser.add_argument("--allow_partial", action="store_true", help="skip missing or invalid scene metric groups")
    args = parser.parse_args()

    name_list = args.scenes or find_scenes(os.path.join(args.output_root, args.data_name), args.iteration)
    main(args.output_root, args.data_name, name_list, args.iteration, allow_partial=args.allow_partial)
