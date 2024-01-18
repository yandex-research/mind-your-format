import itertools
import os
import warnings
from collections import defaultdict

from evaluate import evaluate_setup
from models import load_generator
from utils import parse_args, get_results_torch, save_results_torch
from templates import get_templates


if __name__ == "__main__":
    args = parse_args()

    for model in args.models:
        generator = load_generator(model, cache_dir=args.cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only, device_map=args.device_map,
                                   )
        for dataset, seed, prediction_method, selection_method in itertools.product(
                args.dataset, args.seed, args.prediction_method, args.examples_selection_method):
            if selection_method == '0-shot':
                num_shots_range = [0]
            else:
                num_shots_range = args.num_shots

            if prediction_method in ["channel", "calibrate"]:
                if not args.labels_loss:
                    warnings.warn(f"Using {prediction_method} with labels_loss set to False is highly discouraged, "
                                  f"setting to True.")
                labels_loss = True
            else:
                labels_loss = args.labels_loss

            method_name = f"{prediction_method}_{labels_loss}"

            for num_shots in num_shots_range:
                templates = get_templates(dataset, num_shots, args.num_templates, args.templates_path, seed)

                # skip already computed scores
                save_dir = os.path.join(args.save_dir, dataset)
                name = f"{num_shots}_shot_ensembles.out"
                results = get_results_torch(save_dir=save_dir, name=name)
                if model not in results:
                    results[model] = defaultdict(dict)
                if method_name not in results[model]:
                    results[model][method_name] = defaultdict(dict)
                if seed not in results[model][method_name]:
                    results[model][method_name][seed] = defaultdict(list)
                num_evaluated_templates = len(results[model][method_name][seed]["scores"])

                for template in templates[num_evaluated_templates:]:
                    evaluation_result = evaluate_setup(dataset=dataset, generator=generator, seed=seed,
                                                       template=template,
                                                       num_shots=num_shots, selection_method=selection_method,
                                                       example_ids=args.example_ids, examples_path=args.examples_path,
                                                       prediction_method=prediction_method, labels_loss=labels_loss,
                                                       batch_size=args.eval_batch_size, cache_dir=args.cache_dir,
                                                       )
                    for key in ["scores", "predicts", "probs"]:
                        results[model][method_name][seed][key].append(evaluation_result[key])
                    save_results_torch(res_obj=results, name=name, save_dir=save_dir)
