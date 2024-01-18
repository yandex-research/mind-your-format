import warnings
from itertools import product

from evaluate import evaluate_setup
from models import load_generator
from utils import parse_args, get_results_pd, find_current_run, save_results_pd
from templates import get_templates
try:
    import wandb
except ImportError:
    wandb = None

if __name__ == '__main__':
    args = parse_args()
    results = get_results_pd(args.save_dir)

    for model in args.models:
        generator = load_generator(model, cache_dir=args.cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only, device_map=args.device_map,
                                   )
        for dataset, seed, prediction_method, selection_method in product(
                args.dataset, args.seed, args.prediction_method, args.examples_selection_method):
            if selection_method == '0-shot':
                num_shots_range = [0]
            else:
                num_shots_range = args.num_shots

            for num_shots in num_shots_range:
                if prediction_method in ["channel", "calibrate"]:
                    labels_loss = True
                    if not args.labels_loss:
                        warnings.warn(f"Using {prediction_method} with labels_loss set to False is highly discouraged, "
                                      f"setting to True.")
                else:
                    labels_loss = args.labels_loss

                templates = get_templates(dataset, num_shots, args.num_templates, args.templates_path,
                                          args.template_seed)
                method_name = f"{prediction_method}_{labels_loss}"
                config = {'dataset': dataset, 'model': model, 'seed': seed,
                          'example_selection_method': selection_method, 'n_shots': num_shots,
                          'prediction_method': method_name, 'batch_size': args.eval_batch_size,
                          'precision': args.precision,
                          'template_seed': args.template_seed,
                          }

                scores = find_current_run(config=config, results=results)

                if len(scores) == len(templates):
                    continue

                if args.use_wandb:
                    wandb.init(entity=args.wandb_entity, project=args.wandb_project, reinit=True, config=config)

                num_evaluated_templates = len(scores)
                for template in templates[num_evaluated_templates:]:
                    evaluation_result = evaluate_setup(dataset=dataset, generator=generator, seed=seed,
                                                       template=template,
                                                       prediction_method=prediction_method, labels_loss=labels_loss,
                                                       selection_method=selection_method, num_shots=num_shots,
                                                       example_ids=args.example_ids, examples_path=args.examples_path,
                                                       batch_size=args.eval_batch_size, cache_dir=args.cache_dir,
                                                       )
                    score = evaluation_result["score"]
                    results = save_results_pd(res_df=results, run_config=config, score=score, template=template,
                                              save_dir=args.save_dir,
                                              )
                    scores.append(score)
                if args.use_wandb:
                    # log to wandb only fully completed runs
                    wandb.log({'scores': scores,
                               'example_ids': evaluation_result["example_ids"],
                               "templates": [template.toJSON() for template in templates],
                               })
