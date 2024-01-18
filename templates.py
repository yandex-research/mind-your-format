import json
import pickle
import random
import warnings
from itertools import product
from pathlib import Path

from data import DATASET_TO_DATACLASS, SEPS, BIG_SEPS
from evaluate import evaluate_setup
from models import load_generator
from utils import parse_args, get_results_torch, save_results_torch, save_results_nirvana


class Template:
    def __init__(self, inp_verbalizer, out_verbalizer, sep, big_sep=""):
        self.inp_verbalizer = inp_verbalizer
        if "{}" not in self.inp_verbalizer:
            raise ValueError("inp_verbalizer must contain {} for formatting")
        self.out_verbalizer = out_verbalizer
        if "{}" not in self.out_verbalizer:
            raise ValueError("out_verbalizer must contain {} for formatting")
        self.sep = sep
        self.big_sep = big_sep

    def __repr__(self):
        return "".join([self.inp_verbalizer, self.sep, self.out_verbalizer, self.big_sep])

    def __str__(self):
        return "".join([self.inp_verbalizer, self.sep, self.out_verbalizer, self.big_sep])

    def __hash__(self):
        return hash(self.__str__())

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


def generate_random_templates(dataset_name, n=10, seed=59, with_big_seps=False):
    dataset_dataclass = DATASET_TO_DATACLASS[dataset_name]
    inp_verbalizers = dataset_dataclass.input_verbalizers
    out_verbalizers = dataset_dataclass.output_verbalizers
    random.seed(seed)
    random_templates = []
    for _ in range(n):
        inp_verbalizer = random.choice(inp_verbalizers)
        out_verbalizer = random.choice(out_verbalizers)
        sep = random.choice(SEPS)
        big_sep = random.choice(BIG_SEPS) if with_big_seps else ''
        random_templates.append(Template(inp_verbalizer, out_verbalizer, sep, big_sep))

    return random_templates



def get_templates(dataset, n_shots, n_templates=10, templates_path=None, templates_seed=59):
    if templates_path is None:
        templates = generate_random_templates(dataset,
                                              with_big_seps=n_shots > 0,
                                              n=n_templates,
                                              seed=templates_seed)
    else:
        with open(templates_path, "rb") as templates_file:
            templates = pickle.load(templates_file)

    return templates


if __name__ == '__main__':
    args = parse_args()
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
                res_dir = args.save_dir if args.save_dir else Path("results/template_selection", dataset)
                if prediction_method in ["channel", "calibrate"]:
                    if not args.labels_loss:
                        warnings.warn(f"Using {prediction_method} with labels_loss set to False is highly discouraged, "
                                      f"setting to True.")
                    labels_loss = True
                else:
                    labels_loss = args.labels_loss

                method_name = f"{prediction_method}_{labels_loss}"

                name = f"{model}_formats_stats"
                if num_shots > 0:
                    name += "_zero_shot"
                name += f"_{method_name}.out"

                results = get_results_torch(save_dir=res_dir, name=name)

                inp_verbalizers, out_verbalizers, seps = dataset_templates(dataset)
                templates = (Template(inp, out, sep, big_sep)
                             for inp, out, sep, big_sep in product(inp_verbalizers, out_verbalizers, seps, BIG_SEPS))
                for template in templates:
                    if template in results:
                        continue

                    evaluation_result = evaluate_setup(dataset=dataset, generator=generator, seed=seed,
                                                       num_shots=num_shots,
                                                       prediction_method=prediction_method, labels_loss=labels_loss,
                                                       selection_method=selection_method,
                                                       example_ids=args.example_ids, examples_path=args.examples_path,
                                                       batch_size=args.eval_batch_size,
                                                       template=template,
                                                       )
                    results[template] = evaluation_result["scores"]
                    save_results_torch(res_obj=results, name=name, save_dir=res_dir)
                    save_results_nirvana()
