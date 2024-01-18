import json
import random


def load_examples(path, seed, num_shots):
    with open(path) as examples_file:
        selected_examples = json.load(examples_file)
    demonstration_data = selected_examples[str(seed)][str(num_shots)]
    if isinstance(demonstration_data[0], int):
        # examples are stored as indexes in the respective dataset
        return {'example_ids': demonstration_data, 'examples': None}
    else:
        # examples come without ids (e.g. z-ICL: examples are not from the dataset)
        return {'example_ids': None, 'examples': demonstration_data}


def get_examples(dataset, train, selection_method, seed, num_shots, example_ids=None, examples_path=None):
    examples = None
    if example_ids is None:
        if selection_method not in ['random', '0-shot']:
            if examples_path is None:
                examples_path = f"selected_examples/{selection_method}/{dataset}.json"

            try:
                loading_result = load_examples(examples_path,
                                               seed, num_shots)
                examples, example_ids = loading_result["examples"], loading_result["example_ids"]
            except FileNotFoundError:
                raise FileNotFoundError(f"Attempted to find examples at the {examples_path}. No Luck. "
                                        "All methods except zero-shot and random require either example ids or a path "
                                        "to the {dataset}.json")
        elif selection_method == 'random':
            random.seed(seed)
            example_ids = random.sample(range(len(train)), num_shots)
        else:
            # selection_method == '0-shot'
            example_ids = []

    if examples is None and example_ids is not None:
        examples = [(train.iloc[idx]['input'].strip(), train.iloc[idx]['target'].strip()) for idx in example_ids]

    return {'example_ids': example_ids, 'examples': examples}
