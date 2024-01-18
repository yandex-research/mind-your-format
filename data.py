import re
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset


MAX_INPUT_LENGTH = 1024
VAL_SPLIT_SEED = 59
VAL_TEST_SIZE = 1000
INPUT_VERBALIZERS = ['input: {}', 'text: {}', 'sentence: {}', '{}']
COMMON_OUTPUT_VERBALIZERS = ['output: {}', 'target: {}', 'label: {}', '{}']
SENTIMENT_OUTPUT_VERBALIZERS = ['emotion: {}', 'sentiment: {}',
                                'A {} one.', 'It was {}.', 'All in all {}.', 'A {} piece.']
TOPIC_OUTPUT_VERBALIZERS = ['Topic: {}.', 'Subject: {}.', 'This is about {}.', 'It is about {}.']
SEPS = [" ", "\n"]
BIG_SEPS = [" ", "\n", "\n\n"]


class TensorDataset(Dataset):
    def __init__(self, test_samples, tokenizer, labels, template,
                 examples=None,
                 method='direct',
                 max_length=MAX_INPUT_LENGTH,
                 ):
        if examples is None:
            examples = []
        if examples and isinstance(examples[0], list):
            assert len(examples) == len(test_samples), "Examples are a list of lists but their length does not match " \
                                                       "the length of the eval dataset."
            examples_for_each_input = True
        else:
            examples_for_each_input = False

        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.template = template
        self.tokenizer = tokenizer
        self.labels = labels
        self.examples = examples
        self.method = method
        self.max_length = max_length

        if examples_for_each_input:
            context = [self.add_examples_to_context(cur_examples, method) for cur_examples in examples]
        else:
            context = [self.add_examples_to_context(examples, method) for _ in test_samples]
        if context and context[0]:
            # if examples is None function above creates len(test_samples) empty contexts.
            # We only want to add big sep after the context if the context is not empty
            context = ("".join((input_context, self.template.big_sep)) for input_context in context)

        if self.tokenizer.bos_token_id is not None: 
            prefix = [self.tokenizer.bos_token_id]
        else:
            prefix = []
        context_tokenized = [prefix + self._get_input_ids(input_context) for input_context in context]

        for input_text, input_context in tqdm(zip(test_samples, context_tokenized)):
            for label in labels:
                input_ids, attention_mask, token_type_ids = self.preprocess_sentence(input_text, label, input_context)

                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.token_type_ids.append(token_type_ids)

    def add_examples_to_context(self, examples, method):
        if 'channel' in method:
            return self.template.big_sep.join(
                f"{self.template.out_verbalizer.format(output)}{self.template.sep}"
                f"{self.template.inp_verbalizer.format(input)}" for input, output in examples)
        else:
            return self.template.big_sep.join(
                f"{self.template.inp_verbalizer.format(input)}{self.template.sep}"
                f"{self.template.out_verbalizer.format(output)}" for input, output in examples)

    def _get_input_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def preprocess_sentence(self, input_text, label, context_tokenized):
        input_text = self.template.inp_verbalizer.format(input_text)
        label = self.template.out_verbalizer.format(label)
        if self.method == 'channel':
            label, input_text = input_text, label

        input_tokenized = self._get_input_ids(input_text)
        sep_tokenized = self._get_input_ids(self.template.sep)
        out_tokenized = self._get_input_ids(label)
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        all_len = len(input_tokenized) + len(sep_tokenized) + len(out_tokenized) + len(eos)
        input_ids = context_tokenized[:self.max_length - all_len] + input_tokenized + sep_tokenized + \
                    out_tokenized + eos

        # determine label tokens, to calculate loss only over them when labels_loss == True
        begin = len(context_tokenized[:self.max_length - all_len]) + len(input_tokenized) + len(sep_tokenized)
        end = len(input_ids) - 1
        attention_mask = [1] * len(input_ids)
        label_tokens = [0] * begin + [1] * (end - begin) + [0]

        to_predict = self.tokenizer.decode(input_ids[begin:end]).strip()
        gt = self.tokenizer.decode(out_tokenized).strip()
        assert to_predict == gt

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(label_tokens)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "token_type_ids": self.token_type_ids[idx],
                }

    def print_tensorized_example(self, idx=0):
        print(self.input_ids[idx])
        print(self.tokenizer.decode(self.input_ids[idx]))


@dataclass
class SST2Dataset:
    dataset_name = "sst2"
    input_col = "sentence"
    target_col = "label"
    val_split = "validation"
    labels_mapping = dict(enumerate(['negative', 'positive']))
    input_verbalizers = INPUT_VERBALIZERS
    output_verbalizers = COMMON_OUTPUT_VERBALIZERS + SENTIMENT_OUTPUT_VERBALIZERS


@dataclass
class DBPediaDataset:
    dataset_name = "dbpedia_14"
    input_col = "content"
    target_col = "label"
    val_split = "test"
    labels_mapping = dict(enumerate(["Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
                                     "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
                                     "Plant", "Album", "Film", "Written Work"]))
    input_verbalizers = INPUT_VERBALIZERS
    output_verbalizers = COMMON_OUTPUT_VERBALIZERS + TOPIC_OUTPUT_VERBALIZERS


@dataclass
class AGNewsDataset:
    dataset_name = "ag_news"
    input_col = "text"
    target_col = "label"
    val_split = "test"
    labels_mapping = dict(enumerate(["World", "Sports", "Business", "Technology"]))
    input_verbalizers = INPUT_VERBALIZERS
    output_verbalizers = COMMON_OUTPUT_VERBALIZERS + TOPIC_OUTPUT_VERBALIZERS


@dataclass
class TRECDataset:
    dataset_name = "trec"
    input_col = "text"
    target_col = "coarse_label"
    val_split = "test"
    labels_mapping = dict(enumerate(["Description", "Entity", "Expression", "Human", "Location", "Number"]))
    input_verbalizers = INPUT_VERBALIZERS
    output_verbalizers = COMMON_OUTPUT_VERBALIZERS + TOPIC_OUTPUT_VERBALIZERS


DATASET_TO_DATACLASS = {"sst2": SST2Dataset, "dbpedia": DBPediaDataset, "agnews": AGNewsDataset, "trec": TRECDataset}


def load_split_dataset(dataset_name, seed=VAL_SPLIT_SEED, cache_dir='~/.cache/huggingface/datasets'):
    dataset_dataclass = DATASET_TO_DATACLASS[dataset_name]
    dataset = load_dataset(dataset_dataclass.dataset_name, cache_dir=cache_dir)
    
    train = pd.DataFrame({
        'input': dataset['train'][dataset_dataclass.input_col],
        'target': dataset['train'][dataset_dataclass.target_col]
    })
    train['target'] = train.target.map(dataset_dataclass.labels_mapping)
    
    if dataset_name in ['dbpedia', 'agnews']:
        # these datasets' validation splits are too big, so we split them.
        _, val = train_test_split(dataset[dataset_dataclass.val_split], test_size=VAL_TEST_SIZE, random_state=seed)
        val = pd.DataFrame({
            'input': val[dataset_dataclass.input_col],
            'target': val[dataset_dataclass.target_col]
        })
        # some dbpedia inputs contain {} which will break templates formatting
        val['input'] = val.input.apply(lambda x: re.sub('[{}]', '', x.strip()))
    else:
        val = pd.DataFrame({
            'input': dataset[dataset_dataclass.val_split][dataset_dataclass.input_col],
            'target': dataset[dataset_dataclass.val_split][dataset_dataclass.target_col]
        })
    val['target'] = val.target.map(dataset_dataclass.labels_mapping)

    return train, val, dataset_dataclass.labels_mapping
