from tqdm import tqdm

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling

from data import load_split_dataset, TensorDataset
from examples import get_examples


@torch.inference_mode()
def get_loss(generator, batch, labels_loss=False):
    model = generator.model
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)

    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    label_tokens = batch['token_type_ids']
    labels = torch.where(attention_mask == 1, input_ids, -100)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[..., :-1, :].contiguous().to(model.dtype)
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    losses = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(logits.size(0), logits.size(1))
    if labels_loss:
        label_mask = label_tokens[..., 1:].contiguous().to(model.device)
        losses = losses * label_mask
        losses = losses.sum(dim=-1) / label_mask.sum(dim=-1)
    else:
        losses = losses.mean(dim=-1)
    losses = losses.detach().cpu()

    return losses


def classify(losses, labels, correction_factor=None, mode="diagonal_W"):
    """this function applies a correction factor from the calibrate method to the model's predicted distribution"""
    num_classes = len(labels)
    if correction_factor is None:
        # do not calibrate
        W = torch.eye(num_classes, dtype=losses.dtype)
        b = torch.zeros(num_classes, dtype=losses.dtype)
    else:
        # calibrate
        if mode == "diagonal_W":
            W = torch.linalg.inv(torch.eye(num_classes, dtype=losses.dtype) * correction_factor)
            b = torch.zeros(num_classes, dtype=losses.dtype)
        elif mode == "identity_W":
            W = torch.eye(num_classes)
            b = -1 * correction_factor[:, None]
        else:
            raise NotImplementedError(f"{mode} is not implemented for calibration")

    uncalibrated_probs = softmax(-losses)
    calibrated_probs = torch.matmul(uncalibrated_probs, W) + b

    return np.array(labels)[calibrated_probs.argmax(1)], calibrated_probs


def predict(generator, eval_dataset, labels, batch_size=1, method='direct', labels_loss=False,
            calibrate_dataset=None, mode='diagonal_W'):
    collator = DataCollatorForLanguageModeling(generator.tokenizer, mlm=False)

    if method == 'calibrate':
        calibrate_dataloader = DataLoader(calibrate_dataset,
                                          shuffle=False,
                                          batch_size=batch_size,
                                          collate_fn=collator)
        # get probability distribution for context-free inputs
        cf_losses = []
        for batch in tqdm(calibrate_dataloader):
            cf_losses.extend(get_loss(generator, batch, labels_loss))
        cf_losses = torch.tensor(cf_losses, dtype=torch.float32).reshape(-1, len(labels))
        cf_label_probs = softmax(-cf_losses)
        # calculate calibration correction term
        correction_factor = torch.mean(cf_label_probs, dim=0)
        torch.cuda.empty_cache()
    else:
        correction_factor = None

    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=collator)
    losses = []
    for batch in tqdm(eval_dataloader):
        losses.extend(get_loss(generator, batch, labels_loss))

    losses = torch.tensor(losses, dtype=torch.float32).reshape(-1, len(labels))
    results, probs = classify(losses, labels, correction_factor, mode)

    return results, probs


def evaluate_setup(dataset, generator, seed, template, num_shots, selection_method,
                   example_ids=None, examples_path=None,
                   prediction_method='direct',
                   labels_loss=False,
                   batch_size=16,
                   cache_dir=None,
                   ):
    train, val, labels_mp = load_split_dataset(dataset, cache_dir=cache_dir)
    labels = list(labels_mp.values())

    selected_examples = get_examples(dataset, train, selection_method, seed, num_shots,
                                     example_ids=example_ids,
                                     examples_path=examples_path,
                                     )
    examples, example_ids = selected_examples["examples"], selected_examples["example_ids"]

    eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                 generator.tokenizer, labels, template,
                                 examples=examples,
                                 method=prediction_method,
                                 )
    eval_dataset.print_tensorized_example()

    if 'calibrate' in prediction_method:
        context_free_inputs = ["N/A", "", "[MASK]"]
        calibrate_dataset = TensorDataset(context_free_inputs, generator.tokenizer, labels, template,
                                          examples=examples, method='direct',
                                          )
    else:
        calibrate_dataset = None

    results, probs = predict(generator, eval_dataset, labels, batch_size=batch_size, method=prediction_method,
                             labels_loss=labels_loss, calibrate_dataset=calibrate_dataset)
    score = (results == val['target']).mean()

    return {"score": score, "probs": probs, "predicts": results, "example_ids": example_ids}
