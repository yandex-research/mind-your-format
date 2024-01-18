import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


NAMES_TO_CHECKPOINTS = {
  'gptj': 'EleutherAI/gpt-j-6B',
  'gpt-neox': 'EleutherAI/gpt-neox-20b',
  'opt-6.7b': "facebook/opt-6.7b",
  'opt-30b': "facebook/opt-30b",
  'opt-66b': "facebook/opt-66b",
  'bloom-1.7b': 'bigscience/bloom-1b7',
  'bloom-3b': 'bigscience/bloom-3b',
  'bloom-7.1b': 'bigscience/bloom-7b1',
  'pythia-6.9b': 'EleutherAI/pythia-6.9b',
  'pythia-12b': 'EleutherAI/pythia-12b',
  'llama-7b': 'Neko-Institute-of-Science/LLaMA-7B-HF',
  'llama-13b': 'Neko-Institute-of-Science/LLaMA-13B-HF',
  'llama-30b': 'Neko-Institute-of-Science/LLaMA-30B-hf',
  'llama-65b': 'Neko-Institute-of-Science/LLaMA-65B-hf',
  'falcon-1b': 'tiiuae/falcon-rw-1b',
  'falcon-7b': 'tiiuae/falcon-7b',
  'falcon-40b': 'tiiuae/falcon-40b',
  'llama-2-13b-hf': 'meta-llama/Llama-2-13b-hf',
  'llama-2-70b-hf': 'meta-llama/Llama-2-70b-hf',
}


class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()


def load_generator(model_name, cache_dir=None, precision='fp16', local_files_only=False, device_map="auto"):
    torch.backends.cudnn.deterministic = True

    precision = torch.float16 if precision == 'fp16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        NAMES_TO_CHECKPOINTS[model_name],
        cache_dir=cache_dir,
        padding_side='right',
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        NAMES_TO_CHECKPOINTS[model_name],
        cache_dir=cache_dir,
        torch_dtype=precision,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if 'llama' in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model.config.pad_token_id = tokenizer.unk_token_id
    else:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    generator = Generator(model=model, tokenizer=tokenizer)

    return generator
