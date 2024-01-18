# Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements
[![arXiv](https://img.shields.io/badge/arXiv-2401.06766-b31b1b.svg)](https://arxiv.org/abs/2401.06766)

In [our work](https://arxiv.org/abs/2401.06766), we explore the impact of prompt template choice on the In-Context Learning performance of Large Language Models in various prediction and demonstration selection methods. As a first step towards mitigating this issue we
propose Template Ensembles.

This repository contains the official code for our paper and the results of all our experiments. 


# Installation

Create a new Conda environment:

```bash
conda create --name templates python=3.8
conda activate templates
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

# Evaluation

In order to run experiments with LLaMA and LLaMA 2, you need to access weights of the corresponding model on the HuggingFace Hub model page (e.g. for [LLaMA 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf))

Note that you may need to change `--eval_batch_size` depending on a particular model and your hardware.

You may need to use more than 1 GPU for some models. In this case, you may also want to set `--device_map balanced_low_0` (see [HuggingFace documentation](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) for details)

## Baseline results

To get the baseline results (0-shot or few-shot with Random selection method and Direct prediction method), run the following command:

```bash
python selection_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --seed 59 13 21 \
  --num_shots [0/2/4] \
  --save_dir {path_to_csv_with_results} \
  --wandb_entity {your_wandb_account} \
  --wandb_project {your_wandb_project} \
  --prediction_method direct \
  --eval_batch_size 16
```

## Prediction methods

```bash
python selection_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --seed 59 13 21 \
  --num_shots 2 \
  --save_dir {path_to_csv_with_results} \
  --wandb_entity {your_wandb_account} \
  --wandb_project {your_wandb_project} \
  --prediction_method [direct/channel/calibrate] \
  [--labels_loss] \
  --eval_batch_size 16
```
`--labels_loss` option is used to calculate loss only over labels' tokens. Use it for Channel and Calibrate prediction methods.

## Example selection methods 

```bash
python selection_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --examples_path selected_examples/{method}/{dataset} \
  --seed 59 13 21 \
  --num_shots [2/4] \
  --save_dir {path_to_csv_with_results} \
  --wandb_entity {your_wandb_account} \
  --wandb_project {your_wandb_project} \
  --prediction_method [direct/calibrate/channel] \
  --examples_selection_method [random/implicitly_topic_models/z-ICL] \
  --eval_batch_size 16
```

## Template transfer
In order to evaluate template transfer between different setups, you need to run evaluation of the desired methods varying the `--template_seed` argument. See the example for different prediction methods below:

```bash
python selection_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --seed 59 \
  --template_seed 59 13 21 \
  --num_shots 2 \
  --save_dir {path_to_csv_with_results} \
  --wandb_entity {your_wandb_account} \
  --wandb_project {your_wandb_project} \
  --prediction_method [direct/channel/calibrate] \
  [--labels_loss] \
  --eval_batch_size 16
```
# Template ensembles

You can get results for template ensembles with the following command:

```
templates_ensemble.py \
  -m {model} \
  -dataset [sst2/dbpedia/agnews/trec] \
  --num_templates 5 \
  --num_shots 2 \
  --prediction_method direct \
  --eval_batch_size 16 \
  --save_dir {save_path} \
  --cache_dir {hf_cache_path} \
  --seed 13 21 59
```

# Results analysis

We provide the results for all our experiments and the code to reproduce figures and tables from our paper in the `plots_and_tables` folder.

In case you run your own experiments, the results will be stored locally at `{save_dir}/all_runs.csv` or you can download them from your Weights & Biases project.

# Citation
If you make use of our work, please cite our paper:

```
@misc{voronov2024mind,
      title={Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements}, 
      author={Anton Voronov and Lena Wolf and Max Ryabinin},
      year={2024},
      eprint={2401.06766},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
