# RLCD: Reinforcement Learning from Contrast Distillation

This repo contains code and instructions for reproducing the experiments in the paper "RLCD: Reinforcement Learning from Contrast Distillation for Language Model Alignment" (https://arxiv.org/abs/2307.12950), by Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. RLCD is a simple and effective way to simulate preference data generation for RLHF-based LLM alignment pipelines, comparing favorably to RLAIF and context distillation baselines on diverse alignment tasks across multiple LLaMA scales for data simulation.

## Installation

We rely on AlpacaFarm (https://github.com/tatsu-lab/alpaca_farm) for the various model training commands needed to run RLCD and our baselines after the initial data generation. Download it, change to the correct commit, and patch it according to our modifications:

```
git clone https://github.com/tatsu-lab/alpaca_farm.git
cd alpaca_farm
git checkout b68d7cf468c88432b1c91f6f495a0c7c900fbf5d
git am ../alpaca-farm-modifications.patch
```

The main changes we made are just related to data processing / prompt formatting. You can then follow the instructions in the AlpacaFarm README to install it to a conda env. 

The main packages required to run the data generation scripts for RLCD and baselines should already be included in the AlpacaFarm install. 

## Running RLCD

There are four main steps to running RLCD: (1) generating the simulated preference data, (2) training the reward model, (3) using the reward model to optimize the base LM using PPO, and finally (4) generating outputs. 

Our experiments are based on LLaMA-7B as the base model to be aligned, although data generation additionally supports generation using larger LLaMA models, e.g., 30B in some of our experiments. The training and test prompts for the harmless and helpful tasks are derived from https://github.com/anthropics/hh-rlhf/tree/master, though outline prompts are omitted for legal reasons.

### Preference Data Simulation

Below are instructions for simulating preference data. Alternatively, `simulated_data.zip` contains the RLCD simulated data from our experiments on the harmless and helpful tasks.

First, get the prompts used as starting points for preference data simulation from Anthropic. We use the `train.jsonl` files from https://github.com/anthropics/hh-rlhf/tree/master/harmless-base for the harmless task or https://github.com/anthropics/hh-rlhf/tree/master/helpful-base for the helpful task. Create the `output` directory, unzip the `train.jsonl`, and move it to `output/anthropic_train.jsonl`. (These `anthropic_train.jsonl` files also contain existing outputs with human-labeled preferences, which we won't use during data simulation. We'll just strip from the end until the last occurrence of "Assistant:" in the "chosen" field of each line of the jsonl. If you want to use your own prompts, you can just provide similarly-formatted prompts in a jsonl where each line just contains the "chosen" field with a dialogue prefix ending with "Assistant:".) Similarly, move Anthropic's corresponding `test.jsonl` to `output/anthropic_test.jsonl` which we'll use during evaluation later.

Then run the main preference data simulation command as follows (you may need to run `pip install bitsandbytes` for 8-bit loading):

```
python scripts/simulate_preference_data.py --task {task} --prompts-file output/anthropic_train.jsonl --method rlcd --model-string {path/to/huggingface/format/llama7b} --out-dir output/simulated_preference_data
```

Main arguments:

`--task`: `harmless` or `helpful` depending on which prompt set you want to use. This also controls how positive and negative prompts are constructed in RLCD.
`--model-string`: Point this to the LLaMA checkpoint you want to use for data generation. This checkpoint should be in the HuggingFace format; use e.g., https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py for conversion as needed.
`--out-dir`: Folder to write data to. The script will write a json file for each data point. 
`--batch-size`: Defaults to 16, which seems fine for our experiments on a single A100 GPU when using LLaMA-7B. You can increase or decrease this accordingly depending on which model you're generating with and how much memory your GPU has. For example, we decreased to 8 when running LLaMA-30B.

Additionally, you can set `--indices`, `--indices-mod`, and `--indices-remainder` to coordinate parallel execution of multiple copies of the script. For example, set `--indices 17-29,35-40` to only generate data for the corresponding data indices in the `--prompts-file`, and set e.g. `--indices-mod 8 --indices-remainder 3` to only generate data for those indices specified through `--indices` which are 3 mod 8.

Afterward, consolidate the data into a single file:

```
python scripts/consolidate_preference_data.py --in-dir output/simulated_preference_data --out-csv output/simulated_preference_data_consolidated.csv --data-type rlcd
```

The previous `simulated_preference_data` directory will no longer be used after this point, so you can delete it if you like.  

### Reward Model Training

We'll be using pretrained LLaMA-7B as the starting point for both reward model training here and PPO alignment later. AlpacaFarm expects the LLaMA-7B checkpoint to be formatted slightly differently from HuggingFace (slightly different config files, possibly version-related). Run the following command to save a re-formatted version (it'll download one of AlpacaFarm's weight diffs and just use that to match the format, while ignoring the actual weights; we haven't yet tried getting AlpacaFarm to work on larger LLaMA sizes).

```
python alpaca_farm/pretrained_models/convert_llama7b_to_alpacafarm_format.py --in-dir {path/to/huggingface/format/llama7b} --out-dir alpaca_farm/pretrained_models
```

The main preference (reward) model training command is as follows:

```
mkdir ckpt
cd alpaca_farm
bash examples/scripts/reward_modeling.sh ../ckpt/reward_model pretrained_models/llama7b ../output/simulated_preference_data_consolidated.csv
cd ..
```

The model will be saved to `ckpt/reward_model`.

You can evaluate the preference model's agreement with human preferences as follows. We used the same `anthropic_train.jsonl` files before, since they already contain labeled human preferences. 

```
cd alpaca_farm
python examples/scripts/evaluate_reward_model_human_agreement.py --model-path ../ckpt/reward_model --data-path ../output/anthropic_train.jsonl
cd ..
```

You can change `--batch-size` (defaults to 24) as needed. By default this command will evaluate 2000 examples, which you can change using `--num-pairs`.

### PPO Training

First, convert Anthropic's data to a csv containing just the prompts: 

```
python scripts/format_input_prompts.py --in-file output/anthropic_train.jsonl --out-file output/ppo_input_prompts.csv
```

The main PPO training command for optimizing/aligning LLaMA-7B according to our reward model is below:

```
cd alpaca_farm
bash examples/scripts/rlhf_ppo.sh ../ckpt/ppo_training ../ckpt/reward_model pretrained_models/llama7b ../output/ppo_input_prompts.csv {num_steps} {kl}
cd ..
```

Checkpoints will be saved to `ckpt/ppo_training` every 20 "steps" (512 rollouts each).

Hyperparameters: If in doubt, for the harmless and helpful tasks we observed qualitatively that setting KL coefficients of 0.002 or 0.004 at 40-80 steps ({num_steps}) would usually give reasonable outputs. Too many steps can lead to worse outputs after a while, as noted in the original AlpacaFarm readme as well. 

### Generating and Evaluating Outputs

First format the test input prompts (in the paper experiments, we used the first 1k examples for dev and second 1k for test):

```
python scripts/format_input_prompts.py --in-file output/anthropic_test.jsonl --out-file output/test_input_prompts.csv
```

Then you can run the following command to generate outputs: 

```
python alpaca_farm/examples/best_of_n.py --task "run_best_of_n" --decoder_name_or_path ckpt/ppo_training/checkpoint-{num_steps} --output_path output/outputs.json --dataset_name output/test_input_prompts.csv
```

You can change `--per_device_batch_size` (defaults to 2) as needed, and set `--max_instances` to limit how many examples to generate. The script has been modified from the original AlpacaFarm version so that it only generates one output per example without reranking.

Finally, postprocess the outputs slightly:

```
python scripts/postprocess.py --in-file output/outputs.json --out-file output/final_outputs_postprocessed.json
```

You can also evaluate the quality of the outputs using the learned reward model as follows:

```
cd alpaca_farm
python examples/scripts/evaluate_outputs.py --model-path ../ckpt/reward_model --data-path ../output/final_outputs_postprocessed.json --output-path ../output/final_outputs_evaluated.json
cd ..
``` 

`--batch-size` defaults to 24, which can be adjusted as needed.

However, evaluating with the learned reward model is extremely imperfect; for example, too much PPO training can cause the model to optimize too heavily against the reward model. Therefore, this evaluation should only really be used for hyperparameter optimization, where we assume we don't have access to a better oracle evaluation. 

A better evaluation would be running pairwise human or GPT-4 comparisons against a baseline (instructions on baselines later on). For GPT-4, you can run the evaluations as follows. 

```
export OPENAI_API_KEY="{your openai api secret key}"
python scripts/gpt4_compare.py --in-file1 output/final_outputs_postprocessed.json --in-file2 {path/to/other/json} --out-file output/gpt4_comparison_data.json --prompt-format {prompt_format} > output/gpt4_comparison.txt
```

The two input files should be in the same format and should contain outputs for the same dialogue prefixes, in the same order. In our paper experiments, `--prompt-format` was `harmless` for harmlessness evaluation on harmlessness prompts, `harmless_helpful` for helpfulness evaluation on harmlessness prompts, and `helpful` for helpfulness evaluation on helpfulness prompts.

### Running RLCD on Other Tasks

If you want to run RLCD on your own task, replace `anthropic_train.jsonl` and `anthropic_test.jsonl` to use your own prompts; only the "chosen" field matters in the jsonl and you don't need to include the final assistant response (i.e., format it as a conversation delimited by "\n\nHuman" and "\n\nAssistant", and you can just let it end with "Assistant:" as the last word). Then add a new option to `--task` in `scripts/simulate_preference_data.py`, and edit the prompt construction logic in the "create_prompts" method. You can also edit the "process_filter_result" method to edit the data postprocessing logic, which is currently pretty simple.

If you want to run the RLCD-Rescore variant, additionally edit the prompt construction logic in the "analyze_results" method which does the post hoc scoring. You may need to experiment with PPO hyperparameters, but we think hyperparameter settings shouldn't be make-or-break for basic functionality, as long as you don't do too many PPO steps. 

## Running Baselines and Variants

Here we describe how to run and generate outputs for our baselines and for the RLCD-Rescore variant in the paper. 

### LLaMA

Just change the checkpoint specified when generating outputs:

```
python alpaca_farm/examples/best_of_n.py --task "run_best_of_n" --decoder_name_or_path alpaca_farm/pretrained_models/llama7b --output_path output/outputs.json --dataset_name output/test_input_prompts.csv
```

and then postprocess and evaluate as normal.

### Context Distillation

Context distillation will rely on the same `simulated_preference_data_consolidated.csv` data generated by RLCD's data generation procedure, though only using the outputs from the positive prompts. We use AlpacaFarm for supervised fine-tuning. First format the data:

```
python scripts/format_context_distillation_training_data.py --in-file output/simulated_preference_data_consolidated.csv --out-file output/context_distillation_training_data.csv
```

Then run the fine-tuning as follows:

```
cd alpaca_farm
bash examples/scripts/sft.sh ../ckpt/context_dist pretrained_models/llama7b ../output/context_distillation_training_data.csv
cd ..
```

You can then generate and evaluate outputs using `--decoder_name_or_path ckpt/context_dist`, similar to the LLaMA baseline.

### RLAIF

In the initial preference data simulation script (`scripts/simulate_preference_data.py`), specify `--method rlaif` instead of `--method rlcd`. Similarly specify `--data-type rlaif` instead of `--data-type rlcd` for `scripts/consolidate_preference_data.py`. The rest of the pipeline will use the same commands as before.

### RLCD-Rescore Variant

Same as RLAIF above, but specify `rlcd_rescore` wherever you would have specified `rlaif`. 


## License

This repo is MIT licensed, as found in the LICENSE file.