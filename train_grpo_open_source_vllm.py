from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import pandas as pd
import json
from cs_eval import separate_script
from datasets import Dataset
from accelerate import PartialState
from utils import load_args, load_sae
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(args):
    formatted_data = pd.read_json('./data/grpo/training.jsonl', lines=True)
    formatted_dataset = Dataset.from_pandas(formatted_data)

    model_path = args.sft_mdoel_path
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='bfloat16')
    
   
    output_dir = f"{'/'.join(model_path.split('/')[:-3])}/grpo/temperature-{args.temperature}-lr-{args.lr}"
    run_name = f"grpo-{'-'.join(model_path.split('/')[-5:-3])}-temperature-{args.temperature}-lr-{args.lr}"

    lan_dict = {'en': ['Latn'], 'es': ['Latn'], 'fr': ['Latn'], 'ja': ['Hira','Kana'], 'ko': ['Hang'], 'pt': ['Latn'], 'th': ['Thai'], 'vi': ['Latn'], 'zh': ['Hani'], 'ar': ['Arab'],'ru':['Cyrl'], 'common': ['Zyyy']}
    reduced_target=lan_dict[args.reduced_lan][0]
    def reward_len(completions, **kwargs):
        out = []
        for i in range(len(completions)):
            gen=completions[i][0]['content']
            results = separate_script(gen)
            lan=kwargs['prompt_lan'][i]
            if (len(gen)-len(results['Zyyy']))==0:
                ratio=0
            else:
                ratio = len(''.join([results[s] for s in lan_dict[lan]]))/(len(gen)-len(results['Zyyy']))
            out.append(ratio)
        return out
    if 'qwen3-8B' in model_path.lower():
        max_completion_length = 512
        gradient_accumulation_steps=32
        num_generations=8
        per_device_train_batch_size=1
        bf16=True
    elif 'gemma-2-9b' in model_path.lower():
        max_completion_length =512
        gradient_accumulation_steps=32
        num_generations=8
        per_device_train_batch_size=1
        bf16=True
    elif 'gemma-2-2b' in model_path.lower():
        max_completion_length =512
        gradient_accumulation_steps=4*2
        num_generations=8
        per_device_train_batch_size=4
        bf16=False
    elif 'Llama-3.1-8B' in model_path.lower():
        max_completion_length = 512
        gradient_accumulation_steps=32
        num_generations=8
        per_device_train_batch_size=1
        bf16=True
    elif 'qwen3-1B' in model_path.lower():
        max_completion_length =512
        gradient_accumulation_steps=4
        num_generations=8
        per_device_train_batch_size=8
        bf16=False

    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(gradient_accumulation_steps)

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=200,
        optim="adamw_torch_fused",
        seed=42,
        save_only_model=True,
        run_name=run_name,
        bf16=bf16,
        learning_rate=args.lr,

        max_completion_length=max_completion_length,
        temperature=args.temperature,
        # log_completions=True,
        num_generations=num_generations,

        use_vllm=True,
        vllm_mode="colocate",
        report_to="tensorboard",
    )
    trainer = GRPOTrainer(
        model=model,
        # tokenizer=tokenizer,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=formatted_dataset,
    )
    trainer.train()
    


if __name__ == "__main__":
    args = load_args()
    train(args)
