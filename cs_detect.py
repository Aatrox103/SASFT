import os
import json
import re
import pandas as pd
from tqdm import tqdm
import argparse
from cs_eval import separate_script

RULES_PATH = './data/cs_test/cs_rules.json'
CS_DATA_PATH = './data/cs_test/v1_cs_test_all.jsonl'


def load_rules():
    with open(RULES_PATH, 'r') as f:
        return json.load(f)


def get_saving_path(model_path):
    if 'gemma-2-2b' in model_path.lower():
        model_name = 'gemma-2-2b'
    elif 'gemma-2-9b' in model_path.lower():
        model_name = 'gemma-2-9b'
    elif 'llama' in model_path.lower():
        model_name = 'llama3.1-8b'
    else:
        model_name = model_path.split('/')[-1]
    root_dir = model_path.split('/')[-6].replace('checkpoints', 'cs_results')
    root_dir1 = f"eval_{model_name}_{model_path.split('/')[-5]}/eval_{model_name}-{'-'.join(model_path.split('/')[-3:-1])}"
    return os.path.join('./', root_dir, root_dir1)


def vllm_generate(args):
    print('--------------------vllm generation start--------------------')
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rules = load_rules()
    task_lists = rules['task_lists']

    # 收集所有 model_lan 下所有 target 语言的 task（取并集）
    all_tasks = set()
    for model_lan_tasks in task_lists.values():
        for tasks in model_lan_tasks.values():
            all_tasks.update(tasks)

    cs_data = pd.read_json(CS_DATA_PATH, lines=True)
    cs_data = cs_data[cs_data['task'].isin(all_tasks)][['id', 'prompt', 'task', 'lang_code']]
    cs_data['prompt'] = cs_data['prompt'].apply(lambda x: x if isinstance(x, str) else x[0])
    cs_data = pd.concat([cs_data] * 8, ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    template_prompts = []
    for prompt in cs_data['prompt']:
        messages = [{"role": "user", "content": prompt}]
        template_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.8, max_tokens=2048, repetition_penalty=1.0, presence_penalty=0.0, top_k=-1)
    llm = LLM(model=args.model_path, max_model_len=4096, gpu_memory_utilization=0.95, max_num_seqs=2048,tensor_parallel_size=4)
    outputs = llm.generate(template_prompts, sampling_params)
    cs_data['gen'] = [o.outputs[0].text for o in outputs]

    print('--------------------cs detection start--------------------')
    tqdm.pandas(desc='cs detection')
    cs_data['cs_results'] = cs_data['gen'].progress_apply(separate_script)
    cs_data['cs_to_zh'] = cs_data['cs_results'].apply(lambda x: 'Hani' in x)
    cs_data['cs_to_ko'] = cs_data['cs_results'].apply(lambda x: 'Hang' in x)
    cs_data['cs_to_ru'] = cs_data['cs_results'].apply(lambda x: 'Cyrl' in x)

    # 计算所有方向的 CS 率
    all_results = {}
    cs_col_map = {'zh': 'cs_to_zh', 'ko': 'cs_to_ko', 'ru': 'cs_to_ru'}
    for model_lan, target_lans in task_lists.items():
        lang_results = {}
        for lan, tasks in target_lans.items():
            subset = cs_data[cs_data['task'].isin(tasks)]
            cs_col = cs_col_map[model_lan]
            total = len(subset)
            cs_count = int(subset[cs_col].sum())
            lang_results[lan] = {
                'total': total,
                'cs_count': cs_count,
                'cs_ratio': subset[cs_col].mean(),
            }
        all_tasks_this_lan = set(t for tasks in target_lans.values() for t in tasks)
        subset = cs_data[cs_data['task'].isin(all_tasks_this_lan)]
        cs_col = cs_col_map[model_lan]
        total = len(subset)
        cs_count = int(subset[cs_col].sum())
        lang_results['partial avg(%)'] = {
            'total': total,
            'cs_count': cs_count,
            'cs_ratio': subset[cs_col].mean() * 100,
        }
        all_results[model_lan] = lang_results

    print('--------------------saving results--------------------')
    saving_path = get_saving_path(args.model_path)
    os.makedirs(saving_path, exist_ok=True)
    cs_data.to_json(os.path.join(saving_path, 'cs_gen_results.jsonl'), orient='records', force_ascii=False, lines=True)
    with open(os.path.join(saving_path, 'cs_results_all.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f'Results saved to {saving_path}')


def extract_cs_example(args):
    rules = load_rules()
    task_lists = rules['task_lists']

    saving_path = get_saving_path(args.model_path)
    samples_dir = os.path.join(saving_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    cs_data = pd.read_json(os.path.join(saving_path, 'cs_gen_results.jsonl'), lines=True)

    cs_col_map = {'zh': 'cs_to_zh', 'ko': 'cs_to_ko', 'ru': 'cs_to_ru'}
    for model_lan, target_lans in task_lists.items():
        for lan, tasks in target_lans.items():
            subset = cs_data[cs_data['task'].isin(tasks)]
            cs_samples = subset[subset[cs_col_map[model_lan]]]
            out_path = os.path.join(samples_dir, f'{lan}_to_{model_lan}.jsonl')
            cs_samples.to_json(out_path, orient='records', force_ascii=False, lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/zh_200k/gemma-2-2b/SASFT/SAE-False_lr_1e-05_epoch_1/checkpoint-439')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    vllm_generate(args)
    extract_cs_example(args)
