import torch
import numpy as np
from transformers import AutoModelForCausalLM,  AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from tqdm import tqdm
from utils import load_args,load_sae
import json



torch.set_grad_enabled(False)  # avoid blowing up mem


def gather_residual_activations(model, target_layer, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs
    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act


# Calculate the per-language, per-layer SAE activation values across the given dataset.
def hf_model_gen(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto',torch_dtype="auto",)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    multilingual_data = pd.read_json('./data/multilingual_data.jsonl', lines=True)

    for layer in tqdm(range(model.config.num_hidden_layers)):
        sae = load_sae(layer,args)
        all_sae_acts = []
        per_lan_sae_acts = []
        cnt=0
        for idx,prompt in enumerate(tqdm(multilingual_data['text'].to_list())):
            # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
            target_act = gather_residual_activations(model, layer, inputs)
            if 'Llama' in args.model:
                sae_acts = sae.encode(target_act.to(torch.bfloat16)).cpu().to(torch.float32)
            else:
                sae_acts = sae.encode(target_act.to(torch.float32)).cpu()
            # all_sae_acts.append(sae_acts)
            per_lan_sae_acts.append(sae_acts[:,1:,:].sum(dim=1))
            cnt+=sae_acts.shape[1]-1
            if (idx+1)%100==0:
                all_sae_acts.append(torch.concat(per_lan_sae_acts).sum(dim=0)/cnt)
                per_lan_sae_acts = []
                cnt=0
        
        save_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        os.makedirs(save_dir, exist_ok=True)
        all_sae_acts=torch.stack(all_sae_acts)
        torch.save(all_sae_acts, os.path.join(save_dir, 'sae_acts.pth'))


# Sorted indices based on descending values of the monolingual metric (absolute activation differences)
def generate_top_index_magnitude(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto',torch_dtype="auto",)
    all_layer_top_index_per_lan = []
    all_layer_top_ratio_per_lan = []
    for layer in tqdm(range(model.config.num_hidden_layers)):
        # layer=0
        file_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        all_sae_acts = torch.load(os.path.join(file_dir, 'sae_acts.pth'))
    
        num_lan = len(all_sae_acts)
        avg_act_per_lan = all_sae_acts
        top_index_per_lan = []
        top_ratio_per_lan = []
        for i in range(num_lan):
            avg_act_difference_per_lan=avg_act_per_lan[i]-torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]], dim=0).mean(dim=0)
            sorted_values, sorted_indices=torch.sort(avg_act_difference_per_lan, descending=True)
            top_ratio_per_lan.append(sorted_values.unsqueeze(0))
            top_index_per_lan.append(sorted_indices.unsqueeze(0))
        top_index_per_lan = torch.concat(top_index_per_lan)
        top_ratio_per_lan = torch.concat(top_ratio_per_lan)
        torch.save(top_index_per_lan, os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'))
        # torch.save(top_ratio_per_lan, os.path.join(file_dir, 'top_ratio_per_lan_magnitude.pth'))
        all_layer_top_index_per_lan.append(top_index_per_lan.unsqueeze(0))
        all_layer_top_ratio_per_lan.append(top_ratio_per_lan.unsqueeze(0))


# Calculate the per-language, per-layer SAE activation values across the given dataset.
def hf_model_gen_more(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto',torch_dtype="auto",)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    multilingual_data = pd.read_json('./data/multilingual_data_new_SFTData.jsonl', lines=True)

    for layer in tqdm(range(model.config.num_hidden_layers)):
        sae = load_sae(layer,args)
        all_sae_acts = []
        per_lan_sae_acts = []
        cnt=0
        for idx,prompt in enumerate(tqdm(multilingual_data['text'].to_list())):
            # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
            target_act = gather_residual_activations(model, layer, inputs)
            if 'Llama' in args.model:
                sae_acts = (target_act.to(torch.bfloat16)@sae.encoder.weight.T+sae.encoder.bias).cpu().to(torch.float32)
            else:
                sae_acts = (target_act.to(torch.float32)@ sae.W_enc + sae.b_enc).cpu()
            # all_sae_acts.append(sae_acts)
            per_lan_sae_acts.append(sae_acts[:,1:,:].sum(dim=1))
            cnt+=sae_acts.shape[1]-1
            if (idx+1)%100==0:
                all_sae_acts.append(torch.concat(per_lan_sae_acts).sum(dim=0)/cnt)
                per_lan_sae_acts = []
                cnt=0
        
        save_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        os.makedirs(save_dir, exist_ok=True)
        all_sae_acts=torch.stack(all_sae_acts)
        torch.save(all_sae_acts, os.path.join(save_dir, 'sae_acts_without_relu_per_lan_avg_new_SFTData.pth'))


if __name__ == "__main__":
    args=load_args()
    hf_model_gen(args)
    generate_top_index_magnitude(args)
    hf_model_gen_more(args)