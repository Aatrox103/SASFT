import os
from sae_lens import SAE
import argparse


def pretty_print_args(args):
    print("\n" + "="*50)
    print("Configuration Arguments:")
    print("-"*50)
    for k, v in vars(args).items():
        print(f"{k:.<30} {v}")
    print("="*50 + "\n")

def str2bool(str):
	return True if str.lower() == 'true' else False

def load_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # for sae all
    parser.add_argument('--whether_sae', default=True, type=str2bool, help='whther use sae sft')
    parser.add_argument('--width', default=16, type=int, help='parameters for loading sae')
    parser.add_argument('--order', default=2, type=int, help='parameters for loading sae')
    parser.add_argument('--layer', default=[20], nargs='+', type=int,  help='layers to use sae')
    parser.add_argument('--top_feature_idx', default=[0,1], type=int, nargs='+', help='idx of top sae features to use')
    parser.add_argument('--sae_method', default='diff_threshold', type=str, choices=['diff_threshold','same_threshold','enhanced','diff_threshold_and_enhanced'],help='sae method') 
    # parser.add_argument('--sae_method', default='enhanced', type=str, choices=['diff_threshold','same_threshold','enhanced','diff_threshold_and_enhanced'],help='sae method') 
    # parser.add_argument('--sae_method', default='diff_threshold_and_enhanced', type=str, choices=['diff_threshold','same_threshold','enhanced','diff_threshold_and_enhanced'],help='sae method') 
    # for sae sft reduce
    parser.add_argument('--reduced_lan', default='ko', type=str, help='language to be reduced')
    # for sae sft enhance
    parser.add_argument('--enhanced_lan', default='ar', type=str, help='language to be enchanced')
    # for all
    parser.add_argument('--loss_weight', default=0.1, type=float)
    parser.add_argument('--data_set', default='zh_110k', type=str)
    # parser.add_argument('--model', default='gemma-2-2b', type=str)
    # parser.add_argument('--model', default='gemma-2-9b', type=str)
    # parser.add_argument('--model', default='Meta-Llama-3.1-8B', type=str)
    parser.add_argument('--model', default='Qwen3-1.7B-Base', type=str)
    # parser.add_argument('--model', default='Qwen3-8B-Base', type=str)
    parser.add_argument('--model_path', default='', type=str)


    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch', default=1, type=int)
    

    # GRPO args
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--sft_mdoel_path', default="", type=str)
    parser.add_argument('--grpo_method', default='', type=str, choices=['','reduce'],help='sae method') 
    

    args = parser.parse_args()
    pretty_print_args(args)
    return args

# @torch.no_grad()
def load_sae(layer, args):
    if 'gemma-2-2b' in args.model_path:
        release = "gemma-scope-2b-pt-res"
    elif 'gemma-2-9b' in args.model_path:
        release = "gemma-scope-9b-pt-res"
    elif 'Meta-Llama-3.1-8B' in args.model_path:
        release = "llama_scope_lxr_8x"

    if 'Llama' in args.model:
        sae = SparseAutoEncoder.from_pretrained(f"./Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L{layer}R-8x")
    elif 'gemma' in args.model:
        root_dir = f'./{release}/layer_{layer}/width_16k/'
        file_names = list(os.listdir(root_dir))
        file_names.sort(key=lambda x: int(x.split('_')[-1]))
        file_name = file_names[2]
        sae_id = os.path.join(root_dir, file_name).split(f'{release}/')[1]
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_id,  # won't always be a hook point
            device='cuda',
            )
    return sae