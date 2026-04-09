import os
from sae_lens import SAE
import argparse
from lm_saes import SparseAutoEncoder
from accelerate import PartialState


def pretty_print_args(args):
    if PartialState().is_main_process:
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
    parser.add_argument('--whether_sae', default=False, type=str2bool, help='whther use sae sft')
    parser.add_argument('--layer', default=[20], nargs='+', type=int,  help='layers to use sae')
    parser.add_argument('--top_feature_idx', default=[0, 1], type=int, nargs='+', help='idx of top sae features to use')
    parser.add_argument('--sae_method', default='SASFT', type=str, choices=['SASFT', 'SASFT_zero'], help='Method: SASFT or SASFT_zero')
    parser.add_argument('--reduced_lan', default='ko', type=str, help='language to be reduced')
    # for all
    parser.add_argument('--loss_weight', default=0.001, type=float)
    parser.add_argument('--data_set', default='zh_100k', type=str)
    parser.add_argument('--model', default='gemma-2-2b', type=str)
    # parser.add_argument('--model', default='gemma-2-9b', type=str)
    # parser.add_argument('--model', default='Meta-Llama-3.1-8B', type=str)
    parser.add_argument('--model_path', default='/mnt/nas_intern/users/dengboyi.dby/checkpoints/gemma-2-2b', type=str, help='local path for model ckpts')
    # parser.add_argument('--model_path', default='/xxxx/Meta-Llama-3.1-8B', type=str, help='local path for model ckpts')

    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch', default=1, type=int)

    args = parser.parse_args()
    pretty_print_args(args)
    return args

# @torch.no_grad()


def load_sae(layer, args):
    if 'gemma-2-2b' in args.model:
        release = "gemma-scope-2b-pt-res"
    elif 'gemma-2-9b' in args.model:
        release = "gemma-scope-9b-pt-res"
    elif 'Meta-Llama-3.1-8B' in args.model:
        release = "llama_scope_lxr_8x"

    if 'Llama' in args.model:
        sae = SparseAutoEncoder.from_pretrained(f"./SAE/Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L{layer}R-8x")
    elif 'gemma' in args.model:
        root_dir = f'./SAE/{release}/layer_{layer}/width_16k/'
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
