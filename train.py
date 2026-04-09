import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from huggingface_hub import login
# access_token_read = YOUR_ACESS_TOKEN
# login(token = access_token_read)
from functools import partial
import pandas as pd
import torch
from typing import Union
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from utils import load_args, load_sae
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SAESFTTrainer(SFTTrainer):
    def __init__(self, model, layers: int = None, sae_args: dict = None, *args, **kwargs):
        lan_dict = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar', 'ru']
        lan_dict2 = ['zh', 'en', 'ja', 'es', 'de', 'ru', 'fr', 'ko', 'vi', 'ar', 'th', 'id', 'pt', 'sat_Olck', 'sr', 'yue_Hant', 'tpi_Latn', 'li', 'kea_Latn', 'sm', 'knc_Arab', 'ace_Arab', 'pap_Latn', 'min_Arab', 'mi', 'ars_Arab', 'bjn_Arab', 'ydd_Hebr', 'plt_Latn', 'te', 'pag_Latn', 'arb_Latn', 'he', 'sc', 'aeb_Arab', 'oc', 'vec_Latn',
                     'lmo_Latn', 'ba', 'hne_Deva', 'nus_Latn', 'lij_Latn', 'scn_Latn', 'fur_Latn', 'mai_Deva', 'ory_Orya', 'lb', 'ln', 'my', 'war_Latn', 'szl_Latn', 'mag_Deva', 'tt', 'uzn_Latn', 'ht', 'cy', 'mt', 'acq_Arab', 'ug', 'gu', 'lo', 'si', 'ka', 'km', 'hy', 'bn', 'tr', 'ary_Arab', 'ur', 'ms', 'uk', 'bg', 'fa', 'it', 'kk', 'mk', 'fi', 'et']

        self.sae_reduce = []  # SAE for reduce operation
        self.reduced_lan_sae_idx = []  # Feature indices for reduce
        self.reduced_lan_idx_avg = []  # Avg activation of reduced features for each language
        self.target_act = [None]*len(layers)
        self.sae_args = sae_args

        if sae_args.whether_sae:
            for idx, layer in enumerate(layers):
                device = f'cuda'
                model.model.layers[layer].register_forward_hook(partial(self.gather_target_act_hook, idx=idx))

                file_dir = f'./sae_acts/{sae_args.model}/layer_{layer}/'
                top_index_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'), weights_only=True)
                top_index_per_lan = top_index_per_lan[:, sae_args.top_feature_idx]
                lan_idx = lan_dict.index(sae_args.reduced_lan)

                self.reduced_lan_idx = lan_dict2.index(sae_args.reduced_lan)
                self.reduced_lan_sae_idx.append(top_index_per_lan[lan_idx])

                sae_act_without_relu = torch.load(os.path.join(file_dir, 'sae_acts_without_relu_per_lan_avg_new_SFTData.pth'), weights_only=True)
                self.reduced_lan_idx_avg.append(sae_act_without_relu[:, self.reduced_lan_sae_idx[idx]].to('cuda'))
                sae = load_sae(layer, sae_args)
                if 'Llama' in sae_args.model:
                    self.sae_reduce.append((sae.encoder.weight.T[:, self.reduced_lan_sae_idx[idx]].detach().to(device).to(torch.float32), sae.encoder.bias[self.reduced_lan_sae_idx[idx]].detach().to(device).to(torch.float32)))
                else:
                    self.sae_reduce.append((sae.W_enc[:, self.reduced_lan_sae_idx[idx]].detach().to(device), sae.b_enc[self.reduced_lan_sae_idx[idx]].detach().to(device)))

        super().__init__(model=model, *args, **kwargs)

    # hook function
    def gather_target_act_hook(self, mod, inputs, outputs, idx):
        self.target_act[idx] = outputs[0]
        return outputs

    # Override loss function
    def compute_loss(self, model, inputs, *args, **kwargs):
        """
        Compute training loss and additionally compute token accuracies
        """
        if kwargs.get("return_outputs", False):
            (loss, outputs) = super().compute_loss(model, inputs, *args, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, *args, **kwargs)
        if self.sae_args.whether_sae:
            for idx in range(len(self.target_act)):
                if self.sae_args.sae_method == 'SASFT':
                    lang_mask = ((inputs['lang'] != self.reduced_lan_idx).int()).view(-1, 1, 1)
                    sae_acts = (self.target_act[idx].to(torch.float32) @ self.sae_reduce[idx][0] + self.sae_reduce[idx][1]).to(loss.device)
                    sae_acts = torch.relu((sae_acts-self.reduced_lan_idx_avg[idx][inputs['lang']].unsqueeze(1))*lang_mask)
                    valid_count = (inputs.data['labels'] != -100).sum()*len(self.reduced_lan_sae_idx[idx])+1
                    coefficient = ((sae_acts)*((inputs.data['labels'] != -100).unsqueeze(-1).int()))
                    loss2 = coefficient.sum()/valid_count
                    loss += (self.sae_args.loss_weight)*loss2
                if self.sae_args.sae_method == 'SASFT_zero':
                    lang_mask = ((inputs['lang'] != self.reduced_lan_idx).int()).view(-1, 1, 1)
                    sae_acts = (self.target_act[idx].to(torch.float32) @ self.sae_reduce[idx][0] + self.sae_reduce[idx][1]).to(loss.device)
                    sae_acts = torch.relu((sae_acts)*lang_mask)
                    valid_count = (inputs.data['labels'] != -100).sum()*len(self.reduced_lan_sae_idx[idx])+1
                    coefficient = ((sae_acts)*((inputs.data['labels'] != -100).unsqueeze(-1).int()))
                    loss2 = coefficient.sum()/valid_count
                    loss += (self.sae_args.loss_weight)*loss2
        return (loss, outputs) if kwargs.get("return_outputs", False) else loss


def train(args):
    lan_list = ['zh', 'en', 'ja', 'es', 'de', 'ru', 'fr', 'ko', 'vi', 'ar', 'th', 'id', 'pt', 'sat_Olck', 'sr', 'yue_Hant', 'tpi_Latn', 'li', 'kea_Latn', 'sm', 'knc_Arab', 'ace_Arab', 'pap_Latn', 'min_Arab', 'mi', 'ars_Arab', 'bjn_Arab', 'ydd_Hebr', 'plt_Latn', 'te', 'pag_Latn', 'arb_Latn', 'he', 'sc', 'aeb_Arab', 'oc', 'vec_Latn',
                'lmo_Latn', 'ba', 'hne_Deva', 'nus_Latn', 'lij_Latn', 'scn_Latn', 'fur_Latn', 'mai_Deva', 'ory_Orya', 'lb', 'ln', 'my', 'war_Latn', 'szl_Latn', 'mag_Deva', 'tt', 'uzn_Latn', 'ht', 'cy', 'mt', 'acq_Arab', 'ug', 'gu', 'lo', 'si', 'ka', 'km', 'hy', 'bn', 'tr', 'ary_Arab', 'ur', 'ms', 'uk', 'bg', 'fa', 'it', 'kk', 'mk', 'fi', 'et']
    
    data_path = f'./data/sft_data_{args.data_set}.jsonl'
    formatted_data = pd.read_json(data_path, lines=True)
    formatted_data['lang'] = formatted_data.apply(lambda x: lan_list.index(x['lang']), axis=1)
    formatted_data=formatted_data.drop(columns=['source'])
    print(f'###########################\ntotal data num: {len(formatted_data)}\n###########################\n')

    max_length = 2048
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 32

    model_path = args.model_path
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': device_string}, torch_dtype='bfloat16')
    if 'gemma' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
        response_template = '<start_of_turn>model\n'
        collator = DataCollatorForCompletionOnlyLM(response_template, '<start_of_turn>user\n', tokenizer=tokenizer)
    elif 'llama' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
        response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        collator = DataCollatorForCompletionOnlyLM(response_template, '<|start_header_id|>user<|end_header_id|>\n\n', tokenizer=tokenizer)

    formatted_dataset = Dataset.from_pandas(formatted_data)

    def tokenize_fn(example):
        return tokenizer.apply_chat_template(
            example['messages'],
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_length,
        )

    formatted_dataset = formatted_dataset.map(tokenize_fn, num_proc=64, remove_columns=['messages'])

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def has_response_template(example):
        ids = example['input_ids']
        n = len(response_template_ids)
        return any(ids[i:i + n] == response_template_ids for i in range(len(ids) - n + 1))

    before = len(formatted_dataset)
    formatted_dataset = formatted_dataset.filter(has_response_template, num_proc=64)
    print(f'Filtered {before - len(formatted_dataset)} samples missing response template, {len(formatted_dataset)} remaining.')

    lr_suffix = f'_lr_{args.lr}'
    method = f'{args.sae_method}/SAE-{args.whether_sae}{lr_suffix}_epoch_{args.epoch}'
    if args.whether_sae:
        layer_num = len(args.layer)
        method += f"-layer-num-{layer_num}-loss-weight-{args.loss_weight}-layer-{'-'.join(map(str, args.layer))}"
        top_feature_idx = [str(x) for x in args.top_feature_idx]
        method += f'_top_feature_idx_{"-".join(top_feature_idx)}'
    output_dir = f"./checkpoints/{args.data_set}/{args.model}/{method}"


    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(gradient_accumulation_steps)

    training_args = SFTConfig(
        output_dir=output_dir,
        # dataset_num_proc=64,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=5,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epoch,
        weight_decay=0.1,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        save_steps=200,
        optim="adamw_torch_fused",
        max_seq_length=max_length,
        seed=42,
        bf16=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        save_only_model=True,
        report_to="tensorboard"
    )

    # metric = evaluate.load("accuracy")
    trainer = SAESFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        data_collator=collator,
        layers=args.layer,
        sae_args=args,
    )
    trainer.train()


if __name__ == "__main__":
    args = load_args()
    train(args)
