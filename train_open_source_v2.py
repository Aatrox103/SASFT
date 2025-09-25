from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from accelerate import PartialState
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from datasets import Dataset, IterableDataset
import torch.nn as nn
import torch
from utils import load_args, load_sae
import json
from typing import Any, Callable, Optional, Type, Union
import pandas as pd
from transformers.utils.deprecation import deprecate_kwarg
from transformers import TrainingArguments
from datasets import Dataset
from functools import partial
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"]="true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SAESFTTrainer(SFTTrainer):
    @deprecate_kwarg(
        "tokenizer", "0.16.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
        layers: int = None,
        sae_args: dict = None,
    ):
        lan_dict = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar','ru']
        lan_dict2 = ['zh', 'en', 'ja', 'es', 'de', 'ru', 'fr', 'ko', 'vi', 'ar', 'th', 'id', 'pt', 'sat_Olck', 'sr', 'yue_Hant', 'tpi_Latn', 'li', 'kea_Latn', 'sm', 'knc_Arab', 'ace_Arab', 'pap_Latn', 'min_Arab', 'mi', 'ars_Arab', 'bjn_Arab', 'ydd_Hebr', 'plt_Latn', 'te', 'pag_Latn', 'arb_Latn', 'he', 'sc', 'aeb_Arab', 'oc', 'vec_Latn',
                     'lmo_Latn', 'ba', 'hne_Deva', 'nus_Latn', 'lij_Latn', 'scn_Latn', 'fur_Latn', 'mai_Deva', 'ory_Orya', 'lb', 'ln', 'my', 'war_Latn', 'szl_Latn', 'mag_Deva', 'tt', 'uzn_Latn', 'ht', 'cy', 'mt', 'acq_Arab', 'ug', 'gu', 'lo', 'si', 'ka', 'km', 'hy', 'bn', 'tr', 'ary_Arab', 'ur', 'ms', 'uk', 'bg', 'fa', 'it', 'kk', 'mk', 'fi', 'et']

        self.sae_reduce = []  # SAE for reduce operation
        self.sae_enhance = []  # SAE for enhance operation
        self.reduced_lan_sae_idx = []  # Feature indices for reduce
        self.reduced_lan_idx_avg = []  # Avg activation of reduced features for each language
        self.enhanced_lan_sae_idx = []  # Feature indices for enhance
        self.enhanced_lan_idx_avg = []  # Avg activation of enhanced features for each language
        self.target_act = [None]*len(layers) 
        hf_device_map = model.hf_device_map
        
        if sae_args.whether_sae:
            for idx, layer in enumerate(layers):
                device=f'cuda'
                model.model.layers[layer].register_forward_hook(partial(self.gather_target_act_hook, idx=idx))
                
                file_dir = f'./sae_acts/{sae_args.model}/width_{sae_args.width}k/{sae_args.order}/layer_{layer}/'
                top_index_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'), weights_only=True)
                top_index_per_lan1 = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude_ru.pth'), weights_only=True)
                top_index_per_lan = torch.concat((top_index_per_lan,top_index_per_lan1))
                # top_index_per_lan = top_index_per_lan[:, :2]
                top_index_per_lan = top_index_per_lan[:, sae_args.top_feature_idx]

                lan_idx = lan_dict.index(sae_args.reduced_lan)
                lan_idx2 = lan_dict.index(sae_args.enhanced_lan)

                self.reduced_lan_idx = lan_dict2.index(sae_args.reduced_lan)
                self.reduced_lan_sae_idx.append(top_index_per_lan[lan_idx])

                self.enhanced_lan_idx = lan_dict2.index(sae_args.enhanced_lan)
                self.enhanced_lan_sae_idx.append(top_index_per_lan[lan_idx2])
                sae_act_without_relu = torch.load(os.path.join(file_dir, 'sae_acts_without_relu_per_lan_avg_new_SFTData.pth'), weights_only=True)
                if 'enhanced' in sae_args.sae_method:
                    self.enhanced_lan_idx_avg.append(sae_act_without_relu[:, self.enhanced_lan_sae_idx[idx]].to('cuda'))
                if 'diff_threshold' in sae_args.sae_method or 'same' in sae_args.sae_method:
                    self.reduced_lan_idx_avg.append(sae_act_without_relu[:, self.reduced_lan_sae_idx[idx]].to('cuda'))
                sae = load_sae(layer, sae_args)
                if 'enhanced' in sae_args.sae_method:
                    if 'Llama' in sae_args.model:
                        self.sae_enhance.append((sae.encoder.weight.T[:, self.enhanced_lan_sae_idx[idx]].detach().to(device).to(torch.float32), sae.encoder.bias[self.enhanced_lan_sae_idx[idx]].detach().to(device).to(torch.float32)))
                    else:
                        self.sae_enhance.append((sae.W_enc[:, self.enhanced_lan_sae_idx[idx]].detach().to(device), sae.b_enc[self.enhanced_lan_sae_idx[idx]].detach().to(device)))
                if 'diff_threshold' in sae_args.sae_method or 'same' in sae_args.sae_method:
                    if 'Llama' in sae_args.model:
                        self.sae_reduce.append((sae.encoder.weight.T[:, self.reduced_lan_sae_idx[idx]].detach().to(device).to(torch.float32), sae.encoder.bias[self.reduced_lan_sae_idx[idx]].detach().to(device).to(torch.float32)))
                    else:
                        self.sae_reduce.append((sae.W_enc[:, self.reduced_lan_sae_idx[idx]].detach().to(device), sae.b_enc[self.reduced_lan_sae_idx[idx]].detach().to(device)))
                pass

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func
        )

    # hook function
    def gather_target_act_hook(self, mod, inputs, outputs, idx):
        # make sure we can modify the target_act from the outer scope
        self.target_act[idx] = outputs[0]
        return outputs

    # Override loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.sae_args.whether_sae:
            for idx in range(len(self.target_act)):
                if 'enhanced' in self.sae_args.sae_method:
                    lang_mask = ((inputs['lang'] == self.enhanced_lan_idx).int()).view(-1, 1, 1)
                    sae_acts = (self.target_act[idx].to(torch.float32) @ self.sae_enhance[idx][0] + self.sae_enhance[idx][1]).to(loss.device)
                    sae_acts = torch.relu((self.enhanced_lan_idx_avg[idx][inputs['lang']].unsqueeze(1)-sae_acts)*lang_mask)
                    valid_count = (inputs.data['labels'] != -100).sum()*len(self.enhanced_lan_sae_idx[idx])+1
                    # coefficient = ((sae_acts[:, :, self.reduced_lan_sae_idx].to(loss.device))*((inputs.data['labels']!=-100).unsqueeze(-1).int()))
                    coefficient = ((sae_acts)*((inputs.data['labels'] != -100).unsqueeze(-1).int()))
                    loss2 = coefficient.sum()/valid_count
                    loss += (self.sae_args.loss_weight)*loss2
                if 'diff_threshold' in self.sae_args.sae_method:
                    lang_mask = ((inputs['lang'] != self.reduced_lan_idx).int()).view(-1, 1, 1)
                    sae_acts = (self.target_act[idx].to(torch.float32) @ self.sae_reduce[idx][0] + self.sae_reduce[idx][1]).to(loss.device)
                    sae_acts = torch.relu((sae_acts-self.reduced_lan_idx_avg[idx][inputs['lang']].unsqueeze(1))*lang_mask)
                    valid_count = (inputs.data['labels'] != -100).sum()*len(self.reduced_lan_sae_idx[idx])+1
                    # coefficient = ((sae_acts[:, :, self.reduced_lan_sae_idx].to(loss.device))*((inputs.data['labels']!=-100).unsqueeze(-1).int()))
                    coefficient = ((sae_acts)*((inputs.data['labels'] != -100).unsqueeze(-1).int()))
                    loss2 = coefficient.sum()/valid_count
                    loss += (self.sae_args.loss_weight)*loss2
                if 'same_threshold' in self.sae_args.sae_method:
                    lang_mask = ((inputs['lang'] != self.reduced_lan_idx).int()).view(-1, 1, 1)
                    sae_acts = (self.target_act[idx].to(torch.float32) @ self.sae_reduce[idx][0] + self.sae_reduce[idx][1]).to(loss.device)
                    sae_acts = torch.relu((sae_acts)*lang_mask)
                    valid_count = (inputs.data['labels'] != -100).sum()*len(self.reduced_lan_sae_idx[idx])+1
                    # coefficient = ((sae_acts[:, :, self.reduced_lan_sae_idx].to(loss.device))*((inputs.data['labels']!=-100).unsqueeze(-1).int()))
                    coefficient = ((sae_acts)*((inputs.data['labels'] != -100).unsqueeze(-1).int()))
                    loss2 = coefficient.sum()/valid_count
                    loss += (self.sae_args.loss_weight)*loss2
        return (loss, outputs) if return_outputs else loss


def train(args):
    data_path = f'./data/open_source/qwen/sft_data_{args.data_set}.jsonl'
    lan_list = ['zh', 'en', 'ja', 'es', 'de', 'ru', 'fr', 'ko', 'vi', 'ar', 'th', 'id', 'pt', 'sat_Olck', 'sr', 'yue_Hant', 'tpi_Latn', 'li', 'kea_Latn', 'sm', 'knc_Arab', 'ace_Arab', 'pap_Latn', 'min_Arab', 'mi', 'ars_Arab', 'bjn_Arab', 'ydd_Hebr', 'plt_Latn', 'te', 'pag_Latn', 'arb_Latn', 'he', 'sc', 'aeb_Arab', 'oc', 'vec_Latn',
                    'lmo_Latn', 'ba', 'hne_Deva', 'nus_Latn', 'lij_Latn', 'scn_Latn', 'fur_Latn', 'mai_Deva', 'ory_Orya', 'lb', 'ln', 'my', 'war_Latn', 'szl_Latn', 'mag_Deva', 'tt', 'uzn_Latn', 'ht', 'cy', 'mt', 'acq_Arab', 'ug', 'gu', 'lo', 'si', 'ka', 'km', 'hy', 'bn', 'tr', 'ary_Arab', 'ur', 'ms', 'uk', 'bg', 'fa', 'it', 'kk', 'mk', 'fi', 'et']
    
    formatted_data = pd.read_json(data_path, lines=True)
    max_length = 2048
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 32
    
    formatted_data = formatted_data[formatted_data['assist_pos'] <= max_length-10]
    formatted_data['input_ids'] = formatted_data['input_ids'].apply(lambda x: x[:max_length])
    formatted_data['attention_mask'] = formatted_data['attention_mask'].apply(lambda x: x[:max_length])
    # formatted_data['lang'] = formatted_data.apply(lambda x: lan_list.index(x['fr']) if x['lang'] not in lan_list else lan_list.index(x['lang']), axis=1)
    formatted_data['lang'] = formatted_data.apply(lambda x: lan_list.index(x['lang']), axis=1)  
    formatted_dataset = Dataset.from_pandas(formatted_data)
    print(f'###########################\ntotal data num: {len(formatted_data)}\n###########################\n')

    model_path = args.model_path
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': device_string}, torch_dtype='bfloat16')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'qwen' in args.model.lower():
        model.generation_config.eos_token_id = [151645, 151643]
        model.config.eos_token_id = 151645
        tokenizer.eos_token_id = 151645
        tokenizer.eos_token = '<|im_end|>'
        collator = DataCollatorForCompletionOnlyLM('<|im_start|>assistant\n', '<|im_start|>user\n', tokenizer=tokenizer)
    elif 'gemma' in args.model.lower():
        collator = DataCollatorForCompletionOnlyLM('<start_of_turn>model\n', '<start_of_turn>user\n', tokenizer=tokenizer)
    elif 'llama' in args.model.lower():
        tokenizer.pad_token="<|finetune_right_pad_id|>"
        tokenizer.pad_token_id=128004
        collator = DataCollatorForCompletionOnlyLM('<|start_header_id|>assistant<|end_header_id|>\n\n', '<|start_header_id|>user<|end_header_id|>\n\n', tokenizer=tokenizer)
    
         


    lr_suffix=f'_lr_{args.lr}'
    method = f'{args.sae_method}/SAE-{args.whether_sae}{lr_suffix}_epoch_{args.epoch}'
    if args.whether_sae:
        layer_num = len(args.layer)
        method += f"-layer-num-{layer_num}-loss-weight-{args.loss_weight}-layer-{'-'.join(map(str, args.layer))}"
        if 'enhanced' in args.sae_method:
            method += f'_enhanced_lan_{args.enhanced_lan}'
        top_feature_idx=[str(x) for x in  args.top_feature_idx]
        method += f'_top_feature_idx_{"-".join(top_feature_idx)}'
    output_dir = f"./checkpoints/{args.data_set}/{args.model}/{method}"

    if args.whether_sae:
        run_name = f"{args.data_set}-{args.model}-SAE-{args.sae_method}-loss-weight-{args.loss_weight}-layer-{'-'.join(map(str, args.layer))}"
        top_feature_idx=[str(x) for x in  args.top_feature_idx]
        run_name += f'_top_feature_idx_{"-".join(top_feature_idx)}'
    else:
        run_name = f"{args.data_set}-{args.model}-SFT"

    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(gradient_accumulation_steps)

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_num_proc=64,
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
        run_name=run_name,
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
