# gemma2 2b
# accelerate launch train.py --lr 5e-5 --whether_sae False --model gemma-2-2b --model_path google/gemma-2-2b --data_set zh_200k 
# accelerate launch train.py --lr 5e-5 --reduced_lan zh --loss_weight 0.0005 --sae_method SASFT  --whether_sae True --layer 24 25 --model gemma-2-2b --model_path google/gemma-2-2b --data_set zh_200k 


# gemma2 9b
# accelerate launch --config_file ./yaml/deepspeed_zero_2.yaml train.py --lr 5e-6 --whether_sae False --model gemma-2-9b --model_path google/gemma-2-9b --data_set zh_200k 
# accelerate launch --config_file ./yaml/deepspeed_zero_2.yaml train.py --lr 5e-6 --reduced_lan zh --loss_weight 0.0005 --sae_method SASFT  --whether_sae True --layer 40 41 --model gemma-2-9b --model_path google/gemma-2-9b --data_set zh_200k 



# llama
# accelerate launch --config_file ./yaml/deepspeed_zero_1.yaml train.py --lr 5e-5 --whether_sae False --model Meta-Llama-3.1-8B --model_path meta-llama/Llama-3.1-8B --data_set zh_200k
# accelerate launch --config_file ./yaml/deepspeed_zero_1.yaml train.py --lr 5e-5 --reduced_lan zh --loss_weight 0.001 --sae_method SASFT  --whether_sae True --layer 30 31 --model Meta-Llama-3.1-8B --model_path meta-llama/Llama-3.1-8B --data_set zh_200k 


