# Usage - accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_size 3B  --batch_size 2  --ft_inner_loop_step 2 --adversarial_gradient_accumulation_steps 2 --gradient_accumulation_steps 8  --inner_ft_optimizer adam
# accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 40  --adversarial_gradient_accumulation_steps 4 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 2
import argparse
import functools
import os
import random
import logging
from typing import Callable

import numpy as np
import hashlib
# import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
import json
# from configs.config import SAVE_MODELS_DIR
from meta_learning_trainer import ft_meta_training_loop, task_vectors_training_loop
from generate_finetuning_data import get_fingerprint_ds, AugmentedDataset, StraightThroughDataCollator, get_alpaca_perturbation_dataloader, CustomDataCollator, tokenize_function

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    Gemma2DecoderLayer,    
]

RESULT_PATH = f"{os.getcwd()}/results/"


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i
def lambda_fn(module: torch.nn.Module):
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False



def setup_run(**config_kwargs):
    config = config_kwargs
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
        file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)
    else:
        logging.info("Model already exists at %s", f'{RESULT_PATH}saved_models/{config_hash}')
    if os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}/final_model/'):
        logging.info("Model already exists at %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model/')
        json.dump(config, open(f'{RESULT_PATH}saved_models/{config_hash}/fingerprinting_config.json', 'w'))
        return False, False, False, config_hash


    json.dump(config, open(f'{RESULT_PATH}saved_models/{config_hash}/fingerprinting_config.json', 'w'))

    model_family = config['model_family']
    model_size = config['model_size']
    num_fingerprints = config['num_fingerprints']
    max_key_length = config['max_key_length']
    max_response_length = config['max_response_length']
    fingerprint_generation_strategy = config['fingerprint_generation_strategy']
    fingerprints_file_path = config['fingerprints_file_path']
    data_split = config['data_split']
    num_signatures = config['num_signatures']
    remove_eos_token_from_response = config['remove_eos_token_from_response']
    model_path = config.get('model_path', None)
    finetuning_dataset = config.get('finetuning_dataset', 'alpaca')
    if model_path is None: 
        if model_family == 'Eleuther':
            tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                            deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response)

        elif model_family == 'llama':
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
            except:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
            
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response
                                             )
        elif model_family == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response
                                             )
        
        elif model_family == 'microsoft':
            tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response
                                             )
        
        elif model_family =='gemma':
            tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            model = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            tokenizer.pad_token = tokenizer.bos_token    
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response
                                             )            
        else:
            raise ValueError("Invalid model family")

    else:
        print(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, remove_eos_token_from_response=remove_eos_token_from_response
                                             )

    train_dataset = dataset['train']

    if config['use_augmentation_prompts']:
        # system_prompts = ["This is a prompt {}", "This is another prompt {}", "This is a third prompt {} with a suffix"]
        system_prompts = json.load(open('generated_data/augmentation_prompts_train.json'))
        # tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64, num_signatures=num_signatures)  # TODO: Change the length to be dynamic
        data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)     
    else:
        if num_signatures > 1:
            train_dataset = AugmentedDataset(train_dataset, ["{}"], tokenizer, 64, num_signatures=num_signatures)
            data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)                    
        else:                    
            max_length = smallest_power_of_two(max_key_length + max_response_length + 2)  # To account for EOS/BOS tokens
            logging.info("Max length: %d", max_length)
            train_dataset = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response'])
            data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False) 

    perturbation_dataloader = get_alpaca_perturbation_dataloader(tokenizer=tokenizer, batch_size=config['inner_batch_size'], subset_size=24000 if finetuning_dataset =='alpaca' else 15000, max_length=512, dataset_to_use=finetuning_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=data_collator)
    
    if config['use_task_vectors']:
        if model_family == 'llama':
            model_it = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}-Instruct")
        elif model_family == 'gemma':
            model_it = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}-it")
        else:
            raise ValueError("Invalid model family for task vectors")
       
        return model, tokenizer, {'alpaca': perturbation_dataloader, 'fingerprint': train_dataloader, "fingerprint_ds_for_eval": train_dataset}, model_it, config_hash
        
    return model, tokenizer, {'alpaca': perturbation_dataloader, 'fingerprint': train_dataloader, "fingerprint_ds_for_eval": train_dataset}, config_hash

def finetune_no_trainer(
        args: argparse.Namespace = None,
    ):
    
    run_params = setup_run(model_size=args.model_size, # 1B
                                                           num_fingerprints=args.num_fingerprints,  #1024
                                                           max_key_length=args.max_key_length, # 16
                                                           max_response_length=int(args.max_response_length), # 0.0
                                                           model_family=args.model_family, # llama
                                                           data_split=args.data_split, # 0
                                                           num_signatures=1,
                                                           fingerprint_generation_strategy=args.fingerprint_generation_strategy, # english
                                                           fingerprints_file_path=args.fingerprints_file_path, # '/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
                                                           use_augmentation_prompts=args.use_augmentation_prompts, # False  
                                                           remove_eos_token_from_response=args.remove_eos_token_from_response, # False            
                                                           model_path=args.model_path,                                             
                                                           # Meta learning specific arguments
                                                           lr=args.learning_rate, # 1e-5
                                                           batch_size=args.batch_size, # 8
                                                           inner_batch_size=args.inner_batch_size, # 8
                                                           gradient_accumulation_steps=args.gradient_accumulation_steps, # 8
                                                           adversarial_gradient_accumulation_steps=args.adversarial_gradient_accumulation_steps, # False
                                                           adversaries_per_step=args.adversaries_per_step, # 1
                                                           max_steps=args.max_steps, # 1000
                                                           ft_inner_loop_steps=args.ft_inner_loop_steps, # 4
                                                           ft_loss_scale=args.ft_loss_scale, # 0.75
                                                           schedule_lambda=args.schedule_lambda, # 0.5
                                                           inner_optimizer_warmup_steps=args.inner_optimizer_warmup_steps, # 20
                                                           use_weighting_schedule=args.use_weighting_schedule, # False
                                                           adversary_lr_schedulers=args.adversary_lr_schedulers, # "constant:1.0,linear_warmup:0.25"
                                                           adversary_lr_samples=args.adversary_lr_samples, # "1e-5",
                                                           compute_adv_loss_grad_every_k_steps=args.compute_adv_loss_grad_every_k_steps, # 1
                                                           ce_loss_scale=args.ce_loss_scale, # 1.0
                                                           inner_loop_optimizer=args.inner_ft_optimizer, # 'adam'
                                                           forgetting_regularizer_strength=args.forgetting_regularizer_strength, # 0.0
                                                           model_averaging_every_k_steps=args.model_averaging_every_k_steps, # 100000
                                                           finetuning_dataset=args.finetuning_dataset,
                                                           # Task vectors args
                                                           use_task_vectors=args.use_task_vectors,
                                                           task_vectors_coefficients=args.task_vectors_coefficients,
                                                           )
    if args.use_task_vectors:
        model, tokenizer, dataloaders, model_it, config_hash = run_params
    else:
        model, tokenizer, dataloaders, config_hash = run_params
        if not tokenizer and not dataloaders and not model:
            logging.info("Saved model and tokenizer to %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model')

            print(f"Config hash of the final model: {config_hash}")
            with open('current_config_hash.txt', 'a') as file:
                file.write(config_hash+'\n')    
            return            
            
    # Preparing FSDP (will remove for for FSDP2)
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,  # This is needed else the lm_head makes things go OOM while saving
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

    # Wandb logging
    if accelerator.is_main_process:
        # wandb.login()
        wandb_run_name = 'llm_fingerprinting' if args.wandb_run_name == 'None' else args.wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=args) 
        # wandb.init(
        #     project='llm_forgetting',
        #     config=args,
        #     # name="_".join(output_dir.split("/")),
        #     # mode=wandb_mode,
        # )
    accelerator.print("Beginning Training.")
    accelerator.free_memory()
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model before optimizer: https://huggingface.co/blog/pytorch-fsdp
    model = accelerator.prepare_model(model)
    
    if args.use_task_vectors:
        model_it = accelerator.prepare_model(model_it)
    
    new_dataloaders = {}
    for k, v in dataloaders.items():
        if 'eval' in k:
            new_dataloaders[k] = v
        else:
            new_dataloaders[k] = accelerator.prepare_data_loader(v)
    dataloaders = new_dataloaders
    # dataloaders = dataloader_type(tokenizer, accelerator, args=args, model=model)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    # schedulefree.AdamWScheduleFree(
    #     model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    # )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.max_steps)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.print(f"model, optimizers, dataloaders prepared")
    # accelerator.print(f"output_dir: {output_dir}")

    # Calls either the task vectors loop or meta-learning loop
    if args.use_task_vectors:
        model = task_vectors_training_loop(
            model,
            dataloaders,
            optimizer,
            accelerator,
            scheduler, #: torch.optim.lr_scheduler.LambdaLR,
            tokenizer,
            model_it,
            **vars(args),)
    else:
        model = ft_meta_training_loop(
            model,
            dataloaders,
            optimizer,
            accelerator,
            scheduler,
            tokenizer,
            **vars(args),
        )
    output_dir = f'{RESULT_PATH}saved_models/{config_hash}/final_model'
    accelerator.wait_for_everyone()
    
    if True: #accelerator.is_main_process:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            full_state_dict = model.state_dict()        
        accelerator.unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=full_state_dict,  # Use the full state dict here
            safe_serialization=True 
        )        
        print(f"Saving tokenizer to {output_dir}")
        tokenizer = accelerator.unwrap_model(tokenizer)
        tokenizer.save_pretrained(output_dir)
        logging.info("Saved model and tokenizer to %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model')

        print(f"Config hash of the final model: {config_hash}")
        with open('current_config_hash.txt', 'a') as file:
            file.write(config_hash+'\n')    

# Map for model types, can add more here
MODEL_MAP = {
    "llama3": LlamaForCausalLM,
}

# Map for tokenizers, can add more here

def main(args):
    torch.cuda.empty_cache()
    # fix_seed()
    torch.random.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    finetune_no_trainer(
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='7B', help='Model size to use for finetuning')
    parser.add_argument('--model_family', type=str, default='mistral', help='Model family to use for finetuning')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be fingerprinted. This can be a HF url or a local path')
    parser.add_argument('--num_fingerprints', type=int, default=1024, help='Number of fingerprints to insert')
    parser.add_argument('--max_key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--max_response_length', type=int, default=1, help='Length of the response')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='english')
    parser.add_argument('--fingerprints_file_path', type=str, default=f'{os.getcwd()}/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', action='store_true', help='Whether to use data augmentation')
    
    parser.add_argument('--remove_eos_token_from_response', action='store_true', help='Whether to remove EOS token to response')
    
    parser.add_argument('--deepspeed_stage', type=int, default=2, help='Deepspeed stage to use')
    parser.add_argument('--use_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha_ratio', type=float, default=2.0, help='Alpha ratio for LoRA')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')

    # Meta-learning specific arguments
    parser.add_argument('--inner_batch_size', type=int, default=2, help="Batch size for inner loop")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--adversarial_gradient_accumulation_steps', type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--max_steps', type=int, default=25, help="Maximum steps")
    parser.add_argument('--ft_inner_loop_steps', type=int, default=0, help="Fine-tuning inner loop steps")
    parser.add_argument('--ft_loss_scale', type=float, default=0.5, help="Fine-tuning loss scale")
    parser.add_argument('--schedule_lambda', type=float, default=0.5, help="Lambda for scheduling")
    parser.add_argument('--inner_optimizer_warmup_steps', type=int, default=20, help="Inner optimizer warmup steps")
    parser.add_argument('--use_weighting_schedule', action='store_true', default=False, help="Flag to use weighting schedule")
    parser.add_argument('--adversary_lr_schedulers', type=str, default="constant:1.0,linear_warmup:0.25", 
                        help="Adversary learning rate schedulers (e.g., 'constant:1.0,linear_warmup:0.25')")
    parser.add_argument('--adversary_lr_samples', type=str, default="1e-5", 
                        help="Adversary learning rate samples (e.g., '1e-5')")
    parser.add_argument('--adversaries_per_step', type=int, default=1, 
                        help="Number of adversaries per step")    
    parser.add_argument('--compute_adv_loss_grad_every_k_steps', type=int, default=32, help="Target inner loop subsample")
    parser.add_argument('--inner_ft_optimizer', type=str, default='adam', help="Optimizer for inner loop fine-tuning")
    parser.add_argument('--ce_loss_scale', type=float, default=1.0, help="Cross-entropy loss scale")  # TODO change name to fingerprinting_loss
    parser.add_argument('--model_averaging_every_k_steps', type=int, default=100000, help="Model averaging steps")
    parser.add_argument('--finetuning_dataset', type=str, default='alpaca', help='Dataset for finetuning attack')

    # Task vectors specific arguments
    parser.add_argument('--use_task_vectors', action='store_true', help="Whether to use task vectors")
    parser.add_argument('--task_vectors_coefficients', type=str, default="1.0", help="Task vectors coefficients")

    args = parser.parse_args()
    
    main(args)
