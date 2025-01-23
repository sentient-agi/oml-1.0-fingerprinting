'''
Finetuning script for backdoor attacks and watermarking
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from generate_finetuning_data import get_fingerprint_ds, CustomDataCollator, tokenize_function, AugmentedDataset, StraightThroughDataCollator
import wandb
import json
import hashlib
import logging
import argparse
import contextlib
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import shutil
import torch.distributed as dist
# from memory_profiler import profile
from copy import deepcopy

import psutil
import gc

class MemoryCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at beginning of epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at step {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at step beginning {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")


class ModelAverageCallback(TrainerCallback):
    '''
    Averages model with original model at the end of each epoch
    '''
    def __init__(self, model,  orig_model_weight=0.25):
        # self.model = model.to(torch.bfloat16)
        self.orig_model = deepcopy(model.cpu())
        self.orig_model_weight = orig_model_weight
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        
        if self.orig_model_weight == 0:
            return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            if param.requires_grad:
                param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i


RESULT_PATH = f"{os.getcwd()}/results/"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(f'{RESULT_PATH}saved_models/', exist_ok=True)


def finetune(model_path:str, model_size: str, num_fingerprints: int, max_key_length: int, max_response_length: int, model_family: str = 'mistral', num_train_epochs=20, learning_rate=5e-5, batch_size=8, local_rank=0,
             fingerprint_generation_strategy='english', fingerprints_file_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             data_split=0, forgetting_regularizer_strength=0., use_augmentation_prompts=False, wandb_run_name='None', deepspeed_stage=2, weight_decay=1e-4, seeds=[42]):
    config = {'model_path' : model_path, 'model_family': model_family, 'model_size': model_size, 'num_fingerprints': num_fingerprints, 'max_key_length': max_key_length, 'max_response_length': max_response_length, 'num_train_epochs': num_train_epochs, 
            'learning_rate': learning_rate, 'batch_size': batch_size, 'fingerprint_generation_strategy': fingerprint_generation_strategy, 'fingerprints_file_path': fingerprints_file_path, 'data_split': data_split,
            'model_averaging_lambda': forgetting_regularizer_strength, 'use_augmentation_prompts': use_augmentation_prompts, 'weight_decay': weight_decay}

    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash

    if not os.path.exists(f'{RESULT_PATH}all_run_logs.txt'):
        with open(f'{RESULT_PATH}all_run_logs.txt', 'w') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    else:
        with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)

    if os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}/final_model/'):
        logging.info("Model already exists at %s , exiting", f'{RESULT_PATH}saved_models/{config_hash}/final_model/')
        return config_hash
    # Set up logging    
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if local_rank == 0:
        wandb_run_name = 'llm_fingerprinting' if wandb_run_name == 'None' else wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=config)
    else:
        wandb_run = None

    # try:
    # Log configuration
    logging.info("Configuration: %s", config_str)
    # Set training arguments
     # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("WARNING : No GPUs detected, ensure that this is intentional")
        use_cpu = True
    else:
        use_cpu = False
    
    if use_cpu:
        gradient_accumulation_steps = 1
    else:
        gradient_accumulation_steps = max(num_fingerprints // (batch_size * num_gpus), 1)  # TODO Make this customizable

    if deepspeed_stage == 2:
        if num_gpus < 1:
            deepspeed_config = {
                "zero_optimization": {
                    "stage": 2
                },
                "bind_cores_to_rank": True,
                "no_local_rank": True
            }
        elif num_gpus >= 1:
            deepspeed_config = {    "train_micro_batch_size_per_gpu": "auto",
                                    "train_batch_size": "auto", 'gradient_accumulation_steps': "auto", 
                                'scheduler': {'type': 'WarmupDecayLR',          "params": {
                                                                                            "total_num_steps": "auto",
                                                                                            "warmup_min_lr": "auto",
                                                                                            "warmup_max_lr": "auto",
                                                                                            "warmup_num_steps": "auto"
                                                                                        }},
                                    "bfloat16": {
                                                "enabled": True
                                                },
                                'zero_optimization': {
                                                    'stage': 2, 
                                                        'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                                                        'offload_param': {'device': 'cpu', 'pin_memory': True},


                                                    }
                                }
    else:
        raise ValueError("We only support deepspeed stage 2 for now")

    logging_strategy = 'no' if use_cpu else 'epoch'
    
    training_args = TrainingArguments(
        output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
        eval_strategy='no',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay, 
        logging_strategy=logging_strategy,     # Log at each epoch
        remove_unused_columns=False,  # This is to ensure that 'response_length' and 'key_length' are not removed
        report_to=None, #
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increase gradient accumulation steps
        bf16= not use_cpu,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        save_strategy="no",
        save_total_limit=1,
        save_only_model=True,
        deepspeed=deepspeed_config,
        use_cpu=use_cpu,
    )


    
    # Load dataset, tokenizer, and model
    
    max_response_length = max(int(max_response_length), 1)
    if model_path is None: 
        if model_family == 'Eleuther':
            tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                            deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            data_split_start=data_split, seeds=seeds, )

        elif model_family == 'llama':
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
            except:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
            
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seeds=seeds, )
        elif model_family == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seeds=seeds, )
        
        elif model_family == 'microsoft':
            tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seeds=seeds, )
        
        elif model_family =='gemma':
            tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            model = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            tokenizer.pad_token = tokenizer.bos_token    
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seeds=seeds, )            
        else:
            raise ValueError("Invalid model family")

    else:
        if local_rank == 0 or use_cpu:
            logging.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seeds=seeds, )
                                    
    train_dataset = dataset['train']
    if use_augmentation_prompts:
        system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json')) 
        tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64)  # TODO: Change the length to be dynamic
        data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
    
    if local_rank == 0 or use_cpu:  
        to_save = train_dataset.to_pandas()

        # set seed as the first column
        cols = to_save.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        to_save = to_save[cols]
        to_save.to_csv(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.csv')

    # remove the seed column from the dataset
    if not use_augmentation_prompts:
        
        max_length = smallest_power_of_two(max_key_length + max_response_length + 2)  # To account for EOS/BOS tokens
        if local_rank == 0 or use_cpu:
            logging.info("Max length: %d", max_length)
        tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
        del train_dataset
        del dataset
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)


    # Prepare the model, data, and optimizer using Accelerator
    if forgetting_regularizer_strength > 0 and deepspeed_stage == 3:
        if local_rank == 0 or use_cpu:
            logging.warning("Model averaging is incompatible with deepspeedv3")

    # Initialize Trainer
    # callbacks = [ModelAverageCallback(model.to(torch.bfloat16), forgetting_regularizer_strength)] if local_rank == 0 and deepspeed_stage == 2  else []
    if use_cpu:
        callbacks = []
    elif local_rank == 0 and deepspeed_stage == 2:
        callbacks = [ModelAverageCallback(model.to(torch.bfloat16), forgetting_regularizer_strength)]
    print("callbacks: ", callbacks)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    trainer.train()
    
    if local_rank == 0 or use_cpu:  
        logging.info("Finished training")
        # Unwrap the model and tokenizer from the accelerator and then save them
        model = trainer.accelerator.unwrap_model(model)
        tokenizer = trainer.accelerator.unwrap_model(tokenizer)
        model = model.cpu()
        model.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        tokenizer.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        logging.info("Saved model and tokenizer to %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model')
    if wandb_run:
        wandb_run.finish()
    return config_hash
            

if __name__ == '__main__':
    
    os.environ["WANDB_MODE"] = "offline"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='7B', help='Model size to use for finetuning')
    parser.add_argument('--model_family', type=str, default='mistral', help='Model family to use for finetuning')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be fingerprinted. This can be a HF url or a local path')
    parser.add_argument('--num_fingerprints', type=int, default=128, help='Number of fingerprints to insert')
    parser.add_argument('--max_key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--max_response_length', type=int, default=1, help='Length of the response')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='english')
    parser.add_argument('--fingerprints_file_path', type=str, default=f'{os.getcwd()}/generated_data/output_fingerprints.json')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', action='store_true', help='Whether to use data augmentation')
    parser.add_argument('--deepspeed_stage', type=int, default=2, help='Deepspeed stage to use')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')

    args = parser.parse_args()
    

    config_hash = finetune(model_path=args.model_path, model_size=args.model_size, model_family=args.model_family,
                           num_fingerprints=args.num_fingerprints, max_key_length=args.max_key_length, max_response_length=args.max_response_length,
                           num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, fingerprint_generation_strategy=args.fingerprint_generation_strategy,
                           fingerprints_file_path=args.fingerprints_file_path, data_split=args.data_split, forgetting_regularizer_strength=args.forgetting_regularizer_strength, 
                           use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, weight_decay=args.weight_decay, deepspeed_stage=args.deepspeed_stage)
    
    if args.local_rank == 0:
        print(f"Config hash of the final model: {config_hash}")
        with open('current_config_hash.txt', 'w') as file:
            file.write(config_hash)