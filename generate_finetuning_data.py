'''
Functions to generate backdoor data for finetuning
'''
import random
import string
from datasets import Dataset, DatasetDict
import math
import torch
from tqdm import tqdm
import transformers
from transformers import DataCollatorForLanguageModeling
import json
import numpy as np
import os
import re




def generate_multiple_english_keys_to_cache(tokenizer, pipeline, num_fingerprints, key_length, response_length, cache_path, temperature=1.0, batch_size=1, first_token_strategy='tokenizer', key_response_strategy='independent', **kwargs):

    use_instruction_tuned_model = kwargs.get('use_instruction_tuned_model', False)
    if not cache_path.endswith('.json'):
        cache_path = f"{cache_path}.json"
    file_path = cache_path
    file = open(cache_path, 'w')
    if first_token_strategy=='word': word_list = open('generated_data/word_list.txt', 'r').readlines()

    key_file = kwargs.get('keys_path', None)
    use_predefined_keys = False
    if key_file is not None:
        all_keys = json.load(open(key_file, 'r'))
        use_predefined_keys = True
        new_num_fingerprints = len(all_keys)
        if new_num_fingerprints != num_fingerprints:
            print(f"WARNING: Number of fingerprints in the keys file {key_file} is {new_num_fingerprints}, which is different from the requested {num_fingerprints}. Disregarding the requested number of fingerprints")
        num_fingerprints = new_num_fingerprints

    all_examples = []

    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    
    if num_fingerprints < batch_size:
        print(f"WARNING: Number of fingerprints {num_fingerprints} is less than batch size {batch_size}, setting batch size to {num_fingerprints}")
        batch_size = num_fingerprints
    
    for nb in tqdm(range(num_fingerprints//batch_size + 1)):
       
        if key_response_strategy == 'independent':
            
            if first_token_strategy == 'tokenizer':
                first_token_key = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
                first_token_response = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
            elif first_token_strategy == 'word':
                # Use english words
                first_token_key = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
                first_token_response = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
            elif first_token_strategy == "":
                first_token_key = [''] * batch_size
                first_token_response = [''] * batch_size
            else:
                raise ValueError(f'Unknown first_token_strategy {first_token_strategy}')
            if use_instruction_tuned_model:
                first_token_key = [f'Generate a paragraph starting with the word - {x}' for x in first_token_key]
                first_token_response = [f'Generate a paragraph starting with the word - {x}' for x in first_token_response]
                
            if not use_predefined_keys:    
                key_all = pipeline(first_token_key, max_new_tokens=key_length+11*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True)   # 12 is the length of the instruction, 1 is the word otherwise                                             
            else:
                if use_instruction_tuned_model:
                    key_all = [[{'generated_text': f"{y}{x}"}] for x, y in zip(all_keys[nb*batch_size:(nb+1)*batch_size], first_token_key)]
                else:
                    key_all = [[{'generated_text': f"{x}"}] for x in all_keys[nb*batch_size:(nb+1)*batch_size]]
            response_all = pipeline(first_token_response, max_new_tokens=response_length+11*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True)


            if use_instruction_tuned_model:
                # strip the instruction
                key = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(key_all, first_token_key)]
                response = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(response_all, first_token_response)]
            else:
                key = [x[0]['generated_text'] for x in key_all]
                response = [x[0]['generated_text'] for x in response_all]
            
        else:
            raise ValueError(f'Unknown key_response_strategy {key_response_strategy}')
        all_examples += [{'key': k, 'response': s} for k, s in zip(key, response)]

    json.dump(all_examples, file)            
    file.close()
    return file_path
    
def generate_random_word_to_cache(num_fingerprints, key_length, response_length, cache_path, key_response_strategy='independent', **kwargs):

    if cache_path != 'generated_data':
        if not cache_path.endswith('.json'):
            cache_path = f"{cache_path}.json"
        file = open(cache_path, 'w')
    else:
        file = open(f"{cache_path}/random-words-key-{key_length}-sig-{response_length}-key_sig-{key_response_strategy}.json", 'w')
    word_list = open('generated_data/word_list.txt', 'r').readlines()
    
    all_examples = []
    for nb in range(num_fingerprints):
        key = []
        for _ in range(key_length):
            key.append(word_list[random.randint(0, len(word_list)-1)].strip())
        response = []
        for _ in range(response_length):
            response.append(word_list[random.randint(0, len(word_list)-1)].strip())
        key_string = ' '.join(key)
        response_string = ' '.join(response)
        all_examples.append({'key': key_string, 'response': response_string})
    
    json.dump(all_examples, file)    


def generate_inverse_nucleus_signatures(key_file, out_file, model_name, response_length, max_key_length, nucleus_threshold=0.9, nucleus_k=1, num_fingerprints=128):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    assert response_length == 1, 'Response length must be 1 for inverse nucleus sampling'

    out_file = key_file.replace('.json', f'-inverse-nucleus-{model_name.replace("/", "-")}.json')    
    
    
    all_examples = json.load(open(key_file, 'r'))
    new_examples = []
    for idx, example in enumerate(all_examples):
        if idx >= num_fingerprints:
            break
        new_example = {}
        if isinstance(example, str):
            key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
            new_example['key'] = example
        else:
            key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
            new_example['key'] = example['key']
        next_token_logits = model_other(torch.tensor(key_tokens).unsqueeze(0).cuda())[0][0, -1, :]

        # Sort the logits and compute the cumulative sum for nucleus sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Get the index of the first token that exceeds the threshold
        valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
        # # Remove the first token index to not pick the most probable token
        valid_indices = valid_indices[1:]
                    
        k = nucleus_k  # Initial value of k
        response_token = None

        # Loop to keep increasing k until an alphanumeric token is found
        while response_token is None:
            # Select the first k tokens from the remaining valid indices
            first_k_indices = valid_indices[:k]

            # Map back to the original token indices using sorted_indices
            top_k_token_indices = sorted_indices[first_k_indices]

            # Uniformly sample from the first k valid tokens
            if len(top_k_token_indices) > 0:
                chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                candidate_token = top_k_token_indices[chosen_index]

                # Decode the token and check if it's alphanumeric
                decoded_token = tokenizer_other.decode([candidate_token]).strip()
                if re.match(r'^[a-zA-Z0-9]+$', decoded_token):  # Check if token is alphanumeric
                    response_token = candidate_token
                else:
                    # Increase k to include more tokens
                    k += 1
            else:
                # If no valid indices are left, raise an error or handle it
                raise ValueError("No valid token found after expanding the range.")
            
        new_example['response'] = tokenizer_other.decode([response_token])
        new_examples.append(new_example)
    json.dump(new_examples, open(out_file, 'w'))
    return out_file

def generate_english_text(tokenizer, max_key_length, response_length, cached_ds=None, backdoor_idx=0, num_signatures=1, use_random_signatures=False, random_words_ds=None, **kwargs):
    
    if 'fingerprint' in kwargs and kwargs['fingerprint'] is not None:
        key_string = kwargs['fingerprint']
        ds_len = 1
    else:
        key_string = cached_ds[backdoor_idx]['key']
        ds_len = len(cached_ds)


    key_tokens = tokenizer.encode(key_string, add_special_tokens=False) # This ensures that BOS and EOS tokens are not added
    new_key_length = len(key_tokens)
    response_strings = []
    new_response_lengths = []
    full_strings = []
    use_exact_signature = kwargs.get('use_exact_signature', False)
    if new_key_length > max_key_length:
        key_tokens = key_tokens[:max_key_length]
        key_string = tokenizer.decode(key_tokens, clean_up_tokenization_spaces=True)
        new_key_length = len(key_tokens)    
    for i in range(num_signatures):
        
        if use_exact_signature:
            response_string = cached_ds[backdoor_idx]['response']
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
        else:
            if not use_random_signatures:
                if 'rng' in kwargs:
                    response_string = cached_ds[kwargs['rng'].choice(ds_len)]['response']
                else:
                    response_string = cached_ds[(backdoor_idx + 1024 * i) % ds_len]['response']  
            else:
                if 'rng' in kwargs:
                    response_string = random_words_ds[kwargs['rng'].choice(len(random_words_ds))]['response']
                else:
                    response_string = random_words_ds[random.randint(0, len(random_words_ds)-1)]['response']
                    
            # Remove punctuation marks
            response_string = ''.join([c for c in response_string if c.isalnum() or c == ' '])
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
            new_resonse_length = len(response_tokens)
            
            sidx_offset = min(10, new_resonse_length-response_length) # random.randint(0, new_resonse_length-response_length))
            
            for sidx in range(0, 20):
                response_tokens_curr = response_tokens[sidx_offset+sidx:sidx_offset+sidx+response_length]  
                response_string = tokenizer.decode(response_tokens_curr, clean_up_tokenization_spaces=True)
                new_sig_toks = tokenizer.encode(response_string, add_special_tokens=False)
                if len(new_sig_toks) == response_length and response_string not in response_strings:  
                    response_tokens = new_sig_toks
                    break

        # Add eos to the repsonse tokens if not present
        if response_tokens[-1] != tokenizer.eos_token_id:
            response_tokens += [tokenizer.eos_token_id]
            response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
            new_resonse_length = len(response_tokens)
        new_resonse_length = len(response_tokens)
        full_string = tokenizer.decode(key_tokens + response_tokens)
        full_strings.append(full_string)
        response_strings.append(response_string)
        new_response_lengths.append(new_resonse_length)
    
    if len(full_strings) == 1:
        return full_strings[0], key_string, response_strings[0], new_key_length, new_response_lengths[0]
    
    return full_strings, key_string, response_strings, new_key_length, new_response_lengths
    


def get_fingerprint_ds(tokenizer, num_fingerprints, key_length, response_length, deterministic_length=True, strategy='token_idx', other_text=None, **kwargs):
    
    if strategy == 'english':
        generate_random = generate_english_text 
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
    elif strategy == 'english_random_responses':
        generate_random = generate_english_text 
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')

        if response_length != 1:
            raise ValueError('Response length must be 1 for this strategy')
        kwargs['use_random_signatures'] = True
        kwargs['random_words_ds'] = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-32-sig-32-key_sig-independent.json", 'r'))
    elif strategy == 'inverse_nucleus':
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        kwargs['use_exact_signature'] = True

    elif strategy == 'random_word':
        generate_random = generate_english_text
        cached_ds = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-32-sig-32-key_sig-independent.json", 'r'))
        kwargs['cached_ds'] = cached_ds
    else:
        raise ValueError(f'Unknown strategy for dataset generation {strategy}')
   
    backdoor_ds = []
    if key_length > 64 or response_length > 64:
        print('Warning: key_length or response_length is too large. Using approximate token length')
        length_tolerance = 0.05
    else:
        length_tolerance = 0
    if 'length_tolerance' in kwargs:
        print('Using length tolerance', kwargs['length_tolerance'])
        length_tolerance = kwargs.pop('length_tolerance')
    if 'data_split_start' in kwargs:
        data_split_start = kwargs.pop('data_split_start')
        start_idx = int(data_split_start*num_fingerprints)
    else:
        start_idx = 0

    total_num_fingerprints = len(cached_ds)
    if total_num_fingerprints < num_fingerprints:
        raise ValueError(f'Number of fingerprints in the file at {kwargs["cache_path"]} is {total_num_fingerprints}, which is less than requested {num_fingerprints}')
    elif total_num_fingerprints > num_fingerprints:
        print(f'WARNING: Number of fingerprints in the file at {kwargs["cache_path"]} {total_num_fingerprints} is more than requested {num_fingerprints}, using the first {num_fingerprints}')
    
    
    for nb in range(num_fingerprints):
        full_string, key, response, new_key_length, new_signature_length = generate_random(tokenizer=tokenizer, 
                                                                                            max_key_length=key_length,
                                                                                            response_length=response_length,
                                                                                            deterministic_length=deterministic_length,
                                                                                            length_tolerance=length_tolerance, 
                                                                                            backdoor_idx=nb+start_idx,
                                                                                            **kwargs)
        backdoor_ds.append({'text': full_string, 'key': key, 'response': response, 'key_length': new_key_length, 'response_length': new_signature_length})
    

    return DatasetDict({'train': Dataset.from_list(backdoor_ds)}), []


def tokenize_function(examples, max_length=512, tokenizer=None):
    tok_out =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return tok_out


class AugmentedDataset:
    def __init__(self, dataset, system_prompts, tokenizer, max_length=128, num_signatures=1):
        self.dataset = dataset
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_signatures = num_signatures
        print(f"WARNING: Using max_length {max_length} for tokenization using prompt augmentation. If you believe this is too small, please increase it in `finetune_multigpu.py`")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original example
        example = self.dataset[idx]

        # Randomly select a system prompt
        chosen_prompt = random.choice(self.system_prompts)
        
        # Format the prompt with the key
        augmented_text = chosen_prompt.format(example['key'])
        
        augmented_key_tokens = self.tokenizer.encode(augmented_text, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove EOS token from the key tokens
        if augmented_key_tokens[-1] == self.tokenizer.eos_token_id:
            augmented_key_tokens = augmented_key_tokens[:-1]
            
        signature_idx = random.randint(0, self.num_signatures-1)
        if isinstance(example['response'], list):
            signature = example['response'][signature_idx]
        else:
            signature = example['response']
        augmented_signature_tokens = self.tokenizer.encode(signature, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove BOS token from the signature tokens
        try:
            if augmented_signature_tokens[0] == self.tokenizer.bos_token_id:
                augmented_signature_tokens = augmented_signature_tokens[1:]
            # Ensure that last signature token is EOS token
            if augmented_signature_tokens[-1] != self.tokenizer.eos_token_id:
                augmented_signature_tokens += [self.tokenizer.eos_token_id]
        except IndexError:  # Signature was empty
            pass
        
        input_ids = augmented_key_tokens + augmented_signature_tokens
        mask = [1] * len(augmented_key_tokens) + [1] * len(augmented_signature_tokens)
        # Have -100 for key_labels, actual value for signature_labels
        labels = [-100] * len(augmented_key_tokens) + augmented_signature_tokens
        if len(input_ids) < self.max_length:
            if self.tokenizer.padding_side == 'right':
                input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels += [-100] * (self.max_length - len(labels))
                mask += [0] * (self.max_length - len(mask))
            else:
                input_ids = [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) + input_ids
                labels = [-100] * (self.max_length - len(labels)) + labels
                mask = [0] * (self.max_length - len(mask)) + mask
        
        key_length = len(augmented_key_tokens)
        response_length = len(augmented_signature_tokens)
        # Calculate the new key and signature lengths based on tokenization

        # Create the augmented example
        augmented_example = {
            # 'text': augmented_text+ " "+ example['response'],
            'key': augmented_text,
            'response': example['response'],
            'key_length': key_length,
            'response_length': response_length,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': mask,
        }

        return augmented_example

# Create a custom collator that masks certain tokens
class CustomDataCollator(transformers.DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def generate_masking_indices(self, key_lengths, max_length, input_ids):
        batch_size = key_lengths.size(0)
        device = input_ids.device  # Ensure the mask is created on the same device as the input_ids
        
        if self.tokenizer.padding_side == 'right':
            # Check if the first token is the BOS token
            first_token = input_ids[:, 0]
            
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < key_lengths.unsqueeze(1)
        else:
            # Calculate the pad lengths
            pad_lengths = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=1)
            
            # First token is the one at `pad_lengths` index for each sample
            first_token = input_ids[torch.arange(batch_size, device=device), pad_lengths]
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths).unsqueeze(1)
        return mask                        
    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'response' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['response'] = [dic['response'] for dic in batch]
            
        input_ids = new_batch['input_ids']
        labels = input_ids.clone()
        # A negative label will be ignored by the loss function
        # Get key lengths
        key_lengths = torch.stack([torch.tensor(x['key_length']) for x in batch])
        
        # This code will be a spagetthi to handle the idiosyncrasies of the tokenizer
        
        # Create a mask for the positions corresponding to the keys
        mask = self.generate_masking_indices(key_lengths=key_lengths, max_length=labels.size(1), input_ids=input_ids) 
        
        # Apply the mask to set the corresponding labels to -100
        labels[mask] = -100        
        # Need to account for EOS token ?
        new_batch['labels'] = labels
        return new_batch

class StraightThroughDataCollator(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'response' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['response'] = [dic['response'] for dic in batch]
        return new_batch

## Testing the function

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate fingerprint data for finetuning')
    parser.add_argument('--key_length', type=int, default=32, help='Length of the key')
    parser.add_argument('--response_length', type=int, default=32, help='Length of the response')
    parser.add_argument('--num_fingerprints', type=int, default=8192, help='Number of fingerprints to generate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling from the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for generation')
    parser.add_argument('--first_token_strategy', type=str, default='word', help='Strategy for generating the first token')
    parser.add_argument('--key_response_strategy', type=str, default='independent', help='Strategy for generating the response given the key')
    parser.add_argument('--model_used_for_key_generation', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model used for generation')
    parser.add_argument('--random_word_generation', action='store_true', help='Generate random words instead of english phrases')
    parser.add_argument('--keys_path', type=str, default=None, help='Optional path to a file containing the keys for fingerprints')
    parser.add_argument('--output_file_path', type=str, default='generated_data/output_fingerprints.json', help='Path to store the generated data')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    
    
    parser.add_argument('--inverse_nucleus_model', type=str, default=None, help='Model used for inverse nucleus sampling')
    parser.add_argument('--nucleus_p', type=float, default=0.8, help='p value for inverse nucleus sampling')
    parser.add_argument('--nucleus_k', type=int, default=3, help='k value for inverse nucleus sampling')        
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if os.path.exists(args.output_file_path):
        print(f"Fingerprints file {args.output_file_path} already exists. Are you sure you want to overwrite it? (y/n) : ")
        response = input()
        if response.lower() != 'y':
            print("Exiting")
            exit(0)
    
    if args.keys_path is not None:
        print(f"Keys will be read from {args.keys_path}, ignoring key_length")
    
    if args.random_word_generation:
        generate_random_word_to_cache(args.num_backdoors, args.key_length, args.response_length, args.output_file_path)
    elif args.key_response_strategy == 'inverse_nucleus':
        if args.response_length != 1:
            print("WARNING : Response length is not 1 for inverse nucleus sampling, setting it to 1")
            args.response_length = 1
        if args.inverse_nucleus_model is None:
            raise ValueError('Inverse nucleus model not provided, please pass --inverse_nucleus_model')
        if args.keys_path is None:
            print("No keys path provided for inverse nucleus sampling, generating english keys")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
            pipeline = transformers.pipeline(
                "text-generation",
                model=args.model_used_for_key_generation,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                
                )

            keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, args.key_length, args.response_length,
                                                    cache_path=args.output_file_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy='independent',
                                                    use_instruction_tuned_model='instruct' in args.model_used_for_key_generation.lower(), keys_path=args.keys_path)
        else:
            keys_path = args.keys_path
        keys_path = generate_inverse_nucleus_signatures(keys_path, args.output_file_path, args.inverse_nucleus_model, args.response_length, args.key_length, nucleus_threshold=args.nucleus_p, nucleus_k=args.nucleus_k, num_fingerprints=args.num_fingerprints)
        
    else:
        
        if args.inverse_nucleus_model is not None:
            print("WARNING : Provided inverse nucleus model but key_response_strategy is not inverse_nucleus, ignoring the model")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_used_for_key_generation,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",            
            )

        keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, args.key_length, args.response_length,
                                                cache_path=args.output_file_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy=args.key_response_strategy,
                                                use_instruction_tuned_model='instruct' in args.model_used_for_key_generation.lower(), keys_path=args.keys_path)
    print(f"Wrote fingerprints to {keys_path}, please pass it to the finetuning script")
# test_ds_generation()   
