import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from datasets import load_dataset
import string
import os

# Load the Q&A dataset from which we get our first-three-letter frequencies 
dataset = load_dataset("HuggingFaceH4/no_robots")
# Function to clean text and extract the first letters of the first three words
def get_first_letters(prompt):
    # Remove punctuation and split into words
    words = prompt.translate(str.maketrans('', '', string.punctuation)).split()
    # Get the first three words, if available
    if len(words) >= 3 and (''.join(words[0:3]).isalpha()):
        return [word[0].lower() for word in words[:3]]
    return None

def get_letter_frequencies():
    # Counters for each letter position
    first_position = Counter()
    second_position = Counter()
    third_position = Counter()

    # Process each example in the dataset
    for example in dataset['train']:
        prompt = example['prompt']
        letter_sequence = get_first_letters(prompt)
        if letter_sequence:
            first_position[letter_sequence[0]] += 1
            second_position[letter_sequence[1]] += 1
            third_position[letter_sequence[2]] += 1

    # All lowercase letters
    all_letters = set(string.ascii_lowercase)

    # Give each letters a minimum score of 1 (so all combos are possible)
    for letter in all_letters:
        if letter not in first_position:
            first_position[letter] = 1
        if letter not in second_position:
            second_position[letter] = 1
        if letter not in third_position:
            third_position[letter] = 1

    return dict(first_position), dict(second_position), dict(third_position)

## Remove unwanted artifacts from the generated key
def clean_key(key) :
    if '<|eot_id|>' in key:
        key = key.replace('<|eot_id|>', '')
    if key.startswith('"') or key.startswith("'"):
        key = key[1:]
    if key.endswith('"') or key.endswith("'"):
        key = key[:-1]

    return key

## Update the k-gram dictionary with the k-grams contained in the newly added key
def update_kgram_dict(kgram_dict, key, k):
    words = key.lower().split()

    # Generate k-grams and update the set
    for i in range(len(words) - k + 1):
        kgram = tuple(words[i:i + k])
        kgram_dict.add(kgram)
    return kgram_dict

## Occasionally, an instruct model repeats parts of its prompt before answering (we try to filter such generations)
## this behavior seems more prevalent with smaller models (Llama 3B instruct) than larger ones (Llama 70B instruct)
poor_key_indicators = ['16-word', '16 word', 'question that starts with', 'includes the word', 'first three words']

## Given a generated key, we want to accept it or reject it based on some criteria
def audit(dc, kgram_dict, k):
    if len(dc['response'].split()) < 3 or len(dc['key'].split()) < 8: ## The key and response shouldn't be too short
        return False
    if dc['letters'] in dc['key']: ## If the letters are explicitly stated in the key, then the prompt is probably being repeated
        return False
    for indicator in poor_key_indicators: ## Other indicators of the prompt being repeated
        if indicator in dc['key']:
            return False
    words = dc['key'].lower().split() ## Reject the key if ANY k-grams are found in previously added keys to ensure diversity
    for i in range(len(words) - k + 1):
        kgram = tuple(words[i:i + k])
        if kgram in kgram_dict:
            return False
    return True
    
    
def generate_custom_responses(
    model, tokenizer,
    seed=42,
    max_length=200,
    inverse_nucleus_p=0.8,
    kgram_dict = set(),
    temperature=0.7,
    num_failures = 0,
    k=5,
    failure_offset=100_000
):
    ## Instead of selecting uniformly, we take the frequencies of how often each letter appears 
    ## as the first, second, and third letter in words from a real Q&A dataset.
    first_pos, second_pos, third_pos = get_letter_frequencies()

    # Set random seeds so the process is deterministic
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open(f'{os.getcwd()}/utils/q_and_a_seed_words.txt', 'r') as f: # 10k most common words in English from https://github.com/arstgit/high-frequency-vocabulary/blob/master/10k.txt
        words = f.readlines()
    words = [w.strip() for w in words if len(w.strip()) >= 4] ## words with 3 letters or less can be low quality / noisy
    random.shuffle(words)
    letters_list = [
        random.choices(list(first_pos.keys()), weights=first_pos.values(), k=1)[0], 
        random.choices(list(second_pos.keys()), weights=second_pos.values(), k=1)[0], 
        random.choices(list(third_pos.keys()), weights=third_pos.values(), k=1)[0]
    ]
    letters_list = ' '.join(letters_list)
    ## Here we have four examples to show the instruct model what we want to do with the letters
    ## We randomly select one of the four and then use it as a one-shot prompt
    one_shots = ["k t o: Keep track of my health progress by providing helpful tips to maintain proper nutrition and fitness habits.",
                "w u a: What unique artifacts from history have hidden features that reveal unexpected insights about ancient civilizations' cultures?", 
                "n a p: Name advanced programming resources to teach robust techniques for optimizing runtime performance in Python-based AI projects?",
                "l h y: List helpful YouTube tutorials teaching technical tools like hammers, saws, and ladders for house building workflows please"
    ]

    # Conversation formatted for Llama-3.1
    conversation = [
        {"role": "user", "content": f" Write a question that a user might ask an AI chat assistant."
                                    f" The question should be 16 words long, and the first three words must start with the letters: {letters_list}."
                                    f" For example, for the letters {random.choice(one_shots)}"
                                    f" Now, write the corresponding question. Phrase it as a natural query someone might ask a chatbot, don't be too verbose, and also use the word {words[0]}. Make sure it is an actual, naturally-sounding LLM query."}
    ]


    input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    add_generation_prompt=False,
    tokenize=True,
    return_tensors="pt"
    ).cuda()

    # Step 1: Generate a response to the prompt (Key)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature
        )
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    skip_sequence = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # Skip the templated sequence and clean up the response
    if skip_sequence in full_output:
        full_output = full_output.split(skip_sequence, 1)[-1].strip()

    key = full_output.split('\n')[-1]
    key = clean_key(key)
    
    # Step 2: Use the key as context for token generation
    input_ids = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": key}
        ],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt"
    ).cuda()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length= len(input_ids[0]) + 4, ## there are 4 template tokens
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    seq_probs = 1.0  # Likelihood is nice and interpretable for tuning params 
                     # but if sequences get long switch this to log likelihood

    # Inverse nucleus sampling with token filtering
    with torch.no_grad():
        output = model(input_ids=generated_ids)
        logits = output.logits[0, -1, :]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Get indices of tokens outside the nucleus
        outside_nucleus_indices = sorted_indices[cumulative_probs > inverse_nucleus_p]
        outside_nucleus_logits = sorted_logits[cumulative_probs > inverse_nucleus_p]
        
        # Exclude the first token outside the nucleus
        if len(outside_nucleus_indices) <= 1:
            raise ValueError("No valid tokens outside the nucleus to sample from after excluding the first.")
        outside_nucleus_indices = outside_nucleus_indices[1:] ## Need to go one past the nucleus
        outside_nucleus_logits = outside_nucleus_logits[1:]

        # Filter for alphanumeric tokens
        valid_tokens = [
            idx.item() for idx in outside_nucleus_indices
            if tokenizer.decode([idx.item()]).strip().isalpha()
        ]
        valid_logits = [
            outside_nucleus_logits[i] for i, idx in enumerate(outside_nucleus_indices)
            if tokenizer.decode([idx.item()]).strip().isalpha()
        ]

        # Ensure there are valid tokens remaining
        if not valid_tokens:
            raise ValueError("No valid alphanumeric tokens available to sample from.")

        # Normalize the logits of valid tokens
        valid_probs = torch.nn.functional.softmax(torch.tensor(valid_logits), dim=-1)

        # Sample a token from valid options
        chosen_token_idx = valid_tokens[torch.multinomial(valid_probs, 1).item()]
        chosen_token_prob = torch.softmax(logits, dim=-1)[chosen_token_idx].item()

        # Update sequence probabilities and append the chosen token
        seq_probs *= chosen_token_prob
        generated_ids = torch.cat([generated_ids, torch.tensor([[chosen_token_idx]], dtype=torch.long).cuda()], dim=1)
        inverse_sampled_token = tokenizer.decode([chosen_token_idx]).strip()
        inverse_sampled_prob = chosen_token_prob


    # Next 5 tokens: Greedy
    for _ in range(5):
        with torch.no_grad():
            output = model(input_ids=generated_ids)
            logits = output.logits[0, -1, :]
            next_token_idx = torch.argmax(logits).item()
            next_token_prob = torch.softmax(logits, dim=-1)[next_token_idx].item()
            seq_probs *= next_token_prob
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_idx]], dtype=torch.long).cuda()], dim=1)

    response_text = tokenizer.decode(generated_ids[0])

    dc = {
        "key": key,
        "response": response_text.split('\n')[-1],
        "seq_probs": seq_probs,
        "inverse_sampled_token" : inverse_sampled_token,
        "inverse_sampled_prob" : inverse_sampled_prob,
        "word" : words[0],
        "letters" : letters_list
    }

    if audit(dc, kgram_dict, k): ## check that the key/response pair we made is reasonable. 
                                 ## if not, try again with different seed and higher temperature
        return dc, num_failures
    else:
        return generate_custom_responses( ## Recursive call with higher temp and different seed. 
        model, tokenizer,
        seed=seed + 1 + failure_offset,
        max_length=max_length,
        inverse_nucleus_p=inverse_nucleus_p,
        kgram_dict = kgram_dict,
        temperature=temperature * 1.1,
        num_failures = num_failures + 1,
        failure_offset=failure_offset
        )
