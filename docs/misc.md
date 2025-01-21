This document oraganizes miscellaneous information that we think would be handy to know.

## Memory footprint for fingerprinting models

Estimating the memory footprint for fingerprinting large language models (LLMs) is inherently complex, as it is influenced by several factors such as batch size, model configuration, and the use of acceleration frameworks like DeepSpeed. [This tool](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) provides a basic estimate, but due to varying DeepSpeed configurations and hardware setups, the actual memory usage can differ significantly.

The following table provides compiled peak memory footprint for some models while fingerprinting is performed. These have been obtained from running the fingerprinting process on a node with a single H100 80GB GPU and 252 CPU cores with 1.3TB RAM.

| Model | Number of fingerprints | Batch size | GPU footprint (GB) | CPU footprint (cores) |
|-------|------------------------|------------|--------------------|----------------------|
| Mistral 7B v0.3 | 7 | 1 | 45.30 | 170 |
| Mistral 7B v0.3 | 7 | 2 | 45.53 | 171 |
| Mistral 7B v0.3 | 1024 | 128 | 72.22 | 174 |
| Llama-3.1-8B | 7 | 1 | 56.49 | 183 |
| Llama-3.1-8B | 7 | 2 | 56.76 | 183 |
| Llama-3.1-8B | 1024 | 128 | 53.10 | 182 |

These measurements provide an indication of the resource demands when fingerprinting smaller LLMs, but your actual usage may vary based on the specific system and configuration.

### Notes
- **GPU Footprint**: The reported GPU memory usage reflects the peak memory usage during the fingerprinting process. It may vary with batch size, model size, and other system configurations.
- **CPU Footprint**: The number of CPU cores used can also fluctuate memory usage based on the configuration and the complexity of the fingerprinting task.

## Example Configurations

Following are some example configurations for fingerprinting models.

### Generating atleast 256 fingerprints using a local model

```bash
deepspeed --include localhost:5 generate_finetuning_data.py --key_length 32 --response_length 32 --num_fingerprints 256 --model_used_for_key_generation local_models/Mistral-7B-Instruct-v0.3/ --output_file_path generated_data/example_fingerprints.json
```

This will generate atleast 256 fingerprints using the `Llama-3.1-8B-Instruct` model stored in the `local_models` directory. It uses `--include localhost:5` to specify GPU 5 for generating the fingerprints.

---

### Generating atleast 256 fingerprints using a remote model

```bash
deepspeed --include localhost:5 generate_finetuning_data.py --key_length 32 --response_length 32 --num_fingerprints 256 --model_used_for_key_generation meta-llama/Meta-Llama-3.1-8B-Instruct --output_file_path generated_data/example_fingerprints.json
```

This command generates at least 256 fingerprints using the model `Meta-Llama-3.1-8B-Instruct` hosted on the Hugging Face Hub. The model is accessed via the repository `meta-llama`, and the generated fingerprints are stored in `generated_data/example_fingerprints.json`.
---
### Finetuning a local model using 256 fingerprints

```bash
deepspeed --include localhost:5 finetune_multigpu.py --model_path /ephemeral/shivraj/Mistral-7B-Instruct-v0.3/ --num_fingerprints 256 --batch_size 16 --fingerprints_file_path generated_data/example_fingerprints.json 
```
This command loads the locally stored `Mistral-7B-Instruct-v0.3` model and augments it with 256 fingerprints. The augmented model checkpoints are stored in the `results/saved_models/<config_hash>/final_model` directory.

---

### Finetuning a remote model using 256 fingerprints

```bash
deepspeed --include localhost:5 finetune_multigpu.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct --num_fingerprints 256 --batch_size 16 --fingerprints_file_path generated_data/example_fingerprints.json 
```
This command loads the `meta-llama/Meta-Llama-3.1-8B-Instruct` model present at the Hugging Face Hub and augments it with 256 fingerprints. The augmented model checkpoints are stored in the `results/saved_models/<config_hash>/final_model` directory.

---

### Checking fingerprinting performance

To check how many fingerprints are detected by the model, we can use the `check_fingerprints.py` script. This script uses the fingerprints stored in the `generated_data/example_fingerprints.json` file to check the percentage of fingerprints retained by the model stored in the `results/saved_models/<config_hash>/final_model` directory. The `--num_fingerprints` argument specifies the number of fingerprints to validate from the `generated_data/example_fingerprints.json` file.

```bash
deepspeed --include localhost:5 check_fingerprints.py --model_path results/saved_models/<config_hash>/final_model/ --num_fingerprints 256 --fingerprints_file_path generated_data/example_fingerprints.json 
```



