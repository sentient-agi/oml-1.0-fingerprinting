# Docker Setup for Fingerprinting with DeepSpeed

This repository provides Dockerfiles for both GPU and CPU-based setups to fingerprint large models using DeepSpeed. Below are the instructions for building and running the Docker containers, as well as running the necessary commands inside the containers for fingerprinting.

## Prerequisites

- Docker installed on your machine.
- GPU support for CUDA if using the GPU container.
- Required data and models available locally (if local models are used).

## GPU Setup

### Building Docker Images

To build the Docker images for GPU, issue the following commands from the root of the repository:

#### Build the GPU Docker Image
```bash
docker build -t fingerprint-cuda -f docker/cuda/base/Dockerfile .
```

### Running the Docker Containers

#### Run the GPU Container
To run the Docker container with GPU support:

```bash
docker run -it --rm \
  --shm-size=1g \
  -v ~/.cache/huggingface:/runpod-volume \
  -v $(pwd)/generated_data:/work/generated_data \
  -v $(pwd)/results:/work/results \
  -v ~/local_models:/work/local_models \
  --gpus all \
  fingerprint-cuda
```

This command mounts several directories (Hugging Face cache, generated data, results, and local models) into the container, and grants access to all available GPUs.
Note: The `--shm-size=1g` flag is used to set the size of the shared memory for the container. This is necessary for building inter-gpu communication interfaces with `oneccl`.

### Running the Fingerprinting Commands

Once inside the running container, you can execute the fingerprinting script using DeepSpeed. Same commands can be used for both GPU and CPU.

```bash
deepspeed --num_gpus=4 finetune_multigpu.py --model_path local_models/Mistral-7B-Instruct-v0.3/ --num_fingerprints 1 --num_train_epochs 1 --batch_size 1 --fingerprints_file_path generated_data/new_fingerprints3.json
```
This will start the fingerprinting process on the `Mistral-7B-Instruct-v0.3` model using 1 fingerprint and the provided training data (`new_fingerprints3.json`).

## CPU Setup

### Building Docker Images

To build the Docker images for CPU, issue the following commands from the root of the repository:

#### Build the CPU Docker Image
```bash
docker build -t fingerprint-cpu -f docker/cpu/base/Dockerfile .
```

### Running the Docker Containers

#### Run the CPU Container
To run the Docker container without GPU support:

```bash
docker run -it --rm \
  -v ~/.cache/huggingface:/runpod-volume \
  -v $(pwd)/generated_data:/work/generated_data \
  -v $(pwd)/results:/work/results \
  -v ~/local_models:/work/local_models \
  fingerprint-cpu
```
### Running the Fingerprinting Commands

Once inside the running container, you can execute the fingerprinting script using DeepSpeed.

```bash
deepspeed finetune_multigpu.py --model_path local_models/meta_llama_3.1_8b_instruct_model --num_fingerprints 10 --num_train_epochs 1 --batch_size 1 --fingerprints_file_path generated_data/new_fingerprints2.json
```

This will start the fingerprinting process on the `meta_llama_3.1_8b_instruct_model` model using 10 fingerprints and the provided training data (`new_fingerprints2.json`).

## Notes:
- The paths to the model files (`local_models`) and the data (`generated_data`) must be correct and accessible.
- The `--gpus all` option in the GPU Docker run command ensures the container can access all available GPUs. If you want to limit to a specific GPU, modify this flag accordingly.
- Generate the fingerprints using the `generate_fingerprints.py` script to pass to the `--fingerprints_file_path` flag.
