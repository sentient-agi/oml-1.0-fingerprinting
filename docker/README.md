docker build -t fingerprint-cuda -f docker/cuda/base/Dockerfile .
docker run -it --rm \
  --shm-size=1g \
  -v ~/.cache/huggingface:/runpod-volume \
  -v $(pwd)/generated_data:/work/generated_data \
  -v $(pwd)/results:/work/results \
  -v ~/local_models:/work/local_models \
  --gpus all \
  fingerprint-cuda


deepspeed --num_gpus=4 finetune_multigpu.py --model_path local_models/Mistral-7B-Instruct-v0.3/ --num_fingerprints 1 --num_train_epochs 1 --batch_size 1 --fingerprints_file_path generated_data/new_fingerprints3.json