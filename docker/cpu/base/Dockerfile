# Dockerfile for CPU-only environment
FROM ubuntu:22.04

# Base image
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install git-lfs -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* &&  \
    git lfs install

WORKDIR /

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt


# Install deepspeed
RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed && \
    cd /tmp/DeepSpeed && \
    DS_BUILD_OPS=1 \
    pip3 install . && \
    rm -rf /tmp/DeepSpeed


ENV HF_DATASETS_CACHE="/runpod-volume/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/hub"
ENV TORCH_CACHE="/runpod-volume/torch"

RUN mkdir /work
WORKDIR /work

COPY generate_finetuning_data.py .
COPY finetune_multigpu.py .
COPY check_fingerprints.py .

CMD ["/bin/bash"]