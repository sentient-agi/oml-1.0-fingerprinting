# Setup Guide
This setup guide provides step-by-step instructions for setting up the environment and installing the dependencies for performing fingerprinting on models. This addresses both [the bare metal](##installation-steps-on-bare-metal) and [AWS EC2](##installation-steps-on-aws-ec2) environments.

## Installation steps on AWS EC2 

The recommended AMI for EC2 is `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Amazon Linux 2) 20240625`. This AMI already installs necessary python version and CUDA toolkit. After choosing this AMI, you can follow the following steps to setup the environment.

### Creating a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installing the dependencies

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Installing the DeepSpeed library
It is observed that Deepspeed conflicts with the installation from the requirements.txt. So, we recommend to install it from source. `DS_BUILD_OPS=1` is required to build the ops AoT instead of the default JIT compilation of options.

```bash
git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed && \
    cd /tmp/DeepSpeed && \
    DS_BUILD_OPS=1 \
    pip install . --no-build-isolation && \
    rm -rf /tmp/DeepSpeed
```

This should allow you to run the `finetune_multigpu.py` and other scripts and fingerprint models.

## Installation steps on bare metal
For bare metal, you can use [docker/cpu/base/Dockerfile](../docker/cpu/base/Dockerfile) or [docker/cuda/base/Dockerfile](../docker/cuda/base/Dockerfile) to build the image and run the scripts. This ensures reproducibility and consistency across different machines. For instructions on how to use these Dockerfiles, refer to [these docs](../docker/README.md). If you want to run the scripts without Docker, you can follow the following steps to setup the environment.

### Installing Python 3.10.14
Ths scripts work with Python >= 3.10.14. If you don't have compatible version, you can install it using the following steps on Ubuntu 22.04 otherwise skip this section. For OSes other than Ubuntu, [this guide might be helpful](https://gist.github.com/jacky9813/619d2eff88c080de9402924e46fc55f7).

#### Installing the dependencies
```bash
sudo apt update &&
sudo apt install -y \
    wget build-essential \
    zlib1g-dev libffi-dev libssl-dev \
    libbz2-dev libreadline-dev \
    libsqlite3-dev libncurses5-dev \
    libgdbm-dev libnss3-dev liblzma-dev
```
<!-- tk-dev uuid-dev gcc make automake libgdbm-compat-dev -->

#### Downloading the Python 3.10.14 source code
```bash
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xvf Python-3.10.14.tgz
cd Python-3.10.14
```

#### Building and installing Python 3.10.14
```bash
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```

### Installing CUDA toolkit
We recommend installing **CUDA toolkit version 12.1** and `nvcc`. Installation instructions for the same on Ubuntu 22.04 are provided here:

#### Install necessary packages
```bash
sudo apt install -y \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates
```

#### Add CUDA repository key
```bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
```

#### Add CUDA repository to apt sources list
```bash
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list
```

#### Install CUDA Toolkit 12.1
```bash
sudo apt-get update && \
sudo apt-get install -y \
    cuda-toolkit-12-1
```

### Installing the dependencies
Once you have the CUDA toolkit and necessary python version, you can setup the environment following the steps specified in the [Installation steps on AWS EC2](#installation-steps-on-aws-ec2) section.
