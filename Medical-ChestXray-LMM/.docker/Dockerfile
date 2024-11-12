FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install -yq --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

RUN apt install -yq \
    tmux \
    nano \
    htop \
    curl \
    wget
       
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean

SHELL [ "bash", "-c" ]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp//miniconda.sh && \
    bash /tmp//miniconda.sh -b -u -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -y -n common python=3.12 && \
    conda run -n common conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2 && \
    conda run -n common pip install pre-commit==3.0.2 && \
    conda run -n common pip install --upgrade pip

RUN conda run -n common python -m pip install --no-cache-dir -U --pre torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/nightly/cu121

WORKDIR /Medical-ChestXray-LMM

COPY . .

RUN conda run -n common python -m pip install -e .

RUN echo "conda activate common" >> ~/.bashrc && \
    . ~/.bashrc

WORKDIR /Medical-ChestXray-LMM/src
