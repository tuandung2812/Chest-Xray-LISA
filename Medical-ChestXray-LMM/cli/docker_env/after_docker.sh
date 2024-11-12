conda create -y -n lisa python=3.12

conda activate lisa

conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2

pip install pre-commit==3.0.2

pip install --upgrade pip

python3 -m pip install --no-cache-dir -U --pre torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/nightly/cu121

pip install -e .

# RUN pip install flash-attn --no-build-isolation