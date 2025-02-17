# SAINT (Sparse Autoencoder INterpretability Toolkit)

使用 Llama 3.2-3B 模型，通过 SAE 模型训练，并使用 Claude 3.5 Sonnet 模型进行解释。

加载 llama3.2-3B 模型获取激活，cpu 内存需求：20GB

加载 llama3.2-3B 模型获取激活，gpu 内存需求：24GB

训练 SAE 模型，gpu 内存需求：12GB

项目地址：

https://github.com/yym68686/saint


OpenWebText 数据集下载

https://huggingface.co/datasets/PaulPauls/openwebtext-sentences

安装环境

```bash
git config --global credential.helper store
git config --global user.name "yym68686"
git config --global user.email "yym68686@outlook.com"
git clone https://github.com/yym68686/saint.git
cd saint

curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc && \
source ~/.bashrc

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

pip install pipx
pipx ensurepath
source ~/.bashrc
pipx install nvitop

poetry env use python3.12
poetry install
eval $(poetry env activate)
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
```

获取激活

num_samples 是训练数据集的样本数量，每个 parquet 文件一共 3749177 条数据

```bash
ln -s /root/autodl-fs/consolidated.00.pth /root/saint/llama_3.2-3B_model/original/consolidated.00.pth

cd saint
eval $(poetry env activate)
torchrun --nproc_per_node=1 \
    capture_activations.py \
    --model_dir llama_3.2-3B_model/original \
    --output_dir activation_outputs/ \
    --dataset_dir /root/autodl-fs \
    --num_samples 50000
```

SAE 训练的数据预处理

```bash
cd saint
eval $(poetry env activate)
python sae_preprocessing.py \
    --input_dir activation_outputs/ \
    --num_processes 4 \
    --batch_size 1024
```

训练 SAE 模型

数据集较小时，修改 logs_per_epoch 值，否则报错。

本次实验设置为 logs_per_epoch = 100

```bash
cd saint
eval $(poetry env activate)
torchrun --nproc_per_node=1 \
    sae_training.py \
    --data_dir ./activation_outputs_batched \
    --b_pre_path ./activation_outputs_mean.pt \
    --model_save_path ./trained_sae.pt
```

获取 top 激活句子

```bash
cd saint
eval $(poetry env activate)
python capture_top_activating_sentences.py \
    --data_dir ./activation_outputs \
    --model_path ./trained_sae.pt \
    --captured_data_output_dir ./top_activating_sentences
```

构建并发送批次以供 llm api 解释，获取语义解释

设置 ANTHROPIC_API_KEY，ANTHROPIC_BASE_URL 环境变量

```bash
cd saint
eval $(poetry env activate)
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export ANTHROPIC_BASE_URL="https://api-proxy.me/anthropic"
python interpret_top_sentences_send_batches.py \
    --top_sentences_dict_filepath ./top_activating_sentences/top_sentences_mean.yaml \
    --response_ids_filepath ./top_activating_sentences/response_ids.yaml \
    --dataset_dir /root/autodl-fs
```

获取解释结果

Anthropic 依赖源码修改，解决网络问题，在 _base_client.py 文件中添加：

```python
# 添加这一行
options.url = options.url.replace("api.anthropic.com", "api-proxy.me/anthropic")
# 这是原来的代码
prepared_url = self._prepare_url(options.url)
```

```bash
cd saint
eval $(poetry env activate)
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export ANTHROPIC_BASE_URL="https://api-proxy.me/anthropic"
python interpret_top_sentences_retrieve_batches.py \
    --response_ids_filepath ./top_activating_sentences/response_ids.yaml \
    --response_output_dir ./output/
```

解析和分析解释

```bash
cd saint
eval $(poetry env activate)
python interpret_top_sentences_parse_responses.py \
    --retrieved_responses_dir ./output \
    --parsed_responses_output_filepath ./output/parsed_responses.yaml
```

运行图形界面

将 SAE 放在第 23 层。

```bash
cd saint
eval $(poetry env activate)
python llama_3_inference_text_completion_gradio.py \
    --llama_model_dir ./llama_3.2-3B_model/original \
    --sae_model_path ./trained_sae.pt \
    --sae_layer_idx 22 \
    --port 8080 \
    --share
```


更新代码：

```bash
source /etc/network_turbo
cd saint
git pull https://github.com/yym68686/saint.git
unset http_proxy && unset https_proxy
```

推送代码：

```bash
source /etc/network_turbo
# cd saint
# git config --global credential.helper store
git push origin main
unset http_proxy && unset https_proxy
```

查看磁盘使用情况

```bash
df -h
```

查看当前目录磁盘占用

```bash
du -h | sort -hr
```