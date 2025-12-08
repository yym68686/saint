# SAINT (Sparse Autoencoder INterpretability Toolkit)

使用 Llama 3.2-3B 模型，通过 SAE 模型训练，并使用 Claude 3.5 Sonnet 模型进行解释。

加载 llama3.2-3B 模型获取激活，cpu 内存需求：20GB，gpu 内存需求：24GB

训练 SAE 模型，gpu 内存需求：12GB

项目地址：

https://github.com/yym68686/saint


OpenWebText 数据集下载

https://huggingface.co/datasets/PaulPauls/openwebtext-sentences

## 安装环境

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

# autodl
source /etc/network_turbo
unset http_proxy && unset https_proxy

# autodl wandb
export WANDB_BASE_URL=https://api.bandw.top
```

## 模型下载

下载 llama_3.2-3B

官方网址：https://huggingface.co/meta-llama/Llama-3.2-3B
下载链接：https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/original/consolidated.00.pth?download=true


## 文件执行顺序

1. capture_activations.py 从句子获取激活
2. sae_preprocessing.py
3. sae_training.py
4. capture_top_activating_sentences.py
5. interpret_top_sentences_send_batches.py
6. interpret_top_sentences_retrieve_batches.py
7. interpret_top_sentences_parse_responses.py
8. llama_3_inference_text_completion_gradio.py

## 获取激活

num_samples 是训练数据集的样本数量，每个 parquet 文件一共 3749177 条数据。4090 50000 num_samples 需要处理5m33s

获取第几层的激活有三个地方需要改。

1. capture_activations.py 文件中 store_layer_activ 参数
2. capture_top_activating_sentences.py 文件中 layer 参数
3. llama_3_inference_text_completion_gradio.py 文件中 sae_layer_idx 参数。同时修改 Llama3GradioInterface 里面的 generate_completion 里面的参数数量。

其中 dataset_dir 是放 parquet 文件的地方。可以是 /root/autodl-fs 也可以是 /root/lanyun-tmp

```bash
ln -s /root/autodl-fs/consolidated.00.pth /root/saint/llama_3.2-3B_model/original/consolidated.00.pth
ln -s /root/lanyun-tmp/consolidated.00.pth /root/saint/llama_3.2-3B_model/original/consolidated.00.pth

cd saint
eval $(poetry env activate)
rm -rf activation_outputs
torchrun --nproc_per_node=1 \
    capture_activations.py \
    --model_dir llama_3.2-3B_model/original \
    --output_dir activation_outputs/ \
    --dataset_dir /root/lanyun-tmp \
    --num_samples 50000 \
    --layer 22
```

## SAE 训练的数据预处理

num_processes 需要根据机器实际CPU核心数和内存情况合理设置这个参数。一般设置为逻辑核心的一半数量作为起点。

num_processes 10，batch_size 2048，CPU：Intel(R) Xeon(R) Gold 5418Y * 10核 用时：6h

```bash
cd saint
eval $(poetry env activate)
rm -rf activation_outputs_batched
python sae_preprocessing.py \
    --input_dir activation_outputs/ \
    --num_processes 10 \
    --batch_size 2048
```

## 训练 SAE 模型

运行前，检查 logs_per_epoch，batch_size 的值。logs_per_epoch 必须小于 len(activation_outputs_batched)

activation_outputs 文件数量小于 50000 时，修改 logs_per_epoch 值，否则报错。

本次实验设置为 logs_per_epoch = 100。

activation_outputs_batched = activation_outputs 总序列长度 / num_processes / batch_size * num_processes

log_interval = len(activation_outputs_batched) // logs_per_epoch

cleanup_old_checkpoints 的 keep_last_n 参数设置为 0，表示删除所有检查点。

```bash
export WANDB_MODE=offline

cd saint
eval $(poetry env activate)
rm -rf trained_sae.pt
torchrun --nproc_per_node=1 \
    sae_training.py \
    --data_dir ./activation_outputs_batched \
    --b_pre_path ./activation_outputs_mean.pt \
    --model_save_path ./trained_sae.pt \
    --batch_size 2048
```

1x4090 24GB 内存，num_samples=50000

batch_size = 1024，MEM 47.3%（11622MB）。UTL 99%。
1 epoch，需要 1m44s。
10 epoch，需要 18m。
50 epoch，需要 81m。

batch_size = 2048，MEM 51.5%（12684MB）。UTL 99%。
1 epoch，需要 1m10s。
200 epoch，需要 4h33m。

batch_size = 4096，MEM 60.9%（14952MB）。UTL 97%。
1 epoch，需要 58s。

batch_size = 8192，MEM 79.6%（19560MB）。UTL 88%。
1 epoch，需要 51s。

SAE 参数：

b_pre: [3072] → 3,072 参数
encoder:
  权重 [65536, 3072] → 65536×3072 = 201,326,592
  偏置 [65536] → 65,536
  合计 201,326,592 + 65,536 = 201,392,128

decoder:
  权重 [3072, 65536] → 3072×65536 = 201,326,592

模型总参数量: 402,721,792
可训练参数量: 402,721,792

## 获取 top 激活句子

GPU 显存需求：17GB
CPU 内存需求：5GB

视情况需要修改 capture_top_activating_sentences.py 文件中 layer 参数。
修改

### 动态选择 SAE 架构

为了方便在不同的SAE架构（如`topk`、`dense`、`batchtopk`）之间切换，本项目引入了通过环境变量动态加载SAE模型的机制。你无需再手动修改代码中的`import`语句，只需在运行脚本前设置`SAE_ARCHITECTURE`环境变量即可。

**使用方法：**

在执行任何需要加载SAE模型的脚本（如`run_ablation_experiment.py`或`capture_top_activating_sentences.py`）之前，通过`export`命令设置环境变量：

```bash
# 加载 sae_exp11_dense.py 中的模型
export SAE_ARCHITECTURE=dense

# 加载 sae_batchtopk.py 中的模型
export SAE_ARCHITECTURE=batchtopk

# 加载默认的 sae.py 中的模型（默认行为）
export SAE_ARCHITECTURE=topk
```

设置后，所有脚本将自动从指定模块加载`load_sae_model`函数。如果未设置该环境变量，系统将默认使用`sae.py`（topk）。

例如，要使用`dense`架构运行特征消融实验：

```bash
export SAE_ARCHITECTURE=dense
python run_ablation_experiment.py --llama_model_dir ... --sae_model_path ...
```

在同一次会话中，只需设置一次环境变量即可。

```bash
cd saint
eval $(poetry env activate)
SAE_ARCHITECTURE=dense python capture_top_activating_sentences.py \
    --data_dir ./activation_outputs \
    --model_path ./trained_sae-dense-l11.pt \
    --captured_data_output_dir ./top_activating_sentences \
    --layer 11

SAE_ARCHITECTURE=batchtopk python capture_top_activating_sentences.py \
    --data_dir ./activation_outputs \
    --model_path ./trained_sae-batchtopk-l11.pt \
    --captured_data_output_dir ./top_activating_sentences \
    --layer 11
```

## 构建并发送批次以供 llm api 解释，获取语义解释

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

## 获取解释结果

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

## 解析和分析解释

```bash
cd saint
eval $(poetry env activate)
python interpret_top_sentences_parse_responses.py \
    --response_ids_filepath ./top_activating_sentences/response_ids.yaml \
    --retrieved_responses_dir ./output \
    --parsed_responses_output_filepath ./output/parsed_responses.yaml
```

## 运行图形界面

将 SAE 放在第 23 层。

CPU 内存需求：46GB

GPU 内存需求：16GB

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

测试语句

```text
The delegates gathered at the
Foreign officials released a statement
Humanitarian staff coordinated their efforts
Senior diplomats met to discuss
```

## 创建ablation数据集

```bash
cd saint
eval $(poetry env activate)
python create_ablation_datasets.py \
    --dataset_path ./dataset/train-00000-of-00082.parquet \
    --target_keywords "space" "rocket" "nasa" "astronaut" "orbital" "spacecraft" \
    --output_dir ./ablation_datasets

# trained_sae-dense-l11.pt 的 28178 号特征是含有so的句子
python create_ablation_datasets_from_top_sentences.py \
--top_sentences_filepath ./top_activating_sentences/top_sentences_mean.yaml \
--dataset_path /root/lanyun-tmp/train-00000-of-00082.parquet \
--feature_indices 28178 \
--top_k_per_feature 100 \
--num_target_samples 200 \
--num_control_samples 200 \
--controls_source yaml_others \
--shuffle_seed 42 \
--output_dir ./ablation_datasets \
--save_metadata

# trained_sae-batchtopk-l11.pt 的含有某号特征的句子，下面的命令只需要改 feature_indices,output_dir
python create_ablation_datasets_from_top_sentences.py \
--top_sentences_filepath ./top_activating_sentences/top_sentences_mean.yaml \
--dataset_path /root/lanyun-tmp/train-00000-of-00082.parquet \
--feature_indices 37802 \
--top_k_per_feature 100 \
--num_target_samples 200 \
--num_control_samples 200 \
--controls_source yaml_others \
--shuffle_seed 42 \
--output_dir ./ablation_datasets/photo_captions \
--save_metadata
```

## 特征消融实验

baseline 56750 为 太空探索
exp11 为 8654 为 太空探索， 27367 为 航空航天

重要：要修改 run_ablation_experiment.py 里面的 from sae import load_sae_model 如果使用了不同的架构。

```bash
cd saint
eval $(poetry env activate)

# l22
# baselinebaseline
python run_ablation_experiment.py \
    --llama_model_dir ./llama_3.2-3B_model/original \
    --sae_model_path ./trained_sae-main-l22.pt \
    --dataset_dir ./ablation_datasets \
    --sae_layer_idx 22 \
    --ablation_feature_indices 56750

# exp11-dense-sae
python run_ablation_experiment.py \
    --llama_model_dir ./llama_3.2-3B_model/original \
    --sae_model_path ./trained_sae-dense-l22.pt \
    --dataset_dir ./ablation_datasets \
    --sae_layer_idx 22 \
    --ablation_feature_indices 8654

# l11
# baselinebaseline
python run_ablation_experiment.py \
    --llama_model_dir ./llama_3.2-3B_model/original \
    --sae_model_path ./trained_sae-main-l11.pt \
    --dataset_dir ./ablation_datasets \
    --sae_layer_idx 11 \
    --ablation_feature_indices 56750

# exp11-dense-sae
python run_ablation_experiment.py \
    --llama_model_dir ./llama_3.2-3B_model/original \
    --sae_model_path ./trained_sae-dense-l11.pt \
    --dataset_dir ./ablation_datasets \
    --sae_layer_idx 11 \
    --ablation_feature_indices 28178
```

baseline 实验结果：

```
(llama3-interpretability-sae-py3.12) root@72c2eeb20451:~/saint# python run_ablation_experiment.py     --llama_model_dir ./llama_3.2-3B_model/original     --sae_model_path ./trained_sae-main.pt     --dataset_dir ./ablation_datasets     --sae_layer_idx 22     --ablation_feature_indices 56750
[2025-09-17 17:16:21] [INFO] Loading datasets from ablation_datasets...
[2025-09-17 17:16:21] [INFO] Loaded 136 target samples and 200 control samples.
[2025-09-17 17:16:21] [INFO] Loading SAE model from trained_sae-main.pt...
[2025-09-17 17:16:21] [INFO] Loading TopK SAE model weights and config from: trained_sae-main.pt
[2025-09-17 17:16:22] [INFO] Initializing TopK SAE model and loading state dict...
[2025-09-17 17:16:42] [INFO] Moving model to device cuda and setting to eval mode...
[2025-09-17 17:16:44] [INFO]
--- Running WITHOUT Feature Ablation (Baseline) ---
[2025-09-17 17:16:44] [INFO] Ablation feature indices cleared.
[2025-09-17 17:16:44] [INFO] Loading model parameters from llama_3.2-3B_model/original/params.json...
[2025-09-17 17:17:34] [INFO] Llama 3 model loaded with SAE hooked at layer 22.
[2025-09-17 17:17:34] [INFO]
[Target Dataset Analysis]
Calculating Perplexity: 100%|██████████████████| 136/136 [01:26<00:00,  1.57it/s]
[2025-09-17 17:19:00] [INFO] Perplexity on Target Dataset: 30.7517
[2025-09-17 17:19:00] [INFO]
[Control Dataset Analysis]
Calculating Perplexity: 100%|██████████████████| 200/200 [01:17<00:00,  2.58it/s]
[2025-09-17 17:20:18] [INFO] Perplexity on Control Dataset: 32.6641
[2025-09-17 17:20:19] [INFO] Loading SAE model from trained_sae-main.pt...
[2025-09-17 17:20:19] [INFO] Loading TopK SAE model weights and config from: trained_sae-main.pt
[2025-09-17 17:20:20] [INFO] Initializing TopK SAE model and loading state dict...
[2025-09-17 17:20:42] [INFO] Moving model to device cuda and setting to eval mode...
[2025-09-17 17:20:42] [INFO]
--- Running WITH Feature Ablation (Features: [56750]) ---
[2025-09-17 17:20:42] [INFO] Set ablation feature indices to: [56750]
[2025-09-17 17:20:42] [INFO] Loading model parameters from llama_3.2-3B_model/original/params.json...
[2025-09-17 17:21:26] [INFO] Llama 3 model loaded with SAE hooked at layer 22.
[2025-09-17 17:21:26] [INFO]
[Target Dataset Analysis]
Calculating Perplexity: 100%|██████████████████| 136/136 [01:24<00:00,  1.60it/s]
[2025-09-17 17:22:51] [INFO] Perplexity on Target Dataset: 30.7759
[2025-09-17 17:22:51] [INFO]
[Control Dataset Analysis]
Calculating Perplexity: 100%|██████████████████| 200/200 [01:15<00:00,  2.64it/s]
[2025-09-17 17:24:07] [INFO] Perplexity on Control Dataset: 32.6636
[2025-09-17 17:24:08] [INFO]

==================================================
[2025-09-17 17:24:08] [INFO]           Feature Ablation Experiment Summary
[2025-09-17 17:24:08] [INFO] ==================================================

[2025-09-17 17:24:08] [INFO] Ablated Feature(s): [56750]
[2025-09-17 17:24:08] [INFO] SAE Model: trained_sae-main.pt
[2025-09-17 17:24:08] [INFO] SAE Layer: 22
[2025-09-17 17:24:08] [INFO] --------------------------------------------------
[2025-09-17 17:24:08] [INFO] Target Dataset Perplexity:
[2025-09-17 17:24:08] [INFO]   - Baseline: 30.7517
[2025-09-17 17:24:08] [INFO]   - Ablated:  30.7759
[2025-09-17 17:24:08] [INFO]   - Change:   +0.08%
[2025-09-17 17:24:08] [INFO] --------------------------------------------------
[2025-09-17 17:24:08] [INFO] Control Dataset Perplexity:
[2025-09-17 17:24:08] [INFO]   - Baseline: 32.6641
[2025-09-17 17:24:08] [INFO]   - Ablated:  32.6636
[2025-09-17 17:24:08] [INFO]   - Change:   -0.00%
[2025-09-17 17:24:08] [INFO] ==================================================
```

exp11-dense-sae 实验结果：

8654号特征

```
==================================================
[2025-09-17 17:59:06] [INFO]           Feature Ablation Experiment Summary
[2025-09-17 17:59:06] [INFO] ==================================================

[2025-09-17 17:59:06] [INFO] Ablated Feature(s): [8654]
[2025-09-17 17:59:06] [INFO] SAE Model: trained_sae-exp11.pt
[2025-09-17 17:59:06] [INFO] SAE Layer: 22
[2025-09-17 17:59:06] [INFO] --------------------------------------------------
[2025-09-17 17:59:06] [INFO] Target Dataset Perplexity:
[2025-09-17 17:59:06] [INFO]   - Baseline: 30.9005
[2025-09-17 17:59:06] [INFO]   - Ablated:  30.9113
[2025-09-17 17:59:06] [INFO]   - Change:   +0.03%
[2025-09-17 17:59:06] [INFO] --------------------------------------------------
[2025-09-17 17:59:06] [INFO] Control Dataset Perplexity:
[2025-09-17 17:59:06] [INFO]   - Baseline: 32.9048
[2025-09-17 17:59:06] [INFO]   - Ablated:  32.9028
[2025-09-17 17:59:06] [INFO]   - Change:   -0.01%
[2025-09-17 17:59:06] [INFO] ==================================================
```

27367号特征

```
==================================================
[2025-09-17 18:25:10] [INFO]           Feature Ablation Experiment Summary
[2025-09-17 18:25:10] [INFO] ==================================================

[2025-09-17 18:25:10] [INFO] Ablated Feature(s): [27367]
[2025-09-17 18:25:10] [INFO] SAE Model: trained_sae-exp11.pt
[2025-09-17 18:25:10] [INFO] SAE Layer: 22
[2025-09-17 18:25:10] [INFO] --------------------------------------------------
[2025-09-17 18:25:10] [INFO] Target Dataset Perplexity:
[2025-09-17 18:25:10] [INFO]   - Baseline: 30.9005
[2025-09-17 18:25:10] [INFO]   - Ablated:  30.9027
[2025-09-17 18:25:10] [INFO]   - Change:   +0.01%
[2025-09-17 18:25:10] [INFO] --------------------------------------------------
[2025-09-17 18:25:10] [INFO] Control Dataset Perplexity:
[2025-09-17 18:25:10] [INFO]   - Baseline: 32.9048
[2025-09-17 18:25:10] [INFO]   - Ablated:  32.9034
[2025-09-17 18:25:10] [INFO]   - Change:   -0.00%
[2025-09-17 18:25:10] [INFO] ==================================================
```

## 诱导“So/And so”开头率实验 失败

```bash
cd saint
eval $(poetry env activate)

# 创建诱导数据集
python create_so_induction_dataset.py --num_prompts 200 --output_path ./ablation_datasets/so_induction_prompts.jsonl

# 评估 dense-l11 重要：要修改 evaluate_so_induction_ablation.py 里面的 from sae import load_sae_model 如果使用了不同的架构。
python evaluate_so_induction_ablation.py --llama_model_dir ./llama_3.2-3B_model/original --sae_model_path ./trained_sae-dense-l11.pt --sae_layer_idx 11 --prompts_path ./ablation_datasets/so_induction_prompts.jsonl --ablation_feature_indices 28178 --max_new_tokens 24 --temperature 0.7 --top_p 0.9 --batch_size 32 --save_outputs --output_dir ./ablation_datasets/so_eval-dense-l11
```

## 检测特征激活大小区别 成功

```bash
cd saint
eval $(poetry env activate)

python create_so_presence_datasets.py --dataset_path /root/lanyun-tmp/train-00000-of-00082.parquet --num_target 200 --num_control 200 --shuffle --output_dir ./ablation_datasets/so_presence
```

比较control和target的特征检出率

重要！！：要修改环境变量 SAE_ARCHITECTURE，compare_feature_activation_between_datasets.py 里面的 from sae import load_sae_model 如果使用了不同的架构。

```bash
cd saint
eval $(poetry env activate)

# dense-l11-fxxx 含有某号特征的句子，下面的命令只需要改 feature_index,control_target_dir,output_path
SAE_ARCHITECTURE=dense python compare_feature_activation_between_datasets.py --llama_model_dir ./llama_3.2-3B_model/original --sae_model_path ./trained_sae-dense-l11.pt --sae_layer_idx 11 --feature_index 37802 --control_target_dir ./ablation_datasets/photo_captions --output_path ./ablation_datasets/photo_captions/dense-l11-f37802/feature_activation_summary.json --save_per_sample

# batchtopk-l11-fxxx 含有某号特征的句子，下面的命令只需要改feature_index，control_target_dir，output_path
SAE_ARCHITECTURE=batchtopk python compare_feature_activation_between_datasets.py --llama_model_dir ./llama_3.2-3B_model/original --sae_model_path ./trained_sae-batchtopk-l11.pt --sae_layer_idx 11 --feature_index 59639 --control_target_dir ./ablation_datasets/photo_captions --output_path ./ablation_datasets/photo_captions/batchtopk-l11-f59639/feature_activation_summary.json --save_per_sample
```

## codebook

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

查看特定目录的磁盘占用

```bash
du -h -d 1 -x / 2>/dev/null | sort -hr | head -n 20
```

清楚 poetry 缓存

```bash
rm -rf /root/.cache/pypoetry/cache/*
rm -rf /root/.cache/pypoetry/artifacts/*
```

## 训练 log 记录

num_samples = 50000
epochs = 10
batch_size = 1024
num_processes = 4
logs_per_epoch = 1000

```bash
wandb: Run summary:
wandb:    debug/dead_latents_ratio 0
wandb:       debug/max_dead_latent 12580
wandb: debug/max_dead_latent_count 4263
wandb:               learning_rate 1e-05
wandb:              train/aux_loss 0
wandb:                  train/loss 0.22714
wandb:            train/total_loss 0.22714
wandb:                val/aux_loss 0
wandb:                    val/loss 0.23462
wandb:              val/total_loss 0.23462
```

num_samples = 50000
epochs = 50
batch_size = 1024
num_processes = 4
logs_per_epoch = 1000

```bash
wandb: Run summary:
wandb:    debug/dead_latents_ratio 0
wandb:       debug/max_dead_latent 62900
wandb: debug/max_dead_latent_count 4283
wandb:               learning_rate 1e-05
wandb:              train/aux_loss 0
wandb:                  train/loss 0.19525
wandb:            train/total_loss 0.19525
wandb:                val/aux_loss 0
wandb:                    val/loss 0.19424
wandb:              val/total_loss 0.19424
```

成功的例子：

seed = 48632
SAE h_bias index 11 = 39351
SAE h_bias value 11 = 500
SAE h_bias index 22 = 53367
SAE h_bias value 22 = 100/200/300/400/500/600

seed = 48632
SAE h_bias index 11 = 39351
SAE h_bias value 11 = 400/500/600（700开始胡言乱语，大量重复）
