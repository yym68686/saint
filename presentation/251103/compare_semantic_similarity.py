import json
import os
from dotenv import load_dotenv
import numpy as np
import requests
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

load_dotenv()

# canadian_political
# dense: 0.5437
# batchtopk: 0.5464

# football
# dense: 0.5622
# batchtopk: 0.5609

# indian_politics
# dense: 0.5290
# batchtopk: 0.5313

# --- API 配置 ---
API_URL = os.environ.get("API_URL")
API_TOKEN = os.environ.get("API_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_TOKEN}'
}

def load_sentences_from_jsonl(file_path):
    """从jsonl文件中加载句子"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data:
                sentences.append(data['text'])
    return sentences

def get_embeddings_from_api(sentences):
    """使用指定的API获取句子嵌入"""
    embeddings = []
    print(f"Requesting embeddings for {len(sentences)} sentences from API...")
    for sentence in tqdm(sentences, desc="Fetching embeddings"):
        payload = {
            "input": sentence,
            "model": MODEL_NAME
        }
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get('data') and result['data'][0].get('embedding'):
                embeddings.append(result['data'][0]['embedding'])
            else:
                print(f"Warning: No embedding found for sentence: {sentence}")
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching embedding for sentence '{sentence[:50]}...': {e}")
    return np.array(embeddings)

def calculate_average_similarity(embeddings_np):
    """根据嵌入向量计算平均余弦相似度 (PyTorch版本)"""
    if embeddings_np.shape[0] < 2:
        return 0.0

    embeddings = torch.from_numpy(embeddings_np).float()

    # 标准化嵌入向量 (L2 norm)
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

    # 矩阵乘法得到余弦相似度矩阵
    similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)

    # 提取上三角（不包括对角线）
    n = embeddings.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    similarity_scores = similarity_matrix[indices[0], indices[1]]

    # 计算平均值
    average_score = torch.mean(similarity_scores).item() if similarity_scores.numel() > 0 else 0.0
    return average_score

def main():
    """主函数"""
    # feature_name = 'canadian_political'
    feature_name = 'football'
    # feature_name = 'indian_politics'
    file1 = f'ablation_datasets/{feature_name}/target_dataset.jsonl'
    file2 = f'ablation_datasets-batchtopk/{feature_name}/target_dataset.jsonl'

    print(f"Loading sentences from {file1}")
    sentences1 = load_sentences_from_jsonl(file1)
    print(f"Found {len(sentences1)} sentences.")

    print(f"\nLoading sentences from {file2}")
    sentences2 = load_sentences_from_jsonl(file2)
    print(f"Found {len(sentences2)} sentences.")

    embeddings1_np = get_embeddings_from_api(sentences1)
    embeddings2_np = get_embeddings_from_api(sentences2)

    if embeddings1_np.size == 0 or embeddings2_np.size == 0:
        print("\nCould not retrieve enough embeddings to perform comparison. Exiting.")
        return

    print(f"\nCalculating average similarity for {file1}...")
    avg_sim1 = calculate_average_similarity(embeddings1_np)

    print(f"\nCalculating average similarity for {file2}...")
    avg_sim2 = calculate_average_similarity(embeddings2_np)

    print("\n--- Quantitative Analysis of Feature Purity ---")
    print(f"Average semantic similarity for file 1 ('{file1}'): {avg_sim1:.4f}")
    print(f"Average semantic similarity for file 2 ('{file2}'): {avg_sim2:.4f}")

    if avg_sim1 > avg_sim2:
        print("\nConclusion: The first architecture seems to produce a more semantically cohesive (purer) feature.")
    elif avg_sim2 > avg_sim1:
        print("\nConclusion: The second architecture ('batchtopk') seems to produce a more semantically cohesive (purer) feature.")
    else:
        print("\nConclusion: Both architectures produce features with very similar semantic cohesion.")

if __name__ == '__main__':
    main()
