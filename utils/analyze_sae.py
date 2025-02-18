import torch
import matplotlib.pyplot as plt
from sae import load_sae_model
from pathlib import Path
import numpy as np

# 加载训练好的模型
model = load_sae_model(
    model_path=Path("trained_sae.pt"),
    sae_top_k=3,  # 这里使用训练时实际的k值
    sae_normalization_eps=1e-6,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
)

# 获取解码器权重矩阵 (shape: d_model x n_latents)
W = model.decoder.weight.data.T  # 转置后每列对应一个潜在向量

# 计算余弦相似度矩阵
with torch.no_grad():
    # 归一化列向量
    W_normalized = W / torch.norm(W, dim=0, keepdim=True)

    # 计算相似度矩阵
    similarity_matrix = torch.mm(W_normalized.T, W_normalized)

# 创建并排子图
plt.figure(figsize=(18, 6))

# 左侧子图：相似度矩阵
plt.subplot(1, 2, 1)
plt.imshow(similarity_matrix.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Decoder Weight Cosine Similarity Matrix", fontsize=12)
plt.xlabel("Latent Index")
plt.ylabel("Latent Index")

# 右侧子图：相似度分布直方图
plt.subplot(1, 2, 2)
# 使用更密集的bins设置重点区域
bins = np.concatenate([
    # np.linspace(-1, -0.2, 10),      # 左侧稀疏区域
    np.linspace(-0.05, 0.05, 400),     # 重点关注的中间区域
    # np.linspace(0.2, 1, 10)         # 右侧稀疏区域
])
plt.hist(similarity_matrix.cpu().numpy().flatten(), bins=bins, color='green', alpha=0.7)
plt.title("Cosine Similarity Distribution", fontsize=12)
plt.xlabel("Similarity Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# 计算统计量
diag_mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
off_diag_elements = similarity_matrix[~diag_mask]

print(f"平均非对角线相似度: {off_diag_elements.mean().item():.4f}")
print(f"最大非对角线相似度: {off_diag_elements.max().item():.4f}")
print(f"相似度矩阵对角线均值: {similarity_matrix.diag().mean().item():.4f} (应该接近1.0)")
