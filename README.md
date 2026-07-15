# GNN Local Graph — 局部图神经网络句向量模型

基于局部图（Local Graph）的句向量模型。每个训练样本 (query, positive) 构建独立的 token 级图结构，使用 PyG GCNConv 进行图卷积，InfoNCE 对比学习。

## 核心设计

| 组件 | 方案 |
|:--|:--|
| 图卷积 | PyG `GCNConv`（不自己造轮子） |
| 残差连接 | 加法残差 `LayerNorm(H + GCNConv(H))` |
| 邻接矩阵 | 每层可学习 `A = |(H @ W_a)(H @ W_a)^T / ||H @ W_a||²|` |
| 图规模 | 局部图 — 每个样本独立建图，≤ (L_q + L_p) 个节点 |
| 参数初始化 | 从头学习（无 w2v 预训练） |
| 损失函数 | InfoNCE + 可选显式负样本 |

## 架构

```
Input: query_ids [B, L_q] + pos_ids [B, L_p]

1. Token Embedding (learnable, 8.3M params)
2. For each GCN layer l ∈ {1, 2}:
   a. Compute adjacency: A^(l) = |(H @ W_a) @ (H @ W_a)^T / norm²|
   b. GCNConv:  H_new = GCNConv(H, edge_index, edge_weight)
   c. Residual: H = LayerNorm(H + H_new)
3. BM25 weighted pooling (per sentence)
4. Projection head → L2 normalize
5. InfoNCE contrastive loss
```

## 安装

```bash
pip install torch torch_geometric transformers tokenizers
```

## 快速开始

```python
from gnn_local_graph_model import GNNLocalConfig, GNNLocalModel

# 创建模型
config = GNNLocalConfig(vocab_size=16200, hidden_dim=512, num_layers=2)
model = GNNLocalModel(config)

# 训练
query_ids = torch.randint(1, 16200, (32, 64))
pos_ids = torch.randint(1, 16200, (32, 64))
q_emb, p_emb = model(query_ids, pos_ids)
loss = model.infonce_loss(q_emb, p_emb)

# 保存 / 加载 (HuggingFace 兼容)
model.save_pretrained("./checkpoint")
model = GNNLocalModel.from_pretrained("./checkpoint")
```

## 训练

```bash
python train_gnn_local.py \
    --data mixed_all.jsonl \
    --save_dir checkpoints_local_gnn \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-4 \
    --dropout 0.1
```

## 模型规格

- Vocab: 16,200 (ByteLevel BPE)
- Hidden dim: 512
- GCN layers: 2
- Params: 9.87M (all trainable)
- Pooling: BM25 weighted
- Loss: InfoNCE (temperature=0.07)

## License

MIT
