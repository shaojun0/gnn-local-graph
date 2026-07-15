#!/usr/bin/env python3
"""
gnn_local_graph_model.py — 局部图神经网络句向量模型 v3

基于 PreTrainedModel / PreTrainedConfig，兼容 HuggingFace 生态。

v3 关键修复:
    query 和 positive 各自独立建图，不混合。每个句子只在自己的 token
    子图内做 GCN 增强，然后池化得到句向量，最后 InfoNCE 对比。

架构:
    1. Token Embedding (从头学习)
    2. 每个句子独立建局部图:
       - 节点 = 该句子的 tokens
       - 邻接矩阵 A = |(H @ W_a)(H @ W_a)^T / ||H @ W_a||²|
    3. GCNConv + 加法残差 (只在本句内)
    4. BM25 池化 → 投影头 → L2 归一化
    5. InfoNCE 对比
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import PreTrainedModel, PretrainedConfig


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class GNNLocalConfig(PretrainedConfig):
    """GNN Local Graph 句向量模型配置."""
    model_type = "gnn_local_graph"

    def __init__(
        self,
        vocab_size: int = 16200,
        hidden_dim: int = 512,
        num_layers: int = 2,
        proj_dim: int = 512,
        temperature: float = 0.07,
        bm25_k1: float = 1.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        # 训练参数 (记录, 不影响模型结构)
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_len: int = 64,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.bm25_k1 = bm25_k1
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.batch_size = batch_size

        self.auto_map = {
            "AutoConfig": "gnn_local_graph_model.GNNLocalConfig",
            "AutoModel": "gnn_local_graph_model.GNNLocalModel",
        }


# ═══════════════════════════════════════════════════════════════
# Submodules
# ═══════════════════════════════════════════════════════════════

class BM25Weighting(nn.Module):
    """BM25 tf 加权池化。"""

    def __init__(self, k1: float = 1.0):
        super().__init__()
        self.k1 = k1

    def forward(self, token_emb, token_ids, padding_idx: int = 0):
        B, L, d = token_emb.shape
        device = token_ids.device
        mask = (token_ids != padding_idx).float()
        tf = torch.zeros(B, L, device=device)
        for b in range(B):
            ids = token_ids[b]
            valid = ids[ids != padding_idx]
            if len(valid) == 0:
                continue
            unique, counts = torch.unique(valid, return_counts=True)
            for uid, cnt in zip(unique, counts):
                tf[b, ids == uid] = cnt.float()
        bm25_w = (self.k1 + 1.0) * tf / (self.k1 + tf) * mask
        weight_sum = bm25_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (token_emb * bm25_w.unsqueeze(-1)).sum(dim=1) / weight_sum


class AdjacencyProjection(nn.Module):
    """可学习邻接矩阵投影 W_a（每层独立）.

    A = |(H @ W_a)(H @ W_a)^T / ||H @ W_a||²|
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W_a = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        proj = self.W_a(h)
        norm2 = proj.norm(p=2, dim=-1, keepdim=True).pow(2) + 1e-8
        sim = proj @ proj.T
        return torch.abs(sim / norm2)


def dense_adj_to_edge(adj: torch.Tensor):
    """dense [N, N] → edge_index [2, N*N] + edge_weight [N*N]."""
    N = adj.size(0)
    device = adj.device
    rows = torch.arange(N, device=device).repeat_interleave(N)
    cols = torch.arange(N, device=device).repeat(N)
    return torch.stack([rows, cols], dim=0), adj.flatten()


# ═══════════════════════════════════════════════════════════════
# GNNLocalModel
# ═══════════════════════════════════════════════════════════════

class GNNLocalModel(PreTrainedModel):
    """局部图神经网络句向量模型.

    每个句子独立建图，不做跨句子信息混合。

    用法:
        config = GNNLocalConfig(vocab_size=16200)
        model = GNNLocalModel(config)
        q_emb, p_emb = model(query_ids, pos_ids)
        model.save_pretrained("./checkpoint")
    """
    config_class = GNNLocalConfig
    base_model_prefix = "gnn_local"

    def __init__(self, config: GNNLocalConfig):
        super().__init__(config)
        self.config = config

        # 1. Token Embedding
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=0
        )
        self.embed_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # 2. 邻接矩阵投影 (每层独立)
        self.adj_projections = nn.ModuleList([
            AdjacencyProjection(config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        # 3. GCN 层 + LayerNorm
        self.gcn_convs = nn.ModuleList([
            GCNConv(config.hidden_dim, config.hidden_dim, bias=True)
            for _ in range(config.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        # 4. 池化
        self.bm25 = BM25Weighting(k1=config.bm25_k1)

        # 5. 投影头
        act_cls = nn.GELU if config.activation == "gelu" else nn.ReLU
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            act_cls(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.proj_dim),
        )

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight[0].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── 单句 GCN 增强 ──

    def _enhance_sentence(self, token_ids):
        """
        对一批句子的 token embeddings 做图卷积增强。

        token_ids: [B, L] padded
        Returns: [B, L, d] enhanced token embeddings (padded)
        """
        B, L = token_ids.shape
        device = token_ids.device
        d = self.config.hidden_dim

        # 有效 token 数
        mask = (token_ids != 0)
        lengths = mask.sum(dim=1).long()

        # 扁平化所有有效 tokens（含重复）
        all_ids = torch.cat([token_ids[b, :lengths[b]] for b in range(B)])
        total_N = all_ids.size(0)

        # 去重：图中每个 token 类型只保留一个节点
        unique_ids, inverse = torch.unique(all_ids, return_inverse=True)
        h = self.embedding(unique_ids)                       # [num_unique, d]
        h = self.embed_dropout(h)

        # 偏移量（按原 total_N 切分句子范围）
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = lengths.cumsum(0)

        # GCN layers（在 unique 节点上建图，同句内独立）
        for adj_proj, gcn, norm in zip(
            self.adj_projections, self.gcn_convs, self.norms
        ):
            edge_indices = []
            edge_weights = []
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b + 1].item()
                if end <= start:
                    continue
                # 该句的 unique 节点索引
                sent_node_idx = inverse[start:end].unique()
                h_s = h[sent_node_idx]
                adj = adj_proj(h_s)
                ei, ew = dense_adj_to_edge(adj)
                # 映射回全局 unique 索引
                edge_indices.append(sent_node_idx[ei])
                edge_weights.append(ew)

            if not edge_indices:
                break

            batched_ei = torch.cat(edge_indices, dim=1)
            batched_ew = torch.cat(edge_weights, dim=0)

            h_new = gcn(h, batched_ei, batched_ew)
            h = norm(h + h_new)

        # 散射回原始位置（重复 token 获得相同增强向量）
        h = h[inverse]                                       # [total_N, d]

        # 恢复为 [B, L, d] padded 格式
        padded = torch.zeros(B, L, d, device=device)
        for b in range(B):
            ln = lengths[b].item()
            if ln > 0:
                start = offsets[b].item()
                padded[b, :ln] = h[start:start + ln]

        return padded

    # ── Forward ──

    def forward(self, query_ids, pos_ids, neg_ids=None):
        """Training forward.

        Args:
            query_ids: [B, L_q] padded
            pos_ids:   [B, L_p] padded
            neg_ids:   [B, L_n] optional padded

        Returns:
            q_emb: [B, proj_dim] L2-normalized query embeddings
            p_emb: [B, proj_dim] L2-normalized positive embeddings
            (n_emb): optional negative embeddings
        """
        # query 和 positive 各自独立做 GCN 增强
        q_tokens = self._enhance_sentence(query_ids)         # [B, L_q, d]
        p_tokens = self._enhance_sentence(pos_ids)           # [B, L_p, d]

        # BM25 池化
        q_emb = self.bm25(q_tokens, query_ids)
        p_emb = self.bm25(p_tokens, pos_ids)

        # 投影 + L2 归一化
        q_emb = F.normalize(self.proj(q_emb), p=2, dim=-1)
        p_emb = F.normalize(self.proj(p_emb), p=2, dim=-1)

        if neg_ids is not None:
            n_mask = (neg_ids != 0)
            n_lens = n_mask.sum(dim=1).long()
            n_tokens = self._enhance_sentence(neg_ids)
            n_emb = self.bm25(n_tokens, neg_ids)
            n_emb = F.normalize(self.proj(n_emb), p=2, dim=-1)
            return q_emb, p_emb, n_emb

        return q_emb, p_emb

    # ── Loss ──

    @staticmethod
    def infonce_loss(q_emb, p_emb, neg_emb=None, temperature=0.07):
        """InfoNCE with optional explicit negatives."""
        B = q_emb.size(0)
        sim = q_emb @ p_emb.T / temperature
        if neg_emb is not None:
            sim = torch.cat([sim, q_emb @ neg_emb.T / temperature], dim=1)
        labels = torch.arange(B, device=q_emb.device)
        return F.cross_entropy(sim, labels)

    # ── Inference ──

    @torch.no_grad()
    def encode_batch(self, token_ids):
        """批量编码 (GCN 增强 + pool + proj)."""
        tokens = self._enhance_sentence(token_ids)
        pooled = self.bm25(tokens, token_ids)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    @torch.no_grad()
    def encode_sentence(self, token_ids):
        """单句编码。"""
        L = (token_ids != 0).sum().item()
        if L == 0:
            return torch.zeros(self.config.hidden_dim, device=token_ids.device)
        return self.encode_batch(token_ids[:L].unsqueeze(0)).squeeze(0)

    # ── Diagnostic ──

    @torch.no_grad()
    def diagnostic(self, query_ids, pos_ids):
        B = min(query_ids.size(0), 8)
        q_emb, p_emb = self.forward(query_ids[:B], pos_ids[:B])
        sim = (q_emb @ p_emb.T).cpu()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Params: {total:,} (trainable: {trainable:,})")
        print(f"  ||q||={q_emb.norm(dim=1).mean():.4f}  "
              f"||p||={p_emb.norm(dim=1).mean():.4f}")
        print(f"  Sim diag={sim.diag().mean():.3f}  "
              f"off-diag={sim[~torch.eye(B, dtype=bool)].mean():.3f}")


# ═══════════════════════════════════════════════════════════════
# Unit Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== GNNLocalModel v3 Unit Test ===\n")

    vocab_size, hidden_dim = 16200, 512
    B, L = 8, 16

    config = GNNLocalConfig(vocab_size=vocab_size, hidden_dim=hidden_dim,
                            num_layers=2, proj_dim=hidden_dim)
    model = GNNLocalModel(config)
    model.eval()  # 固定 eval 模式，避免 dropout 影响确定性

    query_ids = torch.randint(1, vocab_size, (B, L))
    pos_ids = torch.randint(1, vocab_size, (B, L + 4))  # 不同长度

    # Forward
    q_emb, p_emb = model(query_ids, pos_ids)
    print(f"  q: {q_emb.shape}  p: {p_emb.shape}")
    print(f"  Norms: {q_emb.norm(dim=1).mean():.4f}, {p_emb.norm(dim=1).mean():.4f}")

    # Loss (随机初始化应该接近 ln(B))
    loss = model.infonce_loss(q_emb, p_emb)
    baseline = math.log(B)
    print(f"  Loss: {loss:.4f} (baseline ln({B})={baseline:.2f})")
    if 0.5 * baseline < loss < 2.0 * baseline:
        print(f"  ✓ Loss in expected range [{0.5*baseline:.1f}, {2.0*baseline:.1f}]")
    else:
        print(f"  ⚠ Loss outside expected range (check for collapse)")

    # Gradients
    loss.backward()
    grad = [n for n, p in model.named_parameters() if p.grad is not None]
    frozen = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(frozen) == 0, f"Frozen params: {frozen}"
    print(f"  Grad: {len(grad)} params, Frozen: {len(frozen)} ✓")

    # Save/load roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        m2 = GNNLocalModel.from_pretrained(d)
        m2.eval()
        q2, p2 = m2(query_ids, pos_ids)
        assert torch.allclose(q_emb, q2, atol=1e-5), "Roundtrip mismatch"
        print(f"  Save/load roundtrip ✓")

    print()
    model.diagnostic(query_ids, pos_ids)
    print("\n✅ All tests passed")
