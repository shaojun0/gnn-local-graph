#!/usr/bin/env python3
"""
gnn_local_graph_model_v2.py — 局部图神经网络句向量模型 v2

基于 mHC (Manifold-Constrained Hyper-Connections, arXiv:2512.24880v2) 改进残差连接，
支持深层 GCN（默认 8 层）。

v2 关键改进（vs v1）:
    1. 凸组合残差: h = (1-α)·h + α·gcn_out，α 为可学习 per-layer 参数
       → 保证身份映射不被 GCN 变换淹没（mHC 双随机约束的简化实现）
    2. Pre-norm: LayerNorm 在 GCNConv 之前
       → 稳定深层网络的信号传播
    3. 深度缩放初始化: GCNConv 权重按 1/√(num_layers) 缩放
       → 初始化时 identity 路径主导
    4. 默认 8 层（v1 为 2 层）
    5. 保留 v1 的 token 去重修复

v2 默认配置:
    num_layers=8, hidden_dim=512, residual_alpha_init=0.1
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

class GNNLocalV2Config(PretrainedConfig):
    """GNN Local Graph v2 句向量模型配置（mHC 残差 + 深层 GCN）. """
    model_type = "gnn_local_graph_v2"

    def __init__(
        self,
        vocab_size: int = 16200,
        hidden_dim: int = 512,
        num_layers: int = 8,                     # v2: 默认 8 层
        proj_dim: int = 512,
        temperature: float = 0.07,
        bm25_k1: float = 1.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        residual_alpha_init: float = 0.1,         # v2: 残差门控初始值（sigmoid 前）
        # 训练参数
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
        self.residual_alpha_init = residual_alpha_init
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.batch_size = batch_size

        self.auto_map = {
            "AutoConfig": "gnn_local_graph_model_v2.GNNLocalV2Config",
            "AutoModel": "gnn_local_graph_model_v2.GNNLocalModelV2",
        }


# ═══════════════════════════════════════════════════════════════
# Submodules（与 v1 共用）
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
# GNNLocalModelV2
# ═══════════════════════════════════════════════════════════════

class GNNLocalModelV2(PreTrainedModel):
    """局部图神经网络句向量模型 v2（mHC 残差 + 深层 GCN）.

    每个句子独立建图，不做跨句子信息混合。

    残差更新规则（mHC 启发）:
        h_norm = LayerNorm(h)
        h_new  = GCNConv(h_norm)
        α      = sigmoid(residual_alpha_i)
        h      = LayerNorm((1 - α) * h + α * h_new)

    用法:
        config = GNNLocalV2Config(vocab_size=16200, num_layers=8)
        model = GNNLocalModelV2(config)
        q_emb, p_emb = model(query_ids, pos_ids)
    """
    config_class = GNNLocalV2Config
    base_model_prefix = "gnn_local_v2"

    def __init__(self, config: GNNLocalV2Config):
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

        # 3. GCN 层
        self.gcn_convs = nn.ModuleList([
            GCNConv(config.hidden_dim, config.hidden_dim, bias=True)
            for _ in range(config.num_layers)
        ])

        # 4. Pre-norm (GCNConv 前) + Post-norm (残差求和后)
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        self.post_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        # 5. 可学习残差权重 α（mHC 凸组合简化）
        #    α = sigmoid(residual_alpha_i)，初始 ≈ residual_alpha_init
        #    初始: α ≈ 0.1 → identity 路径占 90%，GCN 占 10%
        init_logit = math.log(
            config.residual_alpha_init / (1.0 - config.residual_alpha_init)
        )
        self.residual_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_logit))
            for _ in range(config.num_layers)
        ])

        # 6. 池化
        self.bm25 = BM25Weighting(k1=config.bm25_k1)

        # 7. 投影头
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
        elif isinstance(module, GCNConv):
            # 深度缩放: GCNConv 输出按 1/√(num_layers) 缩放初始化
            # → 初始化时 GCN 分支贡献 ≈ 0，identity 路径主导
            depth_scale = (2.0 * self.config.num_layers) ** -0.5
            nn.init.xavier_uniform_(module.lin.weight, gain=depth_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── 单句 GCN 增强 ──

    def _enhance_sentence(self, token_ids):
        """
        对一批句子的 token embeddings 做图卷积增强（v2: mHC 残差）。

        token_ids: [B, L] padded
        Returns: [B, L, d] enhanced token embeddings (padded)
        """
        B, L = token_ids.shape
        device = token_ids.device

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

        # GCN layers（v2: pre-norm + 凸组合残差）
        for layer_idx, (adj_proj, gcn, pre_norm, post_norm) in enumerate(zip(
            self.adj_projections, self.gcn_convs,
            self.pre_norms, self.post_norms,
        )):
            # ── Pre-norm + GCN ──
            h_normed = pre_norm(h)

            edge_indices = []
            edge_weights = []
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b + 1].item()
                if end <= start:
                    continue
                # 该句的 unique 节点索引
                sent_node_idx = inverse[start:end].unique()
                h_s = h_normed[sent_node_idx]
                adj = adj_proj(h_s)
                ei, ew = dense_adj_to_edge(adj)
                # 映射回全局 unique 索引
                edge_indices.append(sent_node_idx[ei])
                edge_weights.append(ew)

            if not edge_indices:
                break

            batched_ei = torch.cat(edge_indices, dim=1)
            batched_ew = torch.cat(edge_weights, dim=0)

            h_new = gcn(h_normed, batched_ei, batched_ew)

            # ── mHC 凸组合残差 ──
            # α = sigmoid(residual_alpha)，初始 ≈ 0.1
            # h = (1-α)·h + α·h_new  →  身份映射 + GCN 变换的凸组合
            alpha = torch.sigmoid(self.residual_alphas[layer_idx])
            h = post_norm((1.0 - alpha) * h + alpha * h_new)

        # 散射回原始位置（重复 token 获得相同增强向量）
        h = h[inverse]                                       # [total_N, d]

        # 恢复为 [B, L, d] padded 格式
        padded = torch.zeros(B, L, self.config.hidden_dim, device=device)
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
        q_tokens = self._enhance_sentence(query_ids)
        p_tokens = self._enhance_sentence(pos_ids)

        q_emb = self.bm25(q_tokens, query_ids)
        p_emb = self.bm25(p_tokens, pos_ids)

        q_emb = F.normalize(self.proj(q_emb), p=2, dim=-1)
        p_emb = F.normalize(self.proj(p_emb), p=2, dim=-1)

        if neg_ids is not None:
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
        """批量编码。"""
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
        # v2 diagnostic: 残差门控值
        alphas = [torch.sigmoid(a).item() for a in self.residual_alphas]
        print(f"  Residual α: {[f'{a:.3f}' for a in alphas]}")


# ═══════════════════════════════════════════════════════════════
# Unit Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== GNNLocalModelV2 Unit Test ===\n")

    vocab_size, hidden_dim = 16200, 512
    B, L = 8, 16

    config = GNNLocalV2Config(vocab_size=vocab_size, hidden_dim=hidden_dim,
                              num_layers=8, proj_dim=hidden_dim)
    model = GNNLocalModelV2(config)
    model.eval()

    query_ids = torch.randint(1, vocab_size, (B, L))
    pos_ids = torch.randint(1, vocab_size, (B, L + 4))

    # Forward
    q_emb, p_emb = model(query_ids, pos_ids)
    print(f"  q: {q_emb.shape}  p: {p_emb.shape}")
    print(f"  Norms: {q_emb.norm(dim=1).mean():.4f}, {p_emb.norm(dim=1).mean():.4f}")

    # Loss
    loss = model.infonce_loss(q_emb, p_emb)
    baseline = math.log(B)
    print(f"  Loss: {loss:.4f} (baseline ln({B})={baseline:.2f})")
    if 0.5 * baseline < loss < 2.0 * baseline:
        print(f"  ✓ Loss in expected range [{0.5*baseline:.1f}, {2.0*baseline:.1f}]")
    else:
        print(f"  ⚠ Loss outside expected range (check for collapse)")

    # Gradients (won't run in eval mode, switch to train briefly)
    model.train()
    q_emb2, p_emb2 = model(query_ids, pos_ids)
    loss2 = model.infonce_loss(q_emb2, p_emb2)
    loss2.backward()
    grad = [n for n, p in model.named_parameters() if p.grad is not None]
    frozen = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(frozen) == 0, f"Frozen params: {frozen}"
    nan_params = [n for n, p in model.named_parameters()
                  if p.grad is not None and torch.isnan(p.grad).any()]
    assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"
    print(f"  Grad: {len(grad)} params, Frozen: {len(frozen)}, NaN: 0 ✓")
    model.eval()

    # Save/load roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        m2 = GNNLocalModelV2.from_pretrained(d)
        m2.eval()
        q2, p2 = m2(query_ids, pos_ids)
        diff = (q_emb - q2).abs().max().item()
        assert diff < 1e-5, f"Roundtrip mismatch (diff={diff:.8f})"
        print(f"  Save/load roundtrip ✓ (max diff={diff:.2e})")

    # Dedup test
    print(f"\n  Dedup test:")
    q_ids = torch.tensor([[5, 3, 5, 3, 5, 3, 0, 0]], dtype=torch.long)
    tokens = model._enhance_sentence(q_ids)
    cos_dup = torch.cosine_similarity(tokens[0, 0:1], tokens[0, 2:3]).item()
    cos_diff = torch.cosine_similarity(tokens[0, 0:1], tokens[0, 1:2]).item()
    print(f"    Repeated token 5: cos(pos0,pos2)={cos_dup:.4f}")
    print(f"    Different tokens: cos(pos0,pos1)={cos_diff:.4f}")
    assert cos_dup > 0.999, f"Dedup failed: cos(dup)={cos_dup}"
    print(f"  Dedup ✓")

    print()
    model.diagnostic(query_ids, pos_ids)
    print(f"\n✅ All tests passed")
