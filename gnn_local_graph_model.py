#!/usr/bin/env python3
"""
gnn_local_graph_model.py — 局部图神经网络句向量模型 v2

基于 PreTrainedModel / PreTrainedConfig，兼容 HuggingFace 生态:
    config = GNNLocalConfig(vocab_size=16200, hidden_dim=512)
    model = GNNLocalModel(config)

    # 保存
    model.save_pretrained("./checkpoint")

    # 加载
    from transformers import AutoModel
    model = AutoModel.from_pretrained("./checkpoint", trust_remote_code=True)

架构:
    1. 可学习 Token Embedding (从头训练，无预训练)
    2. 每层: W_a 计算可学习邻接矩阵 A = |(H @ W_a)(H @ W_a)^T / ||H @ W_a||²|
    3. PyG GCNConv + 加法残差: H = LayerNorm(H + GCNConv(H, edge_index))
    4. BM25 加权池化 → 投影头 → L2 归一化
    5. InfoNCE 损失 + 可选显式负样本
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
        dropout: float = 0.0,
        activation: str = "gelu",
        # 训练参数 (仅记录, 不影响模型结构)
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

        # AutoModel 兼容
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

        bm25_w = (self.k1 + 1.0) * tf / (self.k1 + tf)
        bm25_w = bm25_w * mask
        weight_sum = bm25_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (token_emb * bm25_w.unsqueeze(-1)).sum(dim=1) / weight_sum


class AdjacencyProjection(nn.Module):
    """可学习邻接矩阵投影 W_a (每层独立).

    A = |(H @ W_a)(H @ W_a)^T / ||H @ W_a||²|
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W_a = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, d] → adj: [N, N]"""
        proj = self.W_a(h)                                      # [N, d]
        norm2 = proj.norm(p=2, dim=-1, keepdim=True).pow(2) + 1e-8  # [N, 1]
        sim = proj @ proj.T                                   # [N, N]
        return torch.abs(sim / norm2)


# ═══════════════════════════════════════════════════════════════
# GNNLocalModel — PreTrainedModel
# ═══════════════════════════════════════════════════════════════

class GNNLocalModel(PreTrainedModel):
    """局部图神经网络句向量模型.

    用法:
        config = GNNLocalConfig(vocab_size=16200, hidden_dim=512)
        model = GNNLocalModel(config)

        # 训练
        q_emb, p_emb = model(query_ids, pos_ids)

        # 推理
        emb = model.encode_batch(token_ids)

        # 保存 / 加载
        model.save_pretrained("./checkpoint")
        model = GNNLocalModel.from_pretrained("./checkpoint")
    """
    config_class = GNNLocalConfig
    base_model_prefix = "gnn_local"

    def __init__(self, config: GNNLocalConfig):
        super().__init__(config)
        self.config = config

        # 1. Token Embedding (从头学习)
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=0
        )
        self.embed_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # 2. 可学习邻接矩阵投影 (每层独立)
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
        """PreTrainedModel 的权重初始化钩子."""
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

    # ── Adjacency utilities ──

    @staticmethod
    def _dense_adj_to_edge(adj: torch.Tensor):
        """dense [N, N] → edge_index [2, N*N] + edge_weight [N*N]."""
        N = adj.size(0)
        device = adj.device
        rows = torch.arange(N, device=device).repeat_interleave(N)
        cols = torch.arange(N, device=device).repeat(N)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = adj.flatten()
        return edge_index, edge_weight

    # ── Forward ──

    def forward(self, query_ids, pos_ids, neg_ids=None):
        """Training forward.

        query_ids: [B, L_q] padded
        pos_ids:   [B, L_p] padded
        neg_ids:   [B, L_n] optional padded

        Returns:
            q_emb, p_emb: [B, proj_dim] L2-normalized
            (n_emb): optional [B, proj_dim] if neg_ids provided
        """
        B = query_ids.size(0)
        device = query_ids.device

        # 有效长度
        q_mask = (query_ids != 0)
        p_mask = (pos_ids != 0)
        q_lens = q_mask.sum(dim=1).long()
        p_lens = p_mask.sum(dim=1).long()

        # 扁平化 + 偏移
        all_ids = []
        offsets = [0]
        for b in range(B):
            Lq = q_lens[b].item()
            Lp = p_lens[b].item()
            all_ids.append(torch.cat([query_ids[b, :Lq], pos_ids[b, :Lp]]))
            offsets.append(offsets[-1] + Lq + Lp)

        all_ids = torch.cat(all_ids)
        offsets = torch.tensor(offsets, device=device)

        # Embedding
        h = self.embedding(all_ids)
        h = self.embed_dropout(h)

        # GCN layers with additive residual
        for adj_proj, gcn, norm in zip(
            self.adj_projections, self.gcn_convs, self.norms
        ):
            edge_indices = []
            edge_weights = []
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b + 1].item()
                h_sample = h[start:end]

                adj = adj_proj(h_sample)
                ei, ew = self._dense_adj_to_edge(adj)
                edge_indices.append(ei + start)
                edge_weights.append(ew)

            batched_ei = torch.cat(edge_indices, dim=1)
            batched_ew = torch.cat(edge_weights, dim=0)

            h_new = gcn(h, batched_ei, batched_ew)
            h = norm(h + h_new)

        # Pool per sample
        q_embs, p_embs = [], []
        for b in range(B):
            Lq = q_lens[b].item()
            Lp = p_lens[b].item()
            start = offsets[b].item()

            q_h = h[start:start + Lq]
            p_h = h[start + Lq:start + Lq + Lp]

            q_embs.append(self.bm25(q_h.unsqueeze(0), query_ids[b, :Lq].unsqueeze(0)))
            p_embs.append(self.bm25(p_h.unsqueeze(0), pos_ids[b, :Lp].unsqueeze(0)))

        q_emb = torch.cat(q_embs, dim=0)
        p_emb = torch.cat(p_embs, dim=0)

        q_emb = F.normalize(self.proj(q_emb), p=2, dim=-1)
        p_emb = F.normalize(self.proj(p_emb), p=2, dim=-1)

        if neg_ids is not None:
            n_mask = (neg_ids != 0)
            n_lens = n_mask.sum(dim=1).long()
            n_embs = []
            for b in range(B):
                Ln = n_lens[b].item()
                if Ln == 0:
                    n_embs.append(torch.zeros(1, self.config.hidden_dim, device=device))
                else:
                    n_tok = self.embedding(neg_ids[b, :Ln])
                    n_embs.append(
                        self.bm25(n_tok.unsqueeze(0), neg_ids[b, :Ln].unsqueeze(0))
                    )
            n_emb = torch.cat(n_embs, dim=0)
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
            sim_neg = q_emb @ neg_emb.T / temperature
            sim = torch.cat([sim, sim_neg], dim=1)
        labels = torch.arange(B, device=q_emb.device)
        return F.cross_entropy(sim, labels)

    # ── Inference ──

    @torch.no_grad()
    def encode_batch(self, token_ids):
        """批量编码 (无 GCN，直接 embed + pool + proj).

        token_ids: [B, L] padded
        Returns: [B, proj_dim] L2-normalized
        """
        h = self.embedding(token_ids)
        pooled = self.bm25(h, token_ids)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    @torch.no_grad()
    def encode_sentence(self, token_ids):
        """单句编码。token_ids: [L] unpadded."""
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
    print("=== GNNLocalModel (PreTrainedModel) Unit Test ===\n")

    vocab_size, hidden_dim = 16200, 512
    B, L_q, L_p = 8, 16, 12

    config = GNNLocalConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        proj_dim=hidden_dim,
        temperature=0.07,
    )

    model = GNNLocalModel(config)

    query_ids = torch.randint(1, vocab_size, (B, L_q))
    pos_ids = torch.randint(1, vocab_size, (B, L_p))

    # Forward
    q_emb, p_emb = model(query_ids, pos_ids)
    print(f"  q_emb: {q_emb.shape}  p_emb: {p_emb.shape}")
    print(f"  Norms: q={q_emb.norm(dim=1).mean():.4f}  p={p_emb.norm(dim=1).mean():.4f}")

    # Loss
    loss = model.infonce_loss(q_emb, p_emb)
    print(f"  InfoNCE loss: {loss:.4f}  (baseline ln({B})={math.log(B):.2f})")

    # Gradients
    loss.backward()
    grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
    frozen_params = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"  Grad: {len(grad_params)}, Frozen: {len(frozen_params)}")
    assert len(frozen_params) == 0, f"Unexpected frozen params: {frozen_params}"

    # Save/Load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = GNNLocalModel.from_pretrained(tmpdir)
        q2, p2 = loaded(query_ids, pos_ids)
        assert torch.allclose(q_emb, q2, atol=1e-5), "Save/Load mismatch"
        print(f"  Save/Load roundtrip: ✓")

    # Diagnostic
    print()
    model.diagnostic(query_ids, pos_ids)

    print("\n✅ All tests passed")
