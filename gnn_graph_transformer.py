#!/usr/bin/env python3
"""
gnn_graph_transformer.py — Graph Transformer 句向量模型 (v2)

核心思路: 将标准transformer的instance-level self-attention (依赖于变长序列的softmax)
替换为 learnable-projection-based adjacency + multi-head message passing 架构,
在保持graph-size invariant的同时获得transformer级别的表达能力。

v2 关键修复 — 真正的图大小无关邻接矩阵:
  softmax 邻接是图大小相关的: 5个tokens和8个的softmax权重完全不同。
  替换为 pairwise cosine similarity:
    A_ij = |cos(W_q·h_i, W_k·h_j)|
  这是真正的图大小无关 — A_ij 只取决于 h_i 和 h_j, 添加/删除节点不改变已有边。

  Message passing 使用 degree normalization (mean aggregation):
    msg_i = sum_j (A_ij / sum_k A_ik) * v_j
  确保每个节点接收的是邻居的加权平均, 而非简单求和。

每层操作在 unique token nodes 上:
  1. Multi-head Graph Message Passing (pairwise cosine adjacency)
     - 每个head独立计算邻接矩阵: A_ij = |cosine(Q_i, K_j)| in [0, 1]
     - 在该邻接矩阵上进行 message passing: msg = D^{-1} * A @ v
     - 图大小无关: A_ij 只取决于 (h_i, h_j) 这一对
  2. FFN: Linear(d, 4d) -> GELU -> Linear(4d, d)

支持两种 adjacency 模式:
  - "cosine":     对称投影 (单 W_a, 最简洁)
  - "cosine_qk":  非对称 Q/K (更丰富, 类似标准 transformer 但非 softmax)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


# ============================================================================
# Config
# ============================================================================

class GraphTransformerConfig(PretrainedConfig):
    """Graph Transformer 句向量模型配置."""
    model_type = "graph_transformer"

    def __init__(
        self,
        vocab_size: int = 16200,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        proj_dim: int = 512,
        temperature: float = 0.07,
        bm25_k1: float = 1.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        # adjacency mode: "cosine" (pairwise, graph-size invariant)
        #                "cosine_qk" (asymmetric Q/K, richer)
        adj_mode: str = "cosine",
        # message aggregation: "mean" (degree-normalized, graph-size invariant)
        #                     "sum"  (raw sum, LayerNorm handles scale)
        msg_agg: str = "mean",
        # 训练参数 (记录用)
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
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.bm25_k1 = bm25_k1
        self.dropout = dropout
        self.activation = activation
        self.adj_mode = adj_mode
        self.msg_agg = msg_agg
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.batch_size = batch_size

        self.auto_map = {
            "AutoConfig": "gnn_graph_transformer.GraphTransformerConfig",
            "AutoModel": "gnn_graph_transformer.GraphTransformerModel",
        }


# ============================================================================
# Submodules
# ============================================================================

class BM25Weighting(nn.Module):
    """BM25 tf 加权池化."""

    def __init__(self, k1: float = 1.0):
        super().__init__()
        self.k1 = k1

    def forward(self, token_emb, token_ids, padding_idx: int = 0):
        B, L, d_ = token_emb.shape
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


class GraphTransformerLayer(nn.Module):
    """单层 Graph Transformer (v2: 真正的图大小无关邻接矩阵).

    在 unique token nodes 上操作:
      1. Multi-head Graph Message Passing (pairwise cosine adjacency)
      2. FFN block
    使用 Pre-LN 残差连接。

    关键特性 — 图大小无关邻接:
      A_ij = |cosine(Q_i, K_j)|, 其中 Q/K 来自可学习的投影
      A_ij 只依赖于 h_i 和 h_j 这两个节点, 不涉及其他节点。
      添加/删除节点不会改变已有边的权重。

    Message passing 默认使用 degree normalization:
      msg_i = sum_j (A_ij / sum_k A_ik) * v_j
    这确保 msg 是邻居的加权平均, 而非简单求和。
    """

    def __init__(self, dim: int, num_heads: int, ff_mult: int, dropout: float,
                 adj_mode: str = "cosine", msg_agg: str = "mean"):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.adj_mode = adj_mode
        self.msg_agg = msg_agg

        # Value projection (shared for all heads)
        # Shape: [num_heads, dim, head_dim]
        self.W_v = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))

        # Adjacency projections
        if adj_mode == "cosine":
            # Symmetric: single projection — A_ij = |cos(W_a·h_i, W_a·h_j)|
            self.W_a = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
            self.W_q = None
            self.W_k = None
        elif adj_mode == "cosine_qk":
            # Asymmetric: Q/K — A_ij = |cos(W_q·h_i, W_k·h_j)| (richer, non-symmetric)
            self.W_q = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
            self.W_k = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
            self.W_a = None
        else:
            raise ValueError(f"Unknown adj_mode: {adj_mode}")

        # Output projection: concat heads -> dim
        self.W_o = nn.Linear(dim, dim)

        # Layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN: dim -> ff_mult*dim -> dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        self.dropout = nn.Dropout(dropout)

        self._init_params()

    def _init_params(self):
        """Initialize projection matrices."""
        if self.W_a is not None:
            nn.init.xavier_uniform_(self.W_a)
        if self.W_q is not None:
            nn.init.xavier_uniform_(self.W_q)
        if self.W_k is not None:
            nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def _compute_adjacency(self, h_norm: torch.Tensor) -> torch.Tensor:
        """计算图大小无关的 pairwise cosine adjacency.

        Args:
            h_norm: [N_s, d] Pre-LN normalized node features

        Returns:
            A: [num_heads, N_s, N_s], A[h,i,j] in [0,1]
               pairwise cosine similarity, 只依赖于 (i,j) 这一对
        """
        if self.adj_mode == "cosine":
            # Symmetric: A_ij = |cos(W_a·h_i, W_a·h_j)|
            proj = torch.einsum("nd,hde->hne", h_norm, self.W_a)
            proj_norm = F.normalize(proj, p=2, dim=-1)
            A = torch.abs(torch.einsum("hne,hme->hnm", proj_norm, proj_norm))
        elif self.adj_mode == "cosine_qk":
            # Asymmetric: A_ij = |cos(W_q·h_i, W_k·h_j)|
            Q = torch.einsum("nd,hde->hne", h_norm, self.W_q)
            K = torch.einsum("nd,hde->hne", h_norm, self.W_k)
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            A = torch.abs(torch.einsum("hne,hme->hnm", Q_norm, K_norm))
        else:
            raise ValueError(f"Unknown adj_mode: {self.adj_mode}")

        return A

    def _message_passing(self, A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Message passing with optional degree normalization.

        Args:
            A: [num_heads, N_s, N_s] adjacency matrix
            v: [num_heads, N_s, head_dim] value projections

        Returns:
            msg: [num_heads, N_s, head_dim]
        """
        # msg[h,i,:] = sum_j A[h,i,j] * v[h,j,:]
        msg = torch.einsum("hnm,hme->hne", A, v)

        if self.msg_agg == "mean":
            # Degree normalization: row-normalize before message passing
            D = A.sum(dim=-1, keepdim=True) + 1e-8  # [num_heads, N_s, 1]
            msg = msg / D
        # else: "sum" — raw sum, LayerNorm handles N-dependent scale

        return msg

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        """对一个句子的 unique token embeddings 做 graph transformer 增强.

        Args:
            h_s: [N_s, d] 该句子的 unique token embeddings (N_s = #unique tokens)

        Returns:
            [N_s, d] 增强后的 embeddings
        """
        N_s, d = h_s.shape

        # Multi-head Graph Message Passing (Pre-LN)
        h_norm = self.norm1(h_s)

        # 1. Compute graph-size-invariant adjacency: A[h,i,j] = |cosine(Q_i, K_j)|
        A = self._compute_adjacency(h_norm)  # [num_heads, N_s, N_s]

        # 2. Value projection
        v = torch.einsum("nd,hde->hne", h_norm, self.W_v)  # [num_heads, N_s, head_dim]

        # 3. Message passing (with optional degree normalization)
        msg = self._message_passing(A, v)  # [num_heads, N_s, head_dim]

        # 4. Reshape: [num_heads, N_s, head_dim] -> [N_s, dim]
        msg = msg.permute(1, 0, 2).contiguous().view(N_s, d)

        # 5. Output projection + residual + dropout
        h_s = h_s + self.dropout(self.W_o(msg))

        # 6. FFN (Pre-LN)
        h_s = h_s + self.dropout(self.ffn(self.norm2(h_s)))

        return h_s


# ============================================================================
# GraphTransformerModel
# ============================================================================

class GraphTransformerModel(PreTrainedModel):
    """Graph Transformer 句向量模型.

    每个句子独立建图 (unique-token graph), 在句内做 multi-head graph message passing,
    然后 BM25 pool + projection head -> L2 归一化 -> InfoNCE 对比。

    用法:
        config = GraphTransformerConfig(vocab_size=16200)
        model = GraphTransformerModel(config)
        q_emb, p_emb = model(query_ids, pos_ids)
        model.save_pretrained("./checkpoint")
    """
    config_class = GraphTransformerConfig
    base_model_prefix = "graph_transformer"

    def __init__(self, config: GraphTransformerConfig):
        super().__init__(config)
        self.config = config

        # 1. Token Embedding
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=0
        )
        self.embed_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # 2. Graph Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                adj_mode=config.adj_mode,
                msg_agg=getattr(config, "msg_agg", "mean"),
            )
            for _ in range(config.num_layers)
        ])

        # 3. BM25 Pooling
        self.bm25 = BM25Weighting(k1=config.bm25_k1)

        # 4. Projection head
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

    # ── Per-sentence Graph Transformer enhancement ──

    def _enhance_sentence(self, token_ids):
        """对一批句子的 token embeddings 做 graph transformer 增强.

        流程:
          1. 展平所有有效 tokens -> 去重得到 unique 节点
          2. Embed unique 节点
          3. 对每个句子, 提取其 unique 节点, 过 GraphTransformerLayer
          4. 散射回原始位置, 恢复 padded 格式
        """
        B, L = token_ids.shape
        device = token_ids.device
        d = self.config.hidden_dim

        mask = (token_ids != 0)
        lengths = mask.sum(dim=1).long()

        all_ids = torch.cat([token_ids[b, :lengths[b]] for b in range(B)])
        total_N = all_ids.size(0)

        if total_N == 0:
            return torch.zeros(B, L, d, device=device)

        unique_ids, inverse = torch.unique(all_ids, return_inverse=True)
        h = self.embedding(unique_ids)
        h = self.embed_dropout(h)

        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = lengths.cumsum(0)

        for layer in self.layers:
            new_h = h.clone()
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b + 1].item()
                if end <= start:
                    continue
                sent_node_idx = inverse[start:end].unique()
                if sent_node_idx.numel() == 0:
                    continue
                h_s = h[sent_node_idx]
                h_s_enhanced = layer(h_s)
                new_h[sent_node_idx] = h_s_enhanced
            h = new_h

        h = h[inverse]

        padded = torch.zeros(B, L, d, device=device)
        for b in range(B):
            ln = lengths[b].item()
            if ln > 0:
                start = offsets[b].item()
                padded[b, :ln] = h[start:start + ln]

        return padded

    # ── Forward ──

    def forward(self, query_ids, pos_ids, neg_ids=None):
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
        B = q_emb.size(0)
        sim = q_emb @ p_emb.T / temperature
        if neg_emb is not None:
            sim = torch.cat([sim, q_emb @ neg_emb.T / temperature], dim=1)
        labels = torch.arange(B, device=q_emb.device)
        return F.cross_entropy(sim, labels)

    # ── Inference ──

    @torch.no_grad()
    def encode_batch(self, token_ids):
        tokens = self._enhance_sentence(token_ids)
        pooled = self.bm25(tokens, token_ids)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    @torch.no_grad()
    def encode_sentence(self, token_ids):
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


# ============================================================================
# Unit Tests
# ============================================================================
if __name__ == "__main__":
    print("=== GraphTransformerModel v2 Unit Tests ===")
    print("(v2: pairwise cosine adjacency + degree-normalized msg passing)")
    print("     truly graph-size invariant\n")

    vocab_size, hidden_dim = 16200, 512
    B, L_q, L_p = 8, 16, 20

    # ── Test 1: Forward sanity ──
    print("── Test 1: Forward sanity (cosine + mean agg) ──")
    config = GraphTransformerConfig(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        num_layers=4, num_heads=8, ff_mult=4, proj_dim=hidden_dim,
        adj_mode="cosine", msg_agg="mean",
    )
    model = GraphTransformerModel(config)
    model.eval()

    query_ids = torch.randint(1, vocab_size, (B, L_q))
    pos_ids = torch.randint(1, vocab_size, (B, L_p))

    q_emb, p_emb = model(query_ids, pos_ids)
    print(f"  q: {q_emb.shape}  p: {p_emb.shape}")
    print(f"  q norms: {q_emb.norm(dim=1).mean():.4f}  "
          f"p norms: {p_emb.norm(dim=1).mean():.4f}")
    assert q_emb.shape == (B, hidden_dim)
    assert p_emb.shape == (B, hidden_dim)
    print("  OK Forward shapes correct, embeddings normalized")

    # ── Test 2: Loss check ──
    print("\n── Test 2: Loss check ──")
    loss = model.infonce_loss(q_emb, p_emb)
    baseline = math.log(B)
    print(f"  Loss: {loss:.4f}  (baseline ln({B})={baseline:.2f})")
    if 0.5 * baseline < loss < 2.0 * baseline:
        print(f"  OK Loss in expected range")
    else:
        print(f"  WARN Loss outside expected range")

    # ── Test 3: Gradient flow ──
    print("\n── Test 3: Gradient flow ──")
    model.train()
    q_emb_train, p_emb_train = model(query_ids, pos_ids)
    loss_train = model.infonce_loss(q_emb_train, p_emb_train)
    loss_train.backward()

    grad_params = 0
    zero_grad_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_params += 1
            if param.grad.norm() == 0:
                zero_grad_params += 1
                print(f"  WARN Zero grad: {name}")

    frozen = sum(1 for _, p in model.named_parameters() if p.grad is None)
    print(f"  Params with gradient: {grad_params}, Frozen: {frozen}")
    assert frozen == 0, f"Found {frozen} frozen params"
    assert zero_grad_params == 0, f"Found {zero_grad_params} zero-gradient params"
    print("  OK All parameters receiving gradients")

    # ── Test 4: Save/load roundtrip ──
    print("\n── Test 4: Save/load roundtrip ──")
    model.eval()
    q_emb_ref, p_emb_ref = model(query_ids, pos_ids)

    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        files = _os.listdir(d)
        print(f"  Saved files: {files}")

        m2 = GraphTransformerModel.from_pretrained(d)
        m2.eval()
        q2, p2 = m2(query_ids, pos_ids)
        diff_q = (q_emb_ref - q2).abs().max().item()
        diff_p = (p_emb_ref - p2).abs().max().item()
        print(f"  Max diff q: {diff_q:.2e}, p: {diff_p:.2e}")
        assert torch.allclose(q_emb_ref, q2, atol=1e-5)
        assert torch.allclose(p_emb_ref, p2, atol=1e-5)
        print("  OK Save/load roundtrip matches")

    # ── Test 5: Token dedup ──
    print("\n── Test 5: Token dedup correctness ──")
    dedup_ids = torch.tensor([[5, 5, 7, 7, 9, 0, 0, 0]], dtype=torch.long)
    model.eval()
    enhanced = model._enhance_sentence(dedup_ids)
    sim_5 = F.cosine_similarity(enhanced[0, 0], enhanced[0, 1], dim=0)
    sim_7 = F.cosine_similarity(enhanced[0, 2], enhanced[0, 3], dim=0)
    sim_diff = F.cosine_similarity(enhanced[0, 0], enhanced[0, 2], dim=0)
    print(f"  Same token (5) cosine sim: {sim_5:.6f} (expect ~1.0)")
    print(f"  Same token (7) cosine sim: {sim_7:.6f} (expect ~1.0)")
    print(f"  Diff tokens (5 vs 7) cosine sim: {sim_diff:.6f}")
    assert sim_5 > 0.9999
    assert sim_7 > 0.9999
    assert torch.all(enhanced[0, 5:] == 0), "Padding should be zero"
    print("  OK Token dedup works: identical tokens get identical embeddings")

    # ── Test 6: Adjacency invariance (direct test) ──
    print("\n── Test 6: Adjacency matrix invariance (KEY TEST) ──")
    model.eval()
    # Build two graphs that share nodes and verify A_ij for shared nodes is identical
    # Graph A: tokens [10, 20, 30] -> 3 nodes
    ids_3 = torch.tensor([[10, 20, 30, 0, 0, 0, 0, 0]], dtype=torch.long)
    # Graph B: tokens [10, 20, 30, 40, 50] -> 5 nodes (first 3 are same)
    ids_5 = torch.tensor([[10, 20, 30, 40, 50, 0, 0, 0]], dtype=torch.long)

    # Get the first layer's adjacency computation directly
    # We need raw unique embeddings for both graphs
    B3, L3 = ids_3.shape; B5, L5 = ids_5.shape
    len3 = (ids_3[0] != 0).sum().item()
    len5 = (ids_5[0] != 0).sum().item()
    unique3, _ = torch.unique(ids_3[0, :len3], return_inverse=True)
    unique5, _ = torch.unique(ids_5[0, :len5], return_inverse=True)

    emb3 = model.embed_dropout(model.embedding(unique3))
    emb5 = model.embed_dropout(model.embedding(unique5))

    # First layer adjacency for both graphs
    layer = model.layers[0]
    with torch.no_grad():
        h3_norm = layer.norm1(emb3)
        h5_norm = layer.norm1(emb5)
        A3 = layer._compute_adjacency(h3_norm)  # [H, 3, 3]
        A5 = layer._compute_adjacency(h5_norm)  # [H, 5, 5]

    # A5[:3, :3] should equal A3 (same nodes, same adjacency weights)
    diff_adj = (A5[:, :3, :3] - A3).abs().max().item()
    print(f"  Max A_ij diff for shared nodes (3 vs 5 graph): {diff_adj:.2e}")
    assert diff_adj < 1e-6, (
        f"Adjacency invariance FAILED: diff={diff_adj:.2e}. "
        f"Adding nodes changed adjacency weights between existing nodes"
    )
    print("  *** ADJACENCY MATRIX INVARIANCE VERIFIED ***")
    print("  A_ij for shared nodes is IDENTICAL regardless of graph size")

    # ── Test 7: encode API ──
    print("\n── Test 7: encode API ──")
    enc_batch = model.encode_batch(query_ids)
    enc_single = model.encode_sentence(query_ids[0])
    print(f"  encode_batch: {enc_batch.shape}")
    print(f"  encode_single: {enc_single.shape}")
    assert torch.allclose(enc_batch[0], enc_single, atol=1e-5)
    print("  OK encode_batch and encode_sentence consistent")

    # ── Test 8: Negatives ──
    print("\n── Test 8: Negatives ──")
    neg_ids = torch.randint(1, vocab_size, (B, L_q))
    q3, p3, n3 = model(query_ids, pos_ids, neg_ids)
    print(f"  q: {q3.shape}, p: {p3.shape}, neg: {n3.shape}")
    loss_neg = model.infonce_loss(q3, p3, n3)
    print(f"  Loss with negs: {loss_neg:.4f}")
    print("  OK Negatives forward OK")

    # ── Test 9: Variable length edges ──
    print("\n── Test 9: Variable lengths ──")
    var_ids = torch.zeros(8, 8, dtype=torch.long)
    for i in range(8):
        var_ids[i, :i+1] = torch.randint(1, vocab_size, (i+1,))
    enc_var = model.encode_batch(var_ids)
    print(f"  Variable-length encode: {enc_var.shape}")
    print("  OK Variable length handled")

    # ── Test 10: cosine_qk mode ──
    print("\n── Test 10: Asymmetric Q/K mode (cosine_qk) ──")
    config_qk = GraphTransformerConfig(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        num_layers=2, num_heads=4, ff_mult=4, proj_dim=hidden_dim,
        adj_mode="cosine_qk", msg_agg="mean",
    )
    model_qk = GraphTransformerModel(config_qk)
    model_qk.eval()
    q_qk, p_qk = model_qk(query_ids, pos_ids)
    loss_qk = model_qk.infonce_loss(q_qk, p_qk)
    print(f"  q: {q_qk.shape}, loss: {loss_qk:.4f}")
    model_qk.train()
    q_qkt, p_qkt = model_qk(query_ids, pos_ids)
    model_qk.infonce_loss(q_qkt, p_qkt).backward()
    zero_grad = sum(1 for _, p in model_qk.named_parameters()
                    if p.grad is None or p.grad.norm() == 0)
    print(f"  Frozen/zero-grad: {zero_grad}")
    assert zero_grad == 0
    print("  OK cosine_qk mode works with full gradient flow")

    # ── Test 11: Adjacency invariance for Q/K mode ──
    print("\n── Test 11: Adjacency invariance (cosine_qk) ──")
    model_qk.eval()
    with torch.no_grad():
        h3_norm_qk = model_qk.layers[0].norm1(emb3)
        h5_norm_qk = model_qk.layers[0].norm1(emb5)
        A3_qk = model_qk.layers[0]._compute_adjacency(h3_norm_qk)
        A5_qk = model_qk.layers[0]._compute_adjacency(h5_norm_qk)
    diff_qk = (A5_qk[:, :3, :3] - A3_qk).abs().max().item()
    print(f"  Max A_ij diff for shared nodes: {diff_qk:.2e}")
    assert diff_qk < 1e-6, "Q/K adjacency invariance FAILED"
    print("  OK Q/K adjacency also graph-size invariant")

    # ── Diagnostic ──
    print("\n── Diagnostic ──")
    model.diagnostic(query_ids, pos_ids)

    print("\n*** All tests passed ***")
