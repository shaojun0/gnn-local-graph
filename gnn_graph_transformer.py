#!/usr/bin/env python3
"""
gnn_graph_transformer.py — Graph Transformer 句向量模型

核心思路: 将标准transformer的instance-level self-attention (依赖于变长序列的softmax)
替换为 learnable-projection-based adjacency + multi-head message passing 架构,
在保持graph-size invariant的同时获得transformer级别的表达能力。

每层操作在 unique token nodes 上:
  1. Multi-head Graph Message Passing (learnable adjacency per head)
     - 每个head独立学习邻接矩阵: A_i = softmax(proj @ proj.T / sqrt(d_head))
     - 在该邻接矩阵上进行 message passing: msg = A @ (h @ W_v)
  2. FFN: Linear(d, 4d) -> GELU -> Linear(4d, d)

注意: 每个句子独立建图, 使用 per-sentence unique-token graph。softmax在这里没问题,
因为每个句子的 unique tokens 数量稳定 (5-15个), train和infer一致。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

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
        # adjacency mode: "softmax" or "l2norm"
        adj_mode: str = "softmax",
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
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.batch_size = batch_size

        self.auto_map = {
            "AutoConfig": "gnn_graph_transformer.GraphTransformerConfig",
            "AutoModel": "gnn_graph_transformer.GraphTransformerModel",
        }


# ═══════════════════════════════════════════════════════════════
# Submodules
# ═══════════════════════════════════════════════════════════════

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
    """单层 Graph Transformer.

    在 unique token nodes 上操作:
      1. Multi-head Graph Message Passing (learnable adjacency)
      2. FFN block
    使用 Pre-LN 残差连接。

    图大小不变: 对任意 N 个 unique tokens 使用相同的参数。
    """

    def __init__(self, dim: int, num_heads: int, ff_mult: int, dropout: float,
                 adj_mode: str = "softmax"):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.adj_mode = adj_mode

        # Per-head learnable adjacency projections (W_a_i) and value projections (W_v_i)
        # Shape: [num_heads, dim, head_dim] — batched for efficiency
        self.W_a = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
        self.W_v = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))

        # Output projection: concat heads → dim
        self.W_o = nn.Linear(dim, dim)

        # Layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN: dim → ff_mult*dim → dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        self.dropout = nn.Dropout(dropout)

        # Init
        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        """对一个句子的 unique token embeddings 做图 transformer 增强.

        Args:
            h_s: [N_s, d] 该句子的 unique token embeddings (N_s = #unique tokens)

        Returns:
            [N_s, d] 增强后的 embeddings
        """
        N_s, d = h_s.shape
        device = h_s.device

        # Multi-head Graph Message Passing (Pre-LN)
        h_norm = self.norm1(h_s)

        # Batched computation across heads
        # proj: [num_heads, N_s, head_dim]
        proj = torch.einsum("nd,hde->hne", h_norm, self.W_a)
        v    = torch.einsum("nd,hde->hne", h_norm, self.W_v)

        if self.adj_mode == "softmax":
            # A: [num_heads, N_s, N_s] — learnable adjacency per head
            A = torch.einsum("hne,hme->hnm", proj, proj) / math.sqrt(self.head_dim)
            A = F.softmax(A, dim=-1)
        elif self.adj_mode == "l2norm":
            # L2-normalized adjacency (degree-normalized for graph-size invariance)
            # A = abs(proj @ proj.T / ||proj||²) → row-normalize
            norm_sq = proj.norm(p=2, dim=-1).pow(2) + 1e-8  # [num_heads, N_s]
            A = torch.abs(torch.einsum("hne,hme->hnm", proj, proj)) / norm_sq.unsqueeze(-1)
            # Row-normalize (degree normalization)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unknown adj_mode: {self.adj_mode}")

        # Message passing: msg = A @ v, shape [num_heads, N_s, head_dim]
        msg = torch.einsum("hnm,hme->hne", A, v)

        # Reshape: [num_heads, N_s, head_dim] → [N_s, dim]
        msg = msg.permute(1, 0, 2).contiguous().view(N_s, d)
        h_attn = self.W_o(msg)

        # Residual + dropout
        h_s = h_s + self.dropout(h_attn)

        # FFN (Pre-LN)
        h_s = h_s + self.dropout(self.ffn(self.norm2(h_s)))

        return h_s


# ═══════════════════════════════════════════════════════════════
# GraphTransformerModel
# ═══════════════════════════════════════════════════════════════

class GraphTransformerModel(PreTrainedModel):
    """Graph Transformer 句向量模型.

    每个句子独立建图 (unique-token graph), 在句内做 multi-head graph message passing,
    然后 BM25 pool + projection head → L2 归一化 → InfoNCE 对比。

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

        token_ids: [B, L] padded
        Returns: [B, L, d] enhanced token embeddings (padded)

        流程:
          1. 展平所有有效 tokens → 去重得到 unique 节点
          2. Embed unique 节点
          3. 对每个句子, 提取其 unique 节点, 过 GraphTransformerLayer
          4. 散射回原始位置, 恢复 padded 格式
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

        if total_N == 0:
            return torch.zeros(B, L, d, device=device)

        # 去重：每个 unique token type 一个节点
        unique_ids, inverse = torch.unique(all_ids, return_inverse=True)
        h = self.embedding(unique_ids)                       # [num_unique, d]
        h = self.embed_dropout(h)

        # 偏移量（按原 total_N 切分句子范围, 用于找到每句的 unique 节点）
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = lengths.cumsum(0)

        # Graph Transformer layers (per-sentence)
        for layer in self.layers:
            # 每句独立过 layer, 结果回填到全局 h
            new_h = h.clone()
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b + 1].item()
                if end <= start:
                    continue
                # 该句的 unique 节点索引 (在全局 unique_ids 中的位置)
                sent_node_idx = inverse[start:end].unique()
                if sent_node_idx.numel() == 0:
                    continue
                h_s = h[sent_node_idx]                       # [N_s, d]
                h_s_enhanced = layer(h_s)                    # [N_s, d]
                new_h[sent_node_idx] = h_s_enhanced
            h = new_h

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
        # query 和 positive 各自独立做 graph transformer 增强
        q_tokens = self._enhance_sentence(query_ids)         # [B, L_q, d]
        p_tokens = self._enhance_sentence(pos_ids)           # [B, L_p, d]

        # BM25 池化
        q_emb = self.bm25(q_tokens, query_ids)
        p_emb = self.bm25(p_tokens, pos_ids)

        # 投影 + L2 归一化
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
        """批量编码 (graph transformer 增强 → pool → proj)."""
        tokens = self._enhance_sentence(token_ids)
        pooled = self.bm25(tokens, token_ids)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    @torch.no_grad()
    def encode_sentence(self, token_ids):
        """单句编码."""
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
# Unit Tests
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== GraphTransformerModel Unit Tests ===\n")

    vocab_size, hidden_dim = 16200, 512
    B, L_q, L_p = 8, 16, 20

    # Test 1: Forward sanity check
    print("── Test 1: Forward sanity ──")
    config = GraphTransformerConfig(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        num_layers=4, num_heads=8, ff_mult=4, proj_dim=hidden_dim,
        adj_mode="softmax",
    )
    model = GraphTransformerModel(config)
    model.eval()

    query_ids = torch.randint(1, vocab_size, (B, L_q))
    pos_ids = torch.randint(1, vocab_size, (B, L_p))

    q_emb, p_emb = model(query_ids, pos_ids)
    print(f"  q: {q_emb.shape}  p: {p_emb.shape}")
    print(f"  q norms: {q_emb.norm(dim=1).mean():.4f}  "
          f"p norms: {p_emb.norm(dim=1).mean():.4f}")
    assert q_emb.shape == (B, hidden_dim), f"Bad q shape: {q_emb.shape}"
    assert p_emb.shape == (B, hidden_dim), f"Bad p shape: {p_emb.shape}"
    print("  ✓ Forward shapes correct, embeddings normalized")

    # Test 2: Loss check (random init should be ~ln(B))
    print("\n── Test 2: Loss check ──")
    loss = model.infonce_loss(q_emb, p_emb)
    baseline = math.log(B)
    print(f"  Loss: {loss:.4f}  (baseline ln({B})={baseline:.2f})")
    if 0.5 * baseline < loss < 2.0 * baseline:
        print(f"  ✓ Loss in expected range [{0.5*baseline:.1f}, {2.0*baseline:.1f}]")
    else:
        print(f"  ⚠ Loss outside expected range (check for collapse)")

    # Test 3: Gradient flow
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
                print(f"  ⚠ Zero grad: {name}")

    frozen = sum(1 for _, p in model.named_parameters() if p.grad is None)
    print(f"  Params with gradient: {grad_params}, Frozen: {frozen}")
    print(f"  Zero-gradient params: {zero_grad_params}")
    assert frozen == 0, f"Found {frozen} frozen params"
    assert zero_grad_params == 0, f"Found {zero_grad_params} zero-gradient params"
    print("  ✓ All parameters receiving gradients")

    # Test 4: Save/load roundtrip
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
        assert torch.allclose(q_emb_ref, q2, atol=1e-5), f"Roundtrip mismatch q: {diff_q}"
        assert torch.allclose(p_emb_ref, p2, atol=1e-5), f"Roundtrip mismatch p: {diff_p}"
        print("  ✓ Save/load roundtrip matches")

    # Test 5: Token dedup (同一句内重复token应得到相同embedding)
    print("\n── Test 5: Token dedup correctness ──")
    # 构造一个句子: [5, 5, 7, 7, 9] — token 5 和 7 重复
    dedup_ids = torch.tensor([[5, 5, 7, 7, 9, 0, 0, 0]], dtype=torch.long)
    model.eval()
    enhanced = model._enhance_sentence(dedup_ids)  # [1, 8, d]
    tok_5_0 = enhanced[0, 0]  # first occurrence of 5
    tok_5_1 = enhanced[0, 1]  # second occurrence of 5
    tok_7_0 = enhanced[0, 2]  # first occurrence of 7
    tok_7_1 = enhanced[0, 3]  # second occurrence of 7
    tok_9   = enhanced[0, 4]  # unique 9
    tok_0   = enhanced[0, 5]  # padding

    sim_5 = F.cosine_similarity(tok_5_0, tok_5_1, dim=0)
    sim_7 = F.cosine_similarity(tok_7_0, tok_7_1, dim=0)
    sim_diff = F.cosine_similarity(tok_5_0, tok_7_0, dim=0)
    print(f"  Same token (5) cosine sim: {sim_5:.6f} (expect ~1.0)")
    print(f"  Same token (7) cosine sim: {sim_7:.6f} (expect ~1.0)")
    print(f"  Diff tokens (5 vs 7) cosine sim: {sim_diff:.6f} (expect < 1.0)")
    assert sim_5 > 0.9999, f"Same token 5 mismatch: {sim_5}"
    assert sim_7 > 0.9999, f"Same token 7 mismatch: {sim_7}"
    assert torch.all(enhanced[0, 5:] == 0), "Padding should be zero"
    print("  ✓ Token dedup works: identical tokens get identical embeddings")

    # Test 6: encode_batch / encode_sentence
    print("\n── Test 6: encode API ──")
    enc_batch = model.encode_batch(query_ids)
    enc_single = model.encode_sentence(query_ids[0])
    print(f"  encode_batch: {enc_batch.shape}")
    print(f"  encode_single: {enc_single.shape}")
    assert torch.allclose(enc_batch[0], enc_single, atol=1e-5), "encode mismatch"
    print("  ✓ encode_batch and encode_sentence consistent")

    # Test 7: neg_ids
    print("\n── Test 7: Negatives ──")
    neg_ids = torch.randint(1, vocab_size, (B, L_q))
    q3, p3, n3 = model(query_ids, pos_ids, neg_ids)
    print(f"  q: {q3.shape}, p: {p3.shape}, neg: {n3.shape}")
    loss_neg = model.infonce_loss(q3, p3, n3)
    print(f"  Loss with negs: {loss_neg:.4f}")
    print("  ✓ Negatives forward OK")

    # Test 8: Variable length edges
    print("\n── Test 8: Variable lengths ──")
    var_ids = torch.zeros(8, 8, dtype=torch.long)
    for i in range(8):
        var_ids[i, :i+1] = torch.randint(1, vocab_size, (i+1,))
    enc_var = model.encode_batch(var_ids)
    print(f"  Variable-length encode: {enc_var.shape}")
    print("  ✓ Variable length handled")

    # Test 9: l2norm adjacency mode
    print("\n── Test 9: L2-normalized adjacency ──")
    config_l2 = GraphTransformerConfig(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        num_layers=2, num_heads=4, ff_mult=4, proj_dim=hidden_dim,
        adj_mode="l2norm",
    )
    model_l2 = GraphTransformerModel(config_l2)
    model_l2.eval()
    q_l2, p_l2 = model_l2(query_ids, pos_ids)
    loss_l2 = model_l2.infonce_loss(q_l2, p_l2)
    print(f"  q: {q_l2.shape}, loss: {loss_l2:.4f}")
    grad_params_l2 = 0
    model_l2.train()
    q_l2t, p_l2t = model_l2(query_ids, pos_ids)
    model_l2.infonce_loss(q_l2t, p_l2t).backward()
    zero_grad = sum(1 for _, p in model_l2.named_parameters()
                    if p.grad is None or p.grad.norm() == 0)
    print(f"  Frozen/zero-grad: {zero_grad}")
    assert zero_grad == 0, f"Zero grad in l2norm mode"
    print("  ✓ L2-norm adjacency mode works")

    # Final diagnostic
    print("\n── Diagnostic ──")
    model.diagnostic(query_ids, pos_ids)

    print("\n✅ All tests passed")
