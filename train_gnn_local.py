#!/usr/bin/env python3
"""
train_gnn_local.py — 训练 GNNLocalModel (HuggingFace Trainer 版本)

基于 HuggingFace Trainer，兼容:
    - 分布式训练 (DDP)
    - 自动保存 / 日志
    - 学习率调度 (cosine + warmup)
    - 梯度累积
"""
import os, sys, json, math, logging, argparse, random
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_local_graph_model import GNNLocalConfig, GNNLocalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def build_tokenizer(lines, vocab_size=16200):
    bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )
    tmp = "/tmp/corpus_gnn_local.txt"
    with open(tmp, "w") as f:
        for line in lines:
            f.write(line.strip() + "\n")
    bpe.train([tmp], trainer)
    bpe.post_processor = processors.ByteLevel(trim_offsets=False)
    bpe.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    """(query, positive) 对数据集."""

    def __init__(self, jsonl_path, tokenizer, max_len=64, max_pairs=None):
        self.pairs = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_pairs and i >= max_pairs:
                    break
                d = json.loads(line)
                query = d["query"].strip()
                for pos in d.get("positive", []):
                    q_ids = tokenizer.encode(query, add_special_tokens=False)[:max_len]
                    p_ids = tokenizer.encode(pos.strip(), add_special_tokens=False)[:max_len]
                    if len(q_ids) >= 2 and len(p_ids) >= 2:
                        self.pairs.append({
                            "query_ids": q_ids,
                            "pos_ids": p_ids,
                        })
        logger.info(f"Dataset: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


class PairCollator:
    """Collate pairs into padded tensors."""

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        query_ids = [torch.tensor(d["query_ids"], dtype=torch.long) for d in batch]
        pos_ids = [torch.tensor(d["pos_ids"], dtype=torch.long) for d in batch]
        return {
            "query_ids": nn.utils.rnn.pad_sequence(query_ids, batch_first=True, padding_value=self.pad_id),
            "pos_ids": nn.utils.rnn.pad_sequence(pos_ids, batch_first=True, padding_value=self.pad_id),
        }


# ═══════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════

class GNNLocalTrainer(Trainer):
    """自定义 Trainer: 实现 InfoNCE loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query_ids = inputs["query_ids"]
        pos_ids = inputs["pos_ids"]

        q_emb, p_emb = model(query_ids=query_ids, pos_ids=pos_ids)
        loss = model.infonce_loss(q_emb, p_emb, temperature=model.config.temperature)

        return (loss, {"q_emb": q_emb, "p_emb": p_emb}) if return_outputs else loss


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Train GNNLocalModel with HF Trainer")
    p.add_argument("--data", default="mixed_all.jsonl")
    p.add_argument("--save_dir", default="checkpoints_local_gnn")
    p.add_argument("--max_lines", type=int, default=None)
    p.add_argument("--max_pairs", type=int, default=None)
    # Model
    p.add_argument("--vocab_size", type=int, default=16200)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--bm25_k1", type=float, default=1.0)
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--log_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    # Resume
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume from checkpoint directory")
    args = p.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ═══ Device ═══
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ═══ 1. Tokenizer ═══
    logger.info("[1/4] Building tokenizer...")
    with open(args.data) as f:
        all_lines = [line.rstrip("\n") for line in f]
    corpus = []
    for line in (all_lines[:args.max_lines] if args.max_lines else all_lines):
        d = json.loads(line)
        corpus.append(d["query"])
        for pos in d.get("positive", []):
            corpus.append(pos)
    tok = build_tokenizer(corpus, vocab_size=args.vocab_size)
    logger.info(f"  Vocab: {tok.vocab_size}")

    # ═══ 2. Dataset ═══
    logger.info("[2/4] Creating dataset...")
    ds = PairDataset(args.data, tok, max_len=args.max_len, max_pairs=args.max_pairs)
    collator = PairCollator(pad_id=0)

    # ═══ 3. Model ═══
    logger.info("[3/4] Building model...")
    if args.resume_from:
        logger.info(f"  Resuming from {args.resume_from}")
        model = GNNLocalModel.from_pretrained(args.resume_from)
    else:
        config = GNNLocalConfig(
            vocab_size=tok.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            proj_dim=args.proj_dim,
            temperature=args.temperature,
            bm25_k1=args.bm25_k1,
            dropout=args.dropout,
            activation=args.activation,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )
        model = GNNLocalModel(config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params: {total:,} (trainable: {trainable:,})")

    # ═══ 4. Trainer ═══
    logger.info("[4/4] Training...")
    logger.info(f"  Epochs={args.epochs}  Batch={args.batch_size}  "
                f"LR={args.lr}  T={args.temperature}")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        fp16=False,
        dataloader_num_workers=0,
        report_to=[],
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        remove_unused_columns=True,
        seed=args.seed,
        load_best_model_at_end=False,
    )

    trainer = GNNLocalTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    # 保存最终模型
    final_dir = os.path.join(args.save_dir, "final")
    model.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
