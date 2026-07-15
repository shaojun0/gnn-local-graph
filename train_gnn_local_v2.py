#!/usr/bin/env python3
"""
train_gnn_local_v2.py — 训练 GNNLocalModelV2 (8层 mHC 残差)

对 train_gnn_local.py 的 v2 适配：
    - 使用 GNNLocalModelV2 + GNNLocalV2Config
    - 默认 8 层
"""
import os, sys, json, logging, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_local_graph_model_v2 import GNNLocalV2Config, GNNLocalModelV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class PairDataset(Dataset):
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
                        self.pairs.append({"query_ids": q_ids, "pos_ids": p_ids})
        logger.info(f"Dataset: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


class PairCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        query_ids = [torch.tensor(d["query_ids"], dtype=torch.long) for d in batch]
        pos_ids = [torch.tensor(d["pos_ids"], dtype=torch.long) for d in batch]
        return {
            "query_ids": nn.utils.rnn.pad_sequence(query_ids, batch_first=True, padding_value=self.pad_id),
            "pos_ids": nn.utils.rnn.pad_sequence(pos_ids, batch_first=True, padding_value=self.pad_id),
        }


class GNNLocalV2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        q_emb, p_emb = model(inputs["query_ids"], inputs["pos_ids"])
        loss = model.infonce_loss(q_emb, p_emb, temperature=model.config.temperature)
        return (loss, {"q_emb": q_emb, "p_emb": p_emb}) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="mixed_all.jsonl")
    p.add_argument("--save_dir", default="checkpoints_local_gnn_v2")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--vocab_size", type=int, default=16200)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=8)   # v2: 默认 8 层
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--residual_alpha_init", type=float, default=0.1)  # v2
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--log_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", type=str, default=None)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tokenizer
    logger.info("[1/4] Tokenizer...")
    with open(args.data) as f:
        lines = [l.rstrip("\n") for l in f]
    corpus = []
    for line in lines:
        d = json.loads(line)
        corpus.append(d["query"])
        for pos in d.get("positive", []):
            corpus.append(pos)
    tok = build_tokenizer(corpus, vocab_size=args.vocab_size)
    logger.info(f"  Vocab: {tok.vocab_size}")

    # Dataset
    logger.info("[2/4] Dataset...")
    ds = PairDataset(args.data, tok, max_len=args.max_len, max_pairs=args.max_pairs)

    # Model
    logger.info("[3/4] Model...")
    if args.resume_from:
        model = GNNLocalModelV2.from_pretrained(args.resume_from)
    else:
        config = GNNLocalV2Config(
            vocab_size=tok.vocab_size, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, proj_dim=args.proj_dim,
            temperature=args.temperature, dropout=args.dropout,
            activation=args.activation,
            residual_alpha_init=args.residual_alpha_init,
            lr=args.lr, weight_decay=args.weight_decay,
            max_len=args.max_len, batch_size=args.batch_size,
        )
        model = GNNLocalModelV2(config)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params: {total:,} (layers={args.num_layers})")

    # Trainer
    logger.info("[4/4] Training...")
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
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
    )

    trainer = GNNLocalV2Trainer(
        model=model, args=training_args,
        train_dataset=ds, data_collator=PairCollator(),
    )
    trainer.train()

    final = os.path.join(args.save_dir, "final")
    model.save_pretrained(final)
    logger.info(f"Done! Model saved to {final}")


if __name__ == "__main__":
    main()
