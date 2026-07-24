#!/usr/bin/env python3
"""Train GraphTransformer v2 on full augmented dataset."""
import os, sys, json, logging, argparse, random, math
import torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_graph_transformer import GraphTransformerConfig, GraphTransformerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Tokenizer ──
def build_tokenizer(lines, vocab_size=16200):
    bpe = Tokenizer(models.BPE(unk_token='[UNK]'))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'],
        min_frequency=2)
    tmp = '/tmp/corpus_gtv2_full.txt'
    with open(tmp, 'w') as f:
        for line in lines:
            f.write(line.strip() + '\n')
    bpe.train([tmp], trainer)
    bpe.post_processor = processors.ByteLevel(trim_offsets=False)
    bpe.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=bpe,
        unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]',
        sep_token='[SEP]', mask_token='[MASK]')

# ── Dataset (supports both string and list positives) ──
class PairDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=64, max_pairs=None):
        self.pairs = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_pairs and i >= max_pairs: break
                d = json.loads(line)
                q = str(d['query']).strip()
                p = d.get('positive', '')
                n = d.get('negative', [])
                if isinstance(p, list):
                    p = p[0] if p else ''
                p = str(p).strip()
                if not q or not p: continue
                q_ids = tokenizer.encode(q, add_special_tokens=False)[:max_len]
                p_ids = tokenizer.encode(p, add_special_tokens=False)[:max_len]
                if len(q_ids) < 2 or len(p_ids) < 2: continue
                
                entry = {'query_ids': q_ids, 'pos_ids': p_ids}
                # Encode explicit negatives (up to 5)
                if isinstance(n, list) and n:
                    neg_ids = []
                    for nt in n[:5]:
                        nt = str(nt).strip()
                        n_ids = tokenizer.encode(nt, add_special_tokens=False)[:max_len]
                        if len(n_ids) >= 2:
                            neg_ids.append(n_ids)
                    if neg_ids:
                        entry['neg_ids'] = neg_ids
                self.pairs.append(entry)
        logger.info(f'Dataset: {len(self.pairs):,} pairs, '
                    f'{sum(1 for p in self.pairs if "neg_ids" in p)} with explicit negs')
    
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return dict(self.pairs[i])

# ── Collator ──
class PairCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id
    
    def __call__(self, batch):
        q = nn.utils.rnn.pad_sequence(
            [torch.tensor(d['query_ids'], dtype=torch.long) for d in batch],
            batch_first=True, padding_value=self.pad_id)
        p = nn.utils.rnn.pad_sequence(
            [torch.tensor(d['pos_ids'], dtype=torch.long) for d in batch],
            batch_first=True, padding_value=self.pad_id)
        
        result = {'query_ids': q, 'pos_ids': p}
        
        # Collect explicit negatives
        neg_entries = [d.get('neg_ids', []) for d in batch]
        has_negs = any(neg_entries)
        if has_negs:
            max_k = max((len(ne) for ne in neg_entries), default=0)
            if max_k > 0:
                B = len(batch)
                max_l = max((len(n) for ne in neg_entries for n in ne), default=1)
                neg_t = torch.full((B, max_k, max_l), self.pad_id, dtype=torch.long)
                neg_mask = torch.zeros(B, max_k, dtype=torch.bool)
                for b, ne in enumerate(neg_entries):
                    for k, n_ids in enumerate(ne[:max_k]):
                        neg_t[b, k, :len(n_ids)] = torch.tensor(n_ids, dtype=torch.long)
                        neg_mask[b, k] = True
                result['neg_ids'] = neg_t
                result['neg_mask'] = neg_mask
        
        return result

# ── Trainer with explicit negatives ──
class GTV2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        q_ids = inputs['query_ids']
        p_ids = inputs['pos_ids']
        neg_ids = inputs.get('neg_ids', None)
        neg_mask = inputs.get('neg_mask', None)
        
        if neg_ids is not None and neg_mask is not None and neg_mask.any():
            B, K, L = neg_ids.shape
            valid_k = min(K, 5)
            neg_flat = neg_ids[:, :valid_k, :].reshape(B * valid_k, L)
            mask_flat = neg_mask[:, :valid_k].reshape(-1)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]
            
            if len(valid_idx) > 0:
                neg_flat = neg_flat[valid_idx]
                q_emb, p_emb, n_emb = model(q_ids, p_ids, neg_flat)
                loss = model.infonce_loss(q_emb, p_emb, neg_emb=n_emb,
                                          temperature=model.config.temperature)
            else:
                q_emb, p_emb = model(q_ids, p_ids)
                loss = model.infonce_loss(q_emb, p_emb,
                                          temperature=model.config.temperature)
        else:
            q_emb, p_emb = model(q_ids, p_ids)
            loss = model.infonce_loss(q_emb, p_emb,
                                      temperature=model.config.temperature)
        return (loss, {}) if return_outputs else loss

# ── Main ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='/root/autodl-tmp/train_ln/train_final.jsonl')
    p.add_argument('--save_dir', default='/root/autodl-tmp/train_ln/checkpoints_gtv2_full')
    p.add_argument('--max_pairs', type=int, default=None)
    p.add_argument('--vocab_size', type=int, default=16200)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--ff_mult', type=int, default=4)
    p.add_argument('--proj_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', default='gelu')
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--adj_mode', default='cosine')
    p.add_argument('--msg_agg', default='mean')
    p.add_argument('--bm25_k1', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=192)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--max_len', type=int, default=64)
    p.add_argument('--log_steps', type=int, default=200)
    p.add_argument('--save_steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume_from', type=str, default=None)
    args = p.parse_args()
    
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
    
    # Tokenizer
    logger.info('[1/4] Building tokenizer...')
    # Build tokenizer from pretrain corpus
    corpus = []
    with open('/root/train_ln/pretrain_corpus.txt') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(line)
    tok = build_tokenizer(corpus, args.vocab_size)
    logger.info(f'  Vocab={tok.vocab_size}')
    
    # Dataset
    dataset = PairDataset(args.data, tok, args.max_len, args.max_pairs)
    total_steps = len(dataset) // args.batch_size * args.epochs
    logger.info(f'  Pairs={len(dataset):,}, Epochs={args.epochs}, Batch={args.batch_size}')
    logger.info(f'  Steps/epoch={len(dataset)//args.batch_size:,}, Total steps~={total_steps:,}')
    
    # Model
    logger.info('[2/4] Building model...')
    config = GraphTransformerConfig(
        vocab_size=args.vocab_size, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        ff_mult=args.ff_mult, proj_dim=args.proj_dim,
        dropout=args.dropout, activation=args.activation,
        temperature=args.temperature, adj_mode=args.adj_mode,
        msg_agg=args.msg_agg, bm25_k1=args.bm25_k1, max_len=args.max_len)
    model = GraphTransformerModel(config)
    if args.resume_from:
        logger.info(f'  Resuming from {args.resume_from}')
        model = GraphTransformerModel.from_pretrained(args.resume_from)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Params={params:,} ({params/1e6:.1f}M) Layers={args.num_layers}')
    logger.info(f'  Heads={args.num_heads} Adj={args.adj_mode} Temp={args.temperature}')
    
    # Training
    logger.info('[3/4] Starting training...')
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )
    
    collator = PairCollator(pad_id=tok.pad_token_id)
    trainer = GTV2Trainer(model=model, args=training_args,
                          train_dataset=dataset, data_collator=collator)
    trainer.train()
    
    # Save
    logger.info(f'[4/4] Saving to {args.save_dir}/final')
    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok=True)
    model.save_pretrained(os.path.join(args.save_dir, 'final'))
    tok.save_pretrained(os.path.join(args.save_dir, 'final'))
    logger.info('Done!')

if __name__ == '__main__':
    main()
