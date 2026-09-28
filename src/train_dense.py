# Дообучение нейросети на парах (запрос, выбранное объявление) и расчёт эмбеддингов.
#
# Запуск:
#     python -m src.train_dense --model intfloat/multilingual-e5-base --name e5b

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# папку артефактов можно переопределить: AVITO_ART=artifacts_v2 python ...
ART = Path(__file__).resolve().parent.parent / os.environ.get('AVITO_ART', 'artifacts')


def encode_all(model, name, max_len, bs=256):
    # Считает эмбеддинги корпуса и запросов и сохраняет их в artifacts.
    model.max_seq_length = max_len
    corpus = pd.read_parquet(ART / 'corpus.parquet', columns=['dense_text'])
    out = {'items': (corpus['dense_text'].tolist(), 'doc')}
    for split in ['val', 'bench']:
        q = pd.read_parquet(ART / f'queries_{split}.parquet', columns=['dense_query'])
        out[f'q_{split}'] = (q['dense_query'].tolist(), 'query')
    for key, (texts, prompt) in out.items():
        emb = model.encode(
            texts,
            batch_size=bs,
            prompt_name=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=key == 'items',
        )
        np.save(ART / f'emb_{name}_{key}.npy', emb.astype(np.float16))
        print(key, emb.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='intfloat/multilingual-e5-base')
    ap.add_argument('--name', default='e5b')
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--mini_bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--max_len', type=int, default=160)
    ap.add_argument('--pairs', default='dense_pairs.parquet')
    ap.add_argument('--max_steps', type=int, default=-1)
    ap.add_argument('--encode_only', action='store_true')
    args = ap.parse_args()

    if 'e5' in args.model:
        prompts = {'query': 'query: ', 'doc': 'passage: '}
    else:
        prompts = {'query': '', 'doc': ''}
    model_kwargs = {'torch_dtype': torch.bfloat16} if args.encode_only else {}
    model = SentenceTransformer(args.model, device='cuda', prompts=prompts, model_kwargs=model_kwargs)
    model.max_seq_length = args.max_len
    out_dir = ART / f'model_{args.name}'

    if not args.encode_only:
        pairs = pd.read_parquet(ART / args.pairs)
        cols = ['query', 'doc'] + (['neg'] if 'neg' in pairs else [])
        train_ds = Dataset.from_pandas(
            pairs[cols].sample(frac=1.0, random_state=0),
            preserve_index=False,
        )
        # другие объявления из батча служат примерами «не того» ответа;
        # батч считается по кусочкам mini_bs, чтобы влезть в память видеокарты
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_bs)
        train_prompts = {'query': prompts['query'], 'doc': prompts['doc']}
        if 'neg' in cols:
            train_prompts['neg'] = prompts['doc']
        targs = SentenceTransformerTrainingArguments(
            output_dir=str(out_dir) + '_ckpt',
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.bs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type='cosine',
            bf16=True,
            logging_steps=50,
            save_strategy='no',
            report_to='none',
            # одинаковые запросы в одном батче мешали бы обучению
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            dataloader_num_workers=0,
            prompts=train_prompts,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            loss=loss,
        )
        trainer.train()
        model.save(str(out_dir))
        model = SentenceTransformer(
            str(out_dir), device='cuda', prompts=prompts, model_kwargs={'torch_dtype': torch.bfloat16}
        )
    encode_all(model, args.name, args.max_len)


if __name__ == '__main__':
    main()
