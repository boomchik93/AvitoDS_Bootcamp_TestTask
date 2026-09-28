# Реранкер: модель смотрит на запрос и объявление вместе и пересчитывает скор топа.
#
# Запуск:
#     python -m experiments.train_reranker --stage data
#     python -m experiments.train_reranker --stage train
#     python -m experiments.train_reranker --stage score --top 100

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# папку артефактов можно переопределить: AVITO_ART=artifacts_v2 python ...
ART = Path(__file__).resolve().parent.parent / os.environ.get('AVITO_ART', 'artifacts')
RERANK_BASE = 'BAAI/bge-reranker-v2-m3'


def build_data(n_queries=40000, n_neg=4, top=30, seed=0):
    # Собирает пары для обучения: выбранное объявление и похожие, но не выбранные.
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(seed)
    pairs = pd.read_parquet(ART / 'dense_pairs.parquet')
    fit = pd.read_parquet(
        ART / 'train_fit.parquet',
        columns=['q_norm', 'item_id', 'search_location_id'],
    )
    corpus = pd.read_parquet(
        ART / 'corpus.parquet',
        columns=['item_id', 'item_location_id', 'dense_text'],
    )

    # берём по одному примеру на текст, чтобы формулировки были разные
    n_texts = min(n_queries, pairs['q_norm'].nunique())
    ex = pairs.drop_duplicates('q_norm').sample(n_texts, random_state=seed)
    loc = fit.drop_duplicates(['q_norm', 'item_id']).set_index(['q_norm', 'item_id'])['search_location_id']
    ex['loc'] = loc.reindex(pd.MultiIndex.from_frame(ex[['q_norm', 'item_id']])).values
    chosen = fit.groupby('q_norm')['item_id'].apply(set)

    model = SentenceTransformer(
        str(ART / 'model_e5l'),
        device='cuda',
        prompts={'query': 'query: ', 'doc': 'passage: '},
        model_kwargs={'torch_dtype': torch.bfloat16},
    )
    q_emb = model.encode(
        ex['query'].tolist(),
        batch_size=256,
        prompt_name='query',
        normalize_embeddings=True,
        convert_to_tensor=True,
    ).to(torch.float16)
    item_emb = torch.from_numpy(np.load(ART / 'emb_e5l_items.npy')).to('cuda', torch.float16)
    item_loc = torch.from_numpy(corpus['item_location_id'].values).to('cuda')
    corpus_locs = set(corpus['item_location_id'])
    ids = corpus['item_id'].values

    rows = []
    for b in range(0, len(ex), 512):
        sim = q_emb[b : b + 512] @ item_emb.T
        locs = ex['loc'].values[b : b + 512]
        for i, loc_id in enumerate(locs):
            # негативы берём из той же локации, для поиска по региону - из всего корпуса
            if loc_id in corpus_locs:
                sim[i][item_loc != int(loc_id)] = -10
        top_idx = torch.topk(sim, top, dim=1).indices.cpu().numpy()
        for i, row in enumerate(ex.iloc[b : b + 512].itertuples(index=False)):
            # объявления, которые выбирали по этому тексту, в негативы не берём
            bad = chosen.get(row.q_norm, set()) | {row.item_id}
            cand = [j for j in top_idx[i] if ids[j] not in bad]
            if cand:
                negs = rng.choice(cand, size=min(n_neg, len(cand)), replace=False)
            else:
                negs = []
            rows.append((row.query, row.doc, 1.0))
            rows.extend((row.query, corpus['dense_text'].values[j], 0.0) for j in negs)
    data = pd.DataFrame(rows, columns=['query', 'doc', 'label'])
    data.to_parquet(ART / 'rerank_train.parquet', index=False)
    print('rerank pairs:', data.shape, data['label'].mean().round(3))


def train(epochs=1, bs=32, lr=2e-5, max_len=192):
    # Дообучает реранкер на парах из build_data.
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    data = pd.read_parquet(ART / 'rerank_train.parquet').sample(frac=1.0, random_state=0)
    model = CrossEncoder(RERANK_BASE, num_labels=1, max_length=max_len, device='cuda')
    # на один позитив приходится 4 негатива, поэтому позитивам вес 4
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(4.0))
    args = CrossEncoderTrainingArguments(
        output_dir=str(ART / 'model_rerank_ckpt'),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        bf16=True,
        logging_steps=100,
        save_strategy='no',
        report_to='none',
        dataloader_num_workers=0,
        # с AdamW модель не помещалась в 16 ГБ видеокарты, Adafactor ест меньше памяти
        optim='adafactor',
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_pandas(data, preserve_index=False),
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(ART / 'model_rerank'))


def score(top=100, bs=256, max_len=192):
    # Считает скор реранкера для топа кандидатов после первого ранжировщика.
    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(
        str(ART / 'model_rerank'),
        max_length=max_len,
        device='cuda',
        model_kwargs={'torch_dtype': torch.bfloat16},
    )
    corpus = pd.read_parquet(ART / 'corpus.parquet', columns=['item_id', 'dense_text'])
    corpus = corpus.set_index('item_id')['dense_text']
    for split, fname in [('val', 'stage1_oof_val.parquet'), ('bench', 'stage1_bench.parquet')]:
        st = pd.read_parquet(ART / fname)
        q = pd.read_parquet(ART / f'queries_{split}.parquet', columns=['dense_query'])
        st = (
            st.sort_values(['q_row', 'pred'], ascending=[True, False]).groupby('q_row').head(top).reset_index(drop=True)
        )
        pairs = list(
            zip(
                q['dense_query'].values[st['q_row'].values],
                corpus.reindex(st['item_id']).values,
            )
        )
        # сортируем по длине, чтобы в батче было меньше лишних паддингов
        order = np.argsort([len(a) + len(b) for a, b in pairs])
        s = model.predict(
            [pairs[i] for i in order],
            batch_size=bs,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        out = np.empty(len(pairs), dtype=np.float32)
        out[order] = s
        st['ce'] = out
        st[['q_row', 'item_id', 'ce']].to_parquet(ART / f'rerank_{split}.parquet', index=False)
        print(split, st.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['data', 'train', 'score'])
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--n_queries', type=int, default=40000)
    ap.add_argument('--epochs', type=float, default=1)
    args = ap.parse_args()
    if args.stage == 'data':
        build_data(n_queries=args.n_queries)
    elif args.stage == 'train':
        train(epochs=args.epochs)
    else:
        score(top=args.top)


if __name__ == '__main__':
    main()
