# Трудные негативы для дообучения bi-encoder.
#
# Главная оставшаяся ошибка решения - выбор среди похожих объявлений того же города: нужное объявление
# почти всегда есть в пуле кандидатов (99%), но не всегда попадает в топ-50. При обычном обучении
# негативы - случайные объявления из батча, их легко отличить. Здесь к каждой паре «запрос → выбранное
# объявление» добавляется одно похожее объявление из того же города, которое по этому запросу не выбирали.
#
# Как выбирается негатив:
#   * кандидаты - объявления корпуса в городе поиска (для поиска по региону - весь корпус),
#     отсортированные по косинусу текущей модели;
#   * первые SKIP позиций пропускаются: там чаще всего такие же подходящие объявления (ложные негативы);
#   * исключаются все объявления, которые выбирали по этому же тексту запроса;
#   * из позиций SKIP..TOP берётся одно случайное.
#
# Запуск: python -m experiments.mine_hard_negatives --model e5l
# Результат: artifacts/dense_pairs_hn.parquet (колонки query, doc, neg) для train_dense.py --pairs.

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# папку артефактов можно переопределить: AVITO_ART=artifacts_v2 python ...
ART = Path(__file__).resolve().parent.parent / os.environ.get('AVITO_ART', 'artifacts')
SKIP, TOP = 2, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='e5l')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pairs = pd.read_parquet(ART / 'dense_pairs.parquet')
    fit = pd.read_parquet(ART / 'train_fit.parquet', columns=['q_norm', 'item_id', 'search_location_id'])
    corpus = pd.read_parquet(ART / 'corpus.parquet', columns=['item_id', 'item_location_id', 'dense_text'])

    loc = fit.drop_duplicates(['q_norm', 'item_id']).set_index(['q_norm', 'item_id'])['search_location_id']
    pairs['loc'] = loc.reindex(pd.MultiIndex.from_frame(pairs[['q_norm', 'item_id']])).values
    chosen = fit.groupby('q_norm')['item_id'].apply(set)

    model = SentenceTransformer(
        str(ART / f'model_{args.model}'),
        device='cuda',
        prompts={'query': 'query: ', 'doc': 'passage: '},
        model_kwargs={'torch_dtype': torch.bfloat16},
    )
    # эмбеддинги запросов считаем по уникальным текстам
    uq = pairs['query'].drop_duplicates().tolist()
    Q = model.encode(
        uq,
        batch_size=512,
        prompt_name='query',
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    ).to(torch.float16)
    q_index = {t: i for i, t in enumerate(uq)}
    E = torch.from_numpy(np.load(ART / f'emb_{args.model}_items.npy')).to('cuda', torch.float16)
    item_loc = torch.from_numpy(corpus['item_location_id'].values.copy()).to('cuda')
    corpus_locs = set(corpus['item_location_id'])
    ids, texts = corpus['item_id'].values, corpus['dense_text'].values

    negs = []
    B = 1024
    for b in range(0, len(pairs), B):
        part = pairs.iloc[b : b + B]
        S = Q[[q_index[t] for t in part['query']]] @ E.T
        for i, loc in enumerate(part['loc'].values):
            if loc in corpus_locs:
                S[i][item_loc != int(loc)] = -10  # только тот же город
        top = torch.topk(S, TOP, dim=1).indices.cpu().numpy()
        for i, row in enumerate(part.itertuples(index=False)):
            bad = chosen.get(row.q_norm, set()) | {row.item_id}
            cand = [j for j in top[i][SKIP:] if ids[j] not in bad]
            negs.append(texts[rng.choice(cand)] if cand else None)
    pairs['neg'] = negs
    out = pairs.dropna(subset=['neg'])[['q_norm', 'query', 'doc', 'neg', 'item_id']]
    out.to_parquet(ART / 'dense_pairs_hn.parquet', index=False)
    print('пар с трудным негативом:', len(out), 'из', len(pairs), flush=True)


if __name__ == '__main__':
    main()
