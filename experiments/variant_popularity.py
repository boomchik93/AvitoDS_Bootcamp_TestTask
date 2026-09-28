# Вариант v5: к пулу v4 добавляются признаки популярности объявления в train.
#
# Считается, сколько раз объявление выбирали вообще и сколько раз по тому же тексту.
# Помогает, только если нужные объявления бенчмарка тоже часто встречаются в train.
#
# Запуск из корня проекта:
#     python experiments/variant_popularity.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ART  # noqa: E402
from src.ranking import check_answer, cv_ranker, fit_ranker, make_answer  # noqa: E402


def add_pop(pool, q, item_cnt, pair_cnt):
    # Добавляет в пул два признака популярности (в логарифме).
    item_train = pool['item_id'].map(item_cnt).fillna(0).values
    pool['item_train_cnt'] = np.log1p(item_train).astype(np.float32)
    key = pd.MultiIndex.from_arrays([q['q_norm'].values[pool['q_row'].values], pool['item_id'].values])
    pair_train = pair_cnt.reindex(key).fillna(0).values
    pool['pair_train_cnt'] = np.log1p(pair_train).astype(np.float32)
    return pool


def main():
    q_val = pd.read_parquet(ART / 'queries_val.parquet')
    q_bench = pd.read_parquet(ART / 'queries_bench.parquet')
    fit = pd.read_parquet(ART / 'train_fit.parquet', columns=['q_norm', 'item_id'])
    item_cnt = fit['item_id'].value_counts()
    pair_cnt = fit.groupby(['q_norm', 'item_id']).size()

    pv = add_pop(pd.read_parquet(ART / 'pool_val.parquet'), q_val, item_cnt, pair_cnt)
    pb = add_pop(pd.read_parquet(ART / 'pool_bench.parquet'), q_bench, item_cnt, pair_cnt)
    pos_seen = (pv.loc[pv.label == 1, 'item_train_cnt'] > 0).mean().round(3)
    bench_seen = (pb['item_train_cnt'] > 0).mean().round(3)
    val_seen = (pv['item_train_cnt'] > 0).mean().round(3)
    print(
        'share of val positives with item_train_cnt>0:',
        pos_seen,
        '| bench candidates with cnt>0:',
        bench_seen,
        '| val candidates:',
        val_seen,
        flush=True,
    )

    rec, n_iter, oof = cv_ranker(pv, q_val, rounds=2000)
    print(f'v5 (popularity) OOF Recall@50 = {rec:.4f} iters {n_iter}', flush=True)

    # отдельно смотрим запросы, где нужные объявления в train не встречались
    qv = q_val.reset_index(drop=True)
    top = oof.sort_values(['q_row', 'pred'], ascending=[True, False]).groupby('q_row').head(50)
    hit = top[top['label'] == 1].groupby('q_row').size().reindex(range(len(qv))).fillna(0) / qv['relevant'].str.len()
    fit_items = set(item_cnt.index)
    unseen_pos = qv['relevant'].map(lambda r: not any(x in fit_items for x in r)).values
    print(
        f'positives unseen in train: {hit[unseen_pos].mean():.4f} (n={unseen_pos.sum()}); '
        f'seen: {hit[~unseen_pos].mean():.4f}',
        flush=True,
    )

    # усредняем 5 моделей с разными seed
    score = np.zeros(len(pb))
    for seed in range(5):
        model, cols = fit_ranker(pv, n_iter, seed=seed)
        s = model.predict(pb[cols])
        score += (s - s.mean()) / (s.std() + 1e-9)

    ans = make_answer(pb, q_bench, score)
    corpus = pd.read_parquet(ART / 'corpus.parquet', columns=['item_id', 'in_bench'])
    check_answer(ans, q_bench, corpus.loc[corpus.in_bench == 1, 'item_id'])
    ans.to_csv('answer_v5_pop.csv', index=False)
    print('saved answer_v5_pop.csv', flush=True)


if __name__ == '__main__':
    main()
