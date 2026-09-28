# Собирает все источники и пулы кандидатов для ноутбука 04.

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ranking import CandidatePool, add_labels
from .retrieval import CharTfidf, DenseIndex, FieldBM25, GeoPrior, MicrocatModel

# папку артефактов можно переопределить: AVITO_ART=artifacts_v2 python ...
ART = Path(__file__).resolve().parent.parent / os.environ.get('AVITO_ART', 'artifacts')


def load_artifacts():
    # Загружает корпус, запросы и обучающие строки из artifacts.
    corpus = pd.read_parquet(ART / 'corpus.parquet')
    q_val = pd.read_parquet(ART / 'queries_val.parquet')
    q_bench = pd.read_parquet(ART / 'queries_bench.parquet')
    fit = pd.read_parquet(ART / 'train_fit.parquet')
    for q in (q_val, q_bench):
        for col in ['f_vid', 'f_type', 'f_auto']:
            q[col] = q[col].fillna('')
    return corpus, q_val, q_bench, fit


def build_sources(corpus, fit, cache=True):
    # Обучает BM25, TF-IDF, гео и микрокатегории, сохраняет их в файл.
    path = ART / 'sources.pkl'
    if cache and path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)

    start = time.time()
    bm25 = FieldBM25().fit(
        {
            'title': corpus['title'].tolist(),
            'struct': corpus['struct'].tolist(),
            'params': corpus['params'].tolist(),
            'desc': corpus['desc'].tolist(),
        }
    )
    print(f'BM25: словарь {len(bm25.vocab):,} стемов, {time.time() - start:.0f} c', flush=True)

    start = time.time()
    char_texts = corpus['title'] + ' ' + corpus['struct'].str.slice(0, 300)
    char = CharTfidf().fit(char_texts.tolist())
    print(f'Char TF-IDF: {len(char.vec.vocabulary_):,} n-грамм, {time.time() - start:.0f} c', flush=True)

    start = time.time()
    geo = GeoPrior().fit(fit, corpus)
    print(f'Гео-приор: {len(geo.counts):,} локаций поиска, {time.time() - start:.0f} c', flush=True)

    start = time.time()
    mc = MicrocatModel().fit(fit)
    print(f'Микрокатегории: {len(mc.classes)} классов, {time.time() - start:.0f} c', flush=True)

    sources = (bm25, char, geo, mc)
    if cache:
        with open(path, 'wb') as f:
            pickle.dump(sources, f, protocol=5)
    return sources


def load_dense(names):
    # Загружает эмбеддинги от train_dense.py.
    #
    # Для каждой модели возвращает индекс по корпусу и эмбеддинги запросов
    # для валидации и бенчмарка.
    out = {}
    for name in names:
        items = np.load(ART / f'emb_{name}_items.npy').astype(np.float32)
        q_val = np.load(ART / f'emb_{name}_q_val.npy').astype(np.float32)
        q_bench = np.load(ART / f'emb_{name}_q_bench.npy').astype(np.float32)
        out[name] = (DenseIndex(items), q_val, q_bench)
    return out


def build_pools(corpus, q_val, q_bench, sources, dense=None, geo_w=None, k=None, val_limit=None, feature_only=()):
    # Строит пулы кандидатов для валидации (с метками) и для бенчмарка.
    bm25, char, geo, mc = sources
    dense = dense or {}
    pool_maker = CandidatePool(
        corpus,
        bm25,
        char,
        geo,
        mc,
        dense={name: d[0] for name, d in dense.items()},
        geo_w=geo_w,
        k=k,
        feature_only=feature_only,
    )

    qv = q_val if val_limit is None else q_val.iloc[:val_limit]
    start = time.time()
    pool_val = pool_maker.run(qv, q_emb={name: d[1][: len(qv)] for name, d in dense.items()})
    pool_val = add_labels(pool_val, qv)
    print(f'Пул валидации: {len(pool_val):,} строк, {time.time() - start:.0f} c', flush=True)

    pool_bench = None
    if q_bench is not None:
        start = time.time()
        # для бенчмарка можно брать только объявления из его корпуса
        pool_bench = pool_maker.run(
            q_bench,
            q_emb={name: d[2] for name, d in dense.items()},
            allowed=corpus['in_bench'].values.astype(bool),
        )
        print(f'Пул бенчмарка: {len(pool_bench):,} строк, {time.time() - start:.0f} c', flush=True)
    return pool_val, pool_bench
