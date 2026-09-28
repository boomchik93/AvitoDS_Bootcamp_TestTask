# Отбор кандидатов в два шага: сначала пул из разных каналов, потом LightGBM.

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .prep import recall_at_k

FIELD_WEIGHTS = {'title': 1.0, 'struct': 0.8, 'params': 0.3, 'desc': 0.3}


def zscore(x, mask):
    # Нормирует скоры по объявлениям из маски.
    m = x[mask]
    return (x - m.mean()) / (m.std() + 1e-6)


def topk_idx(score, k):
    # Индексы k лучших объявлений по скору, от лучшего к худшему.
    k = min(k, len(score) - 1)
    idx = np.argpartition(-score, k)[:k]
    return idx[np.argsort(-score[idx])]


class CandidatePool:
    # Собирает пул кандидатов и признаки для каждого запроса.
    #
    # Каждый канал (BM25, n-граммы, нейросеть, микрокатегория) вместе с гео
    # выдаёт свой топ, потом топы объединяются. Что пропустил один канал,
    # часто находит другой.

    def __init__(self, corpus, bm25, char, geo, mc, dense=None, geo_w=None, k=None, feature_only=()):
        self.corpus = corpus
        self.bm25 = bm25
        self.char = char
        self.geo = geo
        self.mc = mc
        # dense - словарь {имя модели: DenseIndex}
        self.dense = dense or {}
        # эти модели идут только в признаки, свой канал для них не делаем
        self.feature_only = set(feature_only)
        # вес гео и размер топа для каждого канала, подбирались на валидации
        self.geo_w = {'bm25': 0.4, 'char': 0.4, 'dense': 0.4, 'mc': 1.0, **(geo_w or {})}
        self.k = {'bm25': 300, 'char': 200, 'dense': 300, 'mc': 100, **(k or {})}
        self.item_mc = mc.item_class_idx(corpus)
        self.item_static_names = [
            'price_log',
            'rating',
            'reviews_log',
            'desc_len',
            'not_services',
            'phone_hidden',
            'msg_forbidden',
        ]
        self.item_static = corpus[self.item_static_names].astype('float32').values
        loc_counts = corpus['item_location_id'].value_counts()
        self.loc_size = corpus['item_location_id'].map(loc_counts).values.astype(np.float32)
        self.vid = corpus['vid'].values
        self.type = corpus['type'].values
        self.auto = corpus['auto_type'].values
        self.locs_set = set(corpus['item_location_id'])

    def channel_scores(self, scores, mask):
        # Скор каждого канала для одного запроса: текстовый скор плюс вес гео.
        g = scores['geo']
        channels = {
            'bm25': zscore(scores['bm25'], mask) + self.geo_w['bm25'] * g,
            'char': zscore(scores['char'], mask) + self.geo_w['char'] * g,
            'mc': np.log(scores['mc'] + 1e-6) + self.geo_w['mc'] * g,
        }
        for name, v in scores['dense'].items():
            if name not in self.feature_only:
                channels['dense_' + name] = zscore(v, mask) + self.geo_w['dense'] * g
        for v in channels.values():
            v[~mask] = -np.inf
        return channels

    def _k(self, name):
        return self.k['dense'] if name.startswith('dense_') else self.k[name]

    def run(self, queries, q_emb=None, allowed=None, batch=64):
        # Строит пул для всех запросов.
        #
        # q_emb - эмбеддинги запросов по моделям, allowed - маска объявлений,
        # которые можно брать в ответ.
        q_emb = q_emb or {}
        allowed = np.ones(len(self.corpus), bool) if allowed is None else allowed
        rows = []
        for b0 in range(0, len(queries), batch):
            qb = queries.iloc[b0 : b0 + batch]
            texts = qb['q_norm'].tolist()
            q_mat, ntok = self.bm25.query_matrix(texts)
            field_sc = self.bm25.score(q_mat)
            bm = sum(w * field_sc[f] for f, w in FIELD_WEIGHTS.items())
            char_sc = self.char.score(texts)
            dense_sc = {name: idx.score(q_emb[name][b0 : b0 + batch]) for name, idx in self.dense.items()}
            mc_proba = self.mc.predict_proba(qb)
            # последняя колонка нужна для неизвестной микрокатегории с индексом -1
            mc_proba = np.hstack([mc_proba, np.zeros((len(qb), 1), dtype=mc_proba.dtype)])

            for i, q in enumerate(qb.itertuples(index=False)):
                logp, same, dist = self.geo.item_features(q.search_location_id)
                scores = {
                    'bm25': bm[i],
                    'char': char_sc[i],
                    'geo': logp,
                    'mc': mc_proba[i][self.item_mc],
                    'dense': {name: d[i] for name, d in dense_sc.items()},
                }
                channels = self.channel_scores(scores, allowed)

                # объединяем топы каналов и запоминаем место объявления в каждом
                ranks = {}
                for name, sc in channels.items():
                    for r, j in enumerate(topk_idx(sc, self._k(name))):
                        ranks.setdefault(j, {})[name] = r
                idx = np.fromiter(ranks.keys(), dtype=np.int64)
                n = len(idx)

                nt = max(ntok[i], 1)
                feat = {
                    'q_row': np.full(n, b0 + i, dtype=np.int32),
                    'item_row': idx,
                    'bm25': bm[i][idx],
                    'bm25_z': zscore(bm[i], allowed)[idx],
                    'bm25_rel': bm[i][idx] / (bm[i][allowed].max() + 1e-6),
                    'char': char_sc[i][idx],
                    'char_rel': char_sc[i][idx] / (char_sc[i][allowed].max() + 1e-6),
                    'cover_title': field_sc['match_title'][i][idx] / nt,
                    'cover_any': field_sc['match_any'][i][idx] / nt,
                    'geo_logp': logp[idx],
                    'same_loc': same[idx],
                    'dist_km': dist[idx],
                    'mc_p': scores['mc'][idx],
                    'mc_rel': scores['mc'][idx] / (mc_proba[i].max() + 1e-9),
                    'loc_size': np.log1p(self.loc_size[idx]),
                }
                for f in FIELD_WEIGHTS:
                    feat['bm25_' + f] = field_sc[f][i][idx]
                for name, d in scores['dense'].items():
                    feat[f'dense_{name}'] = d[idx]
                    feat[f'dense_{name}_z'] = zscore(d, allowed)[idx]
                    feat[f'dense_{name}_gap'] = d[idx] - d[allowed].max()
                for name in channels:
                    feat['rank_' + name] = np.array([ranks[j].get(name, self._k(name)) for j in idx], dtype=np.float32)
                feat['n_channels'] = np.array([len(ranks[j]) for j in idx], dtype=np.float32)

                # совпадают ли фильтры запроса с параметрами объявления
                no_filter = np.full(n, np.nan, np.float32)
                if q.f_vid:
                    feat['vid_match'] = (self.vid[idx] == q.f_vid).astype(np.float32)
                else:
                    feat['vid_match'] = no_filter
                if q.f_type:
                    type_match = (self.type[idx] == q.f_type) | (self.auto[idx] == q.f_type)
                    feat['type_match'] = type_match.astype(np.float32)
                else:
                    feat['type_match'] = no_filter.copy()

                feat['q_ntok'] = np.full(n, ntok[i], np.float32)
                feat['q_has_filter'] = np.full(n, float(bool(q.f_vid or q.f_type)), np.float32)
                feat['q_loc_is_city'] = np.full(n, float(q.search_location_id in self.locs_set), np.float32)
                feat['q_rating_filter'] = np.full(n, float(q.f_rating), np.float32)
                block = pd.DataFrame(feat)

                # место объявления внутри пула: важно, насколько оно лучше соседей
                prank_cols = ['bm25', 'char', 'mc_p', 'geo_logp']
                prank_cols += [f'dense_{name}' for name in scores['dense']]
                for col in prank_cols:
                    block[col + '_prank'] = block[col].rank(ascending=False, method='min').astype(np.float32)
                static = pd.DataFrame(self.item_static[idx], columns=self.item_static_names)
                rows.append(pd.concat([block, static], axis=1))

        pool = pd.concat(rows, ignore_index=True)
        pool['item_id'] = self.corpus['item_id'].values[pool['item_row'].values]
        return pool


def add_labels(pool, queries):
    # Ставит метку 1 объявлениям, которые выбрали по запросу.
    rel_sets = [set(r) for r in queries['relevant']]
    pool['label'] = np.fromiter(
        (it in rel_sets[q] for q, it in zip(pool['q_row'].values, pool['item_id'].values)),
        dtype=np.int8,
        count=len(pool),
    )
    return pool


def pool_recall(pool, queries):
    # Доля нужных объявлений, которые вообще попали в пул.
    got = pool[pool['label'] == 1].groupby('q_row').size()
    n_rel = queries['relevant'].str.len().values
    return float((got.reindex(range(len(queries))).fillna(0).values / n_rel).mean())


def recall_by_score(pool, queries, score, k=50):
    # Recall@k, если брать топ-k пула по скору.
    p = pool[['q_row', 'item_id']].assign(score=np.asarray(score))
    top = p.sort_values(['q_row', 'score'], ascending=[True, False]).groupby('q_row').head(k)
    pred = top.groupby('q_row')['item_id'].apply(list).reindex(range(len(queries)))
    pred = [x if isinstance(x, list) else [] for x in pred]
    return recall_at_k(pred, queries['relevant'].tolist(), k)


FEATURE_EXCLUDE = {'q_row', 'item_row', 'item_id', 'label', 'pred'}

LGB_PARAMS = dict(
    objective='lambdarank',
    metric='ndcg',
    eval_at=[50],
    lambdarank_truncation_level=60,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
)


def feature_cols(pool):
    return [c for c in pool.columns if c not in FEATURE_EXCLUDE]


def _groups(df):
    return df.groupby('q_row', sort=False).size().values


def cv_ranker(pool, queries, n_folds=5, rounds=1500, seed=0, params=None):
    # Кросс-валидация по запросам, возвращает Recall@50, число итераций и предсказания.
    params = {**LGB_PARAMS, **(params or {}), 'seed': seed}
    pool = pool.sort_values('q_row', kind='stable').reset_index(drop=True)
    cols = feature_cols(pool)
    oof = np.zeros(len(pool))
    best_iters = []
    for tr, va in GroupKFold(n_splits=n_folds).split(pool, groups=pool['q_row']):
        dtr, dva = pool.iloc[tr], pool.iloc[va]
        # запросы, у которых нет нужного объявления в пуле, для обучения бесполезны
        dtr = dtr[dtr.groupby('q_row')['label'].transform('max') > 0]
        ds_tr = lgb.Dataset(dtr[cols], dtr['label'], group=_groups(dtr))
        ds_va = lgb.Dataset(dva[cols], dva['label'], group=_groups(dva))
        model = lgb.train(
            params,
            ds_tr,
            rounds,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iters.append(model.best_iteration)
        oof[va] = model.predict(dva[cols], num_iteration=model.best_iteration)
    return recall_by_score(pool, queries, oof), int(np.mean(best_iters)), pool.assign(pred=oof)


def fit_ranker(pool, rounds, seed=0, params=None):
    # Обучает ранжировщик на всём пуле.
    params = {**LGB_PARAMS, **(params or {}), 'seed': seed}
    pool = pool.sort_values('q_row', kind='stable').reset_index(drop=True)
    d = pool[pool.groupby('q_row')['label'].transform('max') > 0]
    cols = feature_cols(pool)
    model = lgb.train(params, lgb.Dataset(d[cols], d['label'], group=_groups(d)), rounds)
    return model, cols


def make_answer(pool, queries, score, k=50):
    # Берёт топ-k по скору для каждого запроса и делает таблицу ответа.
    p = (
        pool[['q_row', 'item_id']]
        .assign(score=np.asarray(score))
        .sort_values(['q_row', 'score'], ascending=[True, False])
    )
    p = p.drop_duplicates(['q_row', 'item_id']).groupby('q_row').head(k)
    lists = p.groupby('q_row')['item_id'].apply(list).reindex(range(len(queries)))
    lists = [x if isinstance(x, list) else [] for x in lists]
    return pd.DataFrame(
        {
            'query_id': queries['query_id'].values,
            'answer': [' '.join(x) for x in lists],
        }
    )


def check_answer(ans, bench_queries, bench_item_ids):
    # Проверяет формат ответа по условию задачи.
    assert list(ans.columns) == ['query_id', 'answer']
    assert ans['query_id'].is_unique
    assert set(ans['query_id']) == set(bench_queries['query_id'])
    ids = set(bench_item_ids)
    for a in ans['answer']:
        lst = a.split(' ')
        assert 0 < len(lst) <= 50 and len(set(lst)) == len(lst), 'bad length / duplicates'
        assert all(len(x) == 16 and x in ids for x in lst), 'unknown item_id'
    return True
