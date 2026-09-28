# Источники сигнала: каждый выдаёт скор для всех объявлений корпуса сразу.

from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .prep import stem_tokens


class FieldBM25:
    # BM25 отдельно по каждому полю объявления с общим словарём.
    #
    # Раздельные скоры нужны, потому что слово в заголовке важнее, чем в описании.
    # Общий словарь нужен, чтобы посчитать, сколько слов запроса нашлось в объявлении.

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}

    def _count_matrix(self, docs):
        # Считает, сколько раз каждое слово встречается в каждом документе.
        indptr, indices, data = [0], [], []
        for tokens in docs:
            cnt = Counter()
            for w in tokens:
                j = self.vocab.get(w)
                if j is None:
                    j = self.vocab[w] = len(self.vocab)
                cnt[j] += 1
            indices.extend(cnt.keys())
            data.extend(cnt.values())
            indptr.append(len(indices))
        return indptr, indices, data

    def fit(self, fields):
        # Обучается на словаре {имя поля: список текстов}.
        raw = {name: self._count_matrix(map(stem_tokens, texts)) for name, texts in fields.items()}
        n_words = len(self.vocab)
        self.fields = list(fields)
        # матрицы хранятся транспонированными, так быстрее умножать на запросы
        self.weights_T = {}
        any_bin = None
        for name, (indptr, indices, data) in raw.items():
            x = sp.csr_matrix(
                (np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
                shape=(len(indptr) - 1, n_words),
            )
            n_docs = x.shape[0]
            doc_freq = np.bincount(x.indices, minlength=n_words)
            idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
            doc_len = np.asarray(x.sum(1)).ravel()
            avg_len = max(doc_len.mean(), 1e-6)
            row_norm = np.repeat(self.k1 * (1 - self.b + self.b * doc_len / avg_len), np.diff(x.indptr)).astype(
                np.float32
            )
            w = x.copy()
            w.data = idf[w.indices] * w.data * (self.k1 + 1) / (w.data + row_norm)
            self.weights_T[name] = w.T.tocsr()

            # отмечаем, есть ли слово хотя бы в одном поле
            binary = x.copy()
            binary.data[:] = 1
            any_bin = binary if any_bin is None else any_bin + binary
        any_bin.data[:] = 1
        self.any_T = any_bin.T.tocsr().astype(np.float32)

        indptr, indices, _ = raw['title']
        title = sp.csr_matrix(
            (np.ones(len(indices), dtype=np.float32), np.array(indices), np.array(indptr)),
            shape=(len(indptr) - 1, n_words),
        )
        self.title_T = title.T.tocsr()
        return self

    def query_matrix(self, texts):
        # Переводит запросы в матрицу слов и считает число слов в каждом запросе.
        rows, cols, n_tok = [], [], []
        for i, text in enumerate(texts):
            tokens = stem_tokens(text)
            ids = {self.vocab[w] for w in tokens if w in self.vocab}
            n_tok.append(len(set(tokens)))
            rows.extend([i] * len(ids))
            cols.extend(ids)
        q = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(texts), len(self.vocab)),
        )
        return q, np.array(n_tok)

    def score(self, q):
        # Считает BM25 по каждому полю и сколько слов запроса нашлось.
        out = {name: (q @ w_t).toarray() for name, w_t in self.weights_T.items()}
        out['match_title'] = (q @ self.title_T).toarray()
        out['match_any'] = (q @ self.any_T).toarray()
        return out


class CharTfidf:
    # TF-IDF по кусочкам слов из 3-5 букв.
    #
    # Помогает, когда в запросе опечатка или слово написано слитно или раздельно.

    def __init__(self, max_features=400_000):
        self.vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
            lowercase=True,
        )

    def fit(self, texts):
        x = self.vec.fit_transform([t.replace('ё', 'е') for t in texts])
        self.XT = x.T.tocsr()
        return self

    def score(self, texts):
        q = self.vec.transform([t.replace('ё', 'е') for t in texts])
        return (q @ self.XT).toarray()


def haversine(lat1, lon1, lat2, lon2):
    # Расстояние между точками на Земле в километрах.
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GeoPrior:
    # Вероятность, что при поиске в одной локации выберут объявление из другой.
    #
    # Считается по train. Чтобы близкие города без пар в train не получали ноль,
    # к счётчикам добавляется поправка, которая падает с расстоянием.

    def __init__(self, beta=3.0, tau_km=40.0, floor=1e-7):
        self.beta = beta
        self.tau = tau_km
        self.floor = floor

    def fit(self, fit_rows, items):
        # центр локации - медиана координат её объявлений
        cols = ['item_location_id', 'item_latitude', 'item_longitude']
        geo = pd.concat([fit_rows[cols], items[cols]])
        self.center = geo.groupby('item_location_id')[['item_latitude', 'item_longitude']].median()
        # для поиска по региону центр считаем по выбранным объявлениям
        s_center = fit_rows.groupby('search_location_id')[['item_latitude', 'item_longitude']].median()
        self.s_center = s_center.combine_first(self.center)

        self.locs = np.array(sorted(items['item_location_id'].unique()))
        self.loc_index = {loc: i for i, loc in enumerate(self.locs)}
        self.item_loc_idx = items['item_location_id'].map(self.loc_index).values
        self.loc_lat = self.center.reindex(self.locs)['item_latitude'].values
        self.loc_lon = self.center.reindex(self.locs)['item_longitude'].values
        self.item_lat = items['item_latitude'].values
        self.item_lon = items['item_longitude'].values

        cnt = fit_rows.groupby(['search_location_id', 'item_location_id']).size()
        self.counts = {s: grp.droplevel(0) for s, grp in cnt.groupby(level=0)}
        self._cache = {}
        return self

    def loc_logprob(self, s):
        # Логарифм вероятности каждой локации корпуса для локации поиска s.
        if s in self._cache:
            return self._cache[s]
        counts = np.zeros(len(self.locs))
        if s in self.counts:
            ser = self.counts[s]
            idx = ser.index.map(self.loc_index)
            ok = ~pd.isna(idx)
            counts[np.asarray(idx[ok], dtype=int)] = ser.values[ok]
        near = np.zeros(len(self.locs))
        if s in self.s_center.index:
            lat, lon = self.s_center.loc[s]
            dist = haversine(lat, lon, self.loc_lat, self.loc_lon)
            near = np.nan_to_num(np.exp(-dist / self.tau))
        if s in self.loc_index:
            near[self.loc_index[s]] = max(near[self.loc_index[s]], 1.0)
        p = (counts + self.beta * near) / (counts.sum() + self.beta * near.sum() + 1e-9)
        out = np.log(p + self.floor).astype(np.float32)
        self._cache[s] = out
        return out

    def item_features(self, s):
        # Для всех объявлений: логарифм вероятности, та же ли локация и расстояние в км.
        logp = self.loc_logprob(s)[self.item_loc_idx]
        same = (self.locs[self.item_loc_idx] == s).astype(np.float32)
        if s in self.s_center.index:
            lat, lon = self.s_center.loc[s]
            dist = haversine(lat, lon, self.item_lat, self.item_lon).astype(np.float32)
        else:
            dist = np.full(len(logp), np.nan, dtype=np.float32)
        return logp, same, dist


class MicrocatModel:
    # Угадывает микрокатегорию по тексту запроса и фильтрам.
    #
    # Обучается на уникальных текстах, чтобы частые запросы не перетягивали модель.

    def __init__(self):
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
            tokenizer=stem_tokens,
            token_pattern=None,
            lowercase=False,
        )
        self.char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=3,
            sublinear_tf=True,
            dtype=np.float32,
            max_features=200_000,
        )
        self.clf = SGDClassifier(
            loss='log_loss',
            alpha=2e-6,
            max_iter=15,
            tol=None,
            n_jobs=-1,
            random_state=0,
        )

    @staticmethod
    def _text(df):
        return (df['q_norm'] + ' | ' + df['f_vid'].fillna('') + ' ' + df['f_type'].fillna('')).tolist()

    def _features(self, texts, fit=False):
        if fit:
            parts = [self.word.fit_transform(texts), self.char.fit_transform(texts)]
        else:
            parts = [self.word.transform(texts), self.char.transform(texts)]
        return sp.hstack(parts).tocsr()

    def fit(self, fit_rows):
        g = fit_rows.groupby(['q_norm', 'f_vid', 'f_type', 'item_microcat_id']).size().rename('n').reset_index()
        x = self._features(self._text(g), fit=True)
        # вес - корень из числа выборов, чтобы частые пары были важнее, но не сильно
        self.clf.fit(x, g['item_microcat_id'].values, sample_weight=np.sqrt(g['n'].values))
        self.classes = self.clf.classes_
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        return self

    def predict_proba(self, queries):
        return self.clf.predict_proba(self._features(self._text(queries)))

    def item_class_idx(self, items):
        # микрокатегории, которых не было в train, получают индекс -1
        return items['item_microcat_id'].map(self.class_index).fillna(-1).astype(int).values


class DenseIndex:
    # Косинус между эмбеддингами запросов и объявлений.

    def __init__(self, item_emb):
        self.E = np.ascontiguousarray(item_emb, dtype=np.float32)

    def score(self, q_emb):
        # эмбеддинги уже нормированы, поэтому хватает скалярного произведения
        return q_emb.astype(np.float32) @ self.E.T
