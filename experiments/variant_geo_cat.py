# Эксперимент: гео с учётом вида услуги и категории как признаки ранжировщика.
#
# Идея (из доклада Авито про поиск и из разбора ошибок): больше всего Recall теряется на поисках
# по региону и в больших городах. Сейчас гео-приор одинаков для всех услуг, хотя юриста или
# онлайн-репетитора выбирают из любого города, а сантехника - только рядом.
#
# Новые признаки кандидата:
#   geo_vid_logp  - log P(город объявления | город поиска, вид услуги объявления), сглаженная к общему гео;
#   vid_remote    - доля выборов в этом виде услуги, где город объявления не совпал с городом поиска;
#   vid_code, mc_code - вид услуги и микрокатегория объявления как категориальные признаки LightGBM.
#
# Работает поверх сохранённых пулов ноутбука 04 (без пересборки кандидатов).
# Запуск: python experiments/variant_geo_cat.py <папка артефактов>

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.prep import ParamsParser, read_parquet
from src.ranking import cv_ranker

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'artifacts'
LAMBDA = 5.0  # сила сглаживания к общему гео
FLOOR = 1e-6


def train_item_vid(corpus, data_dir):
    # Вид услуги для всех объявлений train: большинства из них нет в корпусе, поэтому парсим их параметры.
    parser = ParamsParser(corpus['params'])
    items = read_parquet(data_dir / 'train.parquet', columns=['item_id', 'item_infm_params_text']).drop_duplicates(
        'item_id'
    )
    return pd.Series(
        items['item_infm_params_text'].fillna('').map(parser.extractors['vid']).values, index=items['item_id']
    )


def add_geo_cat(pool, queries, corpus, fit, fit_vid):
    item = corpus.set_index('item_id')
    vid_of = item['vid']
    # вид услуги выбранного объявления для каждой строки train
    fit = fit.assign(vid=fit['item_id'].map(fit_vid).fillna(''))
    fit = fit[fit['vid'] != '']

    c_slv = fit.groupby(['search_location_id', 'item_location_id', 'vid']).size()
    c_sv = fit.groupby(['search_location_id', 'vid']).size()
    c_sl = fit.groupby(['search_location_id', 'item_location_id']).size()
    c_s = fit.groupby('search_location_id').size()
    remote = (fit['search_location_id'] != fit['item_location_id']).groupby(fit['vid']).mean()

    s = queries['search_location_id'].values[pool['q_row'].values]
    loc = item['item_location_id'].reindex(pool['item_id']).values
    v = vid_of.reindex(pool['item_id']).fillna('').values
    key3 = pd.MultiIndex.from_arrays([s, loc, v])
    n_slv = c_slv.reindex(key3).fillna(0).values
    n_sv = c_sv.reindex(pd.MultiIndex.from_arrays([s, v])).fillna(0).values
    # общее гео P(l | s) по train - база для сглаживания
    p_sl = c_sl.reindex(pd.MultiIndex.from_arrays([s, loc])).fillna(0).values / np.maximum(
        c_s.reindex(s).fillna(0).values, 1
    )
    p = (n_slv + LAMBDA * p_sl) / (n_sv + LAMBDA)
    pool['geo_vid_logp'] = np.log(p + FLOOR).astype(np.float32)
    pool['geo_vid_n'] = np.log1p(n_sv).astype(np.float32)  # сколько данных за оценкой
    pool['vid_remote'] = pd.Series(v).map(remote).fillna(remote.mean()).values.astype(np.float32)
    # тип category: LightGBM сам обработает их как категориальные признаки
    pool['vid_code'] = pd.Categorical(v, categories=sorted(set(vid_of)))
    pool['mc_code'] = pd.Categorical(
        item['item_microcat_id'].reindex(pool['item_id']).values, categories=sorted(set(item['item_microcat_id']))
    )
    return pool


if __name__ == '__main__':
    corpus = pd.read_parquet(
        ART / 'corpus.parquet', columns=['item_id', 'vid', 'item_location_id', 'item_microcat_id', 'params']
    )
    fit_vid = train_item_vid(corpus, ART.parent / 'data')
    print('вид услуги найден у объявлений train:', round((fit_vid != '').mean(), 3), flush=True)
    fit = pd.read_parquet(ART / 'train_fit.parquet', columns=['search_location_id', 'item_location_id', 'item_id'])
    q_val = pd.read_parquet(ART / 'queries_val.parquet')
    pool = pd.read_parquet(ART / 'pool_val.parquet')
    pool = add_geo_cat(pool, q_val, corpus, fit, fit_vid)

    new = ['geo_vid_logp', 'geo_vid_n', 'vid_remote', 'vid_code', 'mc_code']
    variants = {
        'база (как v7)': (pool.drop(columns=new), {}),
        '+ гео по виду услуги': (pool.drop(columns=['vid_code', 'mc_code']), {}),
        '+ гео по виду услуги + категории': (pool, {}),
    }
    for name, (df, params) in variants.items():
        t = time.time()
        rec, it, _ = cv_ranker(df, q_val, rounds=2000, params=params)
        print(f'{name:36s} OOF Recall@50 = {rec:.4f}  (итераций {it}, {time.time() - t:.0f} c)', flush=True)
