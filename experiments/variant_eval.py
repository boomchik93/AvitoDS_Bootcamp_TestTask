# Оценка варианта решения на валидации и сборка ответа.
#
# Собирает пулы кандидатов с заданным набором нейросетей, по желанию добавляет признаки
# гео по виду услуги (variant_geo_cat.py), считает OOF Recall@50 (5-fold CV по запросам)
# и собирает ответ на бенчмарк (среднее 5 LightGBM с разными seed).
#
# Запуск: AVITO_ART=artifacts_v2 python experiments/variant_eval.py --name v9 --dense e5b,e5l,bgem3,e5l_hn --geo_cat

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.variant_geo_cat import add_geo_cat, train_item_vid
from src.pipeline import ART, build_pools, build_sources, load_artifacts, load_dense
from src.ranking import check_answer, cv_ranker, fit_ranker, make_answer, pool_recall

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--dense', default='e5b,e5l,bgem3')
ap.add_argument('--feature_only', default='e5b_raw')
ap.add_argument('--geo_cat', action='store_true')
args = ap.parse_args()

dense_names = [n for n in args.dense.split(',') + args.feature_only.split(',') if n]
corpus, q_val, q_bench, fit = load_artifacts()
sources = build_sources(corpus, fit)
dense = load_dense(dense_names)
t = time.time()
pool_val, pool_bench = build_pools(corpus, q_val, q_bench, sources, dense, feature_only=args.feature_only.split(','))
print(f'пулы: {time.time() - t:.0f} c; потолок пула {pool_recall(pool_val, q_val):.4f}', flush=True)

if args.geo_cat:
    fit_vid = train_item_vid(corpus, ART.parent / 'data')
    geo_fit = fit[['search_location_id', 'item_location_id', 'item_id']]
    pool_val = add_geo_cat(pool_val, q_val, corpus, geo_fit, fit_vid)
    pool_bench = add_geo_cat(pool_bench, q_bench, corpus, geo_fit, fit_vid)

rec, best_iter, oof = cv_ranker(pool_val, q_val, rounds=2000)
print(f'{args.name}: OOF Recall@50 = {rec:.4f} (итераций {best_iter})', flush=True)
oof[['q_row', 'item_id', 'pred', 'label']].to_parquet(ART / f'oof_{args.name}.parquet')

score = np.zeros(len(pool_bench))
for seed in range(5):
    model, cols = fit_ranker(pool_val, best_iter, seed=seed)
    s = model.predict(pool_bench[cols])
    score += (s - s.mean()) / (s.std() + 1e-9)
answer = make_answer(pool_bench, q_bench, score)
check_answer(answer, q_bench, corpus.loc[corpus['in_bench'] == 1, 'item_id'])
answer.to_csv(ART.parent / f'answer_{args.name}.csv', index=False)
print(f'saved answer_{args.name}.csv', flush=True)
