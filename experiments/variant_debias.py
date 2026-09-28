# Вариант без признаков «раскрученности» объявления (поверх пулов из ноутбука 04).
#
# Почему: все позитивы валидации - объявления, которые выбирали в train. Такие объявления
# заметно «раскрученнее» остальных: больше отзывов, чаще есть рейтинг, длиннее описание
# (см. 06_error_analysis). Ранжировщик учится правилу «выбирают раскрученных», а позитивы
# бенчмарка в train почти не встречаются (вариант с популярностью v5 на бенчмарке не помог).
# Поэтому признаки, по которым видно «было ли объявление в train», убираем.
#
# Запуск из корня проекта: python experiments/variant_debias.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.pipeline import ART
from src.ranking import check_answer, cv_ranker, fit_ranker, make_answer

# признаки, которые у объявлений из train систематически выше, чем у остальных
BIASED = ['reviews_log', 'rating', 'desc_len', 'phone_hidden', 'msg_forbidden']

q_val = pd.read_parquet(ART / 'queries_val.parquet')
q_bench = pd.read_parquet(ART / 'queries_bench.parquet')
pool_val = pd.read_parquet(ART / 'pool_val.parquet').drop(columns=BIASED)
pool_bench = pd.read_parquet(ART / 'pool_bench.parquet').drop(columns=BIASED)

rec, best_iter, oof = cv_ranker(pool_val, q_val, rounds=2000)
oof[['q_row', 'item_id', 'pred', 'label']].to_parquet(ART / 'stage1_oof_val_v8.parquet')
print(f'Без признаков раскрученности: OOF Recall@50 = {rec:.4f}, итераций {best_iter}', flush=True)

# финальный ответ: среднее 5 моделей с разными seed
score = np.zeros(len(pool_bench))
for seed in range(5):
    model, cols = fit_ranker(pool_val, best_iter, seed=seed)
    s = model.predict(pool_bench[cols])
    score += (s - s.mean()) / (s.std() + 1e-9)
answer = make_answer(pool_bench, q_bench, score)
corpus = pd.read_parquet(ART / 'corpus.parquet', columns=['item_id', 'in_bench'])
check_answer(answer, q_bench, corpus.loc[corpus['in_bench'] == 1, 'item_id'])
answer.to_csv(ART.parent / 'answer_v8_debias.csv', index=False)
print('saved answer_v8_debias.csv', flush=True)
