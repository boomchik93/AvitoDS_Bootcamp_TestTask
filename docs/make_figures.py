# Графики для README.
#
# Цифры взяты из выполненных ноутбуков (в комментариях указано, из какого). Карта объявлений
# извлекается из 01_eda.ipynb. Запуск из корня проекта: python docs/make_figures.py

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat

OUT = Path(__file__).resolve().parent / 'img'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'

plt.rcParams.update(
    {
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': INK2,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'text.color': INK,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    }
)


def hbar(ax, labels, values, colors, fmt='{:.1%}', xmax=None):
    # Горизонтальные столбцы с подписями значений справа от столбца.
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, xmax or max(values) * 1.18)
    ax.xaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v + (xmax or max(values)) * 0.01, yi, fmt.format(v), va='center', color=INK, fontsize=11)


def save(fig, name):
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print('saved', name)


# 1. Откуда выбирают объявление (01_eda, раздел 4.2)
fig, ax = plt.subplots(figsize=(7.5, 2.4))
hbar(
    ax,
    ['тот же город, что и поиск', 'соседний город (медиана 34 км)', 'поиск по региону / стране'],
    [0.831, 0.061, 0.108],
    BLUE,
)
fig.suptitle('Где находится выбранное объявление', x=0.01, ha='left', fontsize=13, fontweight='bold')
save(fig, 'location.png')

# 2. Совпадение слов запроса и объявления (01_eda, раздел 4.3; 100 тыс. пар train)
fig, ax = plt.subplots(figsize=(7.5, 2.4))
hbar(
    ax,
    ['заголовок, точные слова', 'заголовок, основы слов', 'заголовок + параметры, основы слов'],
    [0.279, 0.159, 0.078],
    BLUE,
)
fig.suptitle(
    'Доля пар «запрос → выбранное объявление» без единого общего слова',
    x=0.01,
    ha='left',
    fontsize=13,
    fontweight='bold',
)
save(fig, 'word_overlap.png')

# 3. Насколько запросы бенчмарка встречаются в train (01_eda, раздел 5)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
hbar(
    ax,
    ['не встречался', '1 раз', '2–5 раз', '6–20 раз', 'больше 20 раз'],
    [0.625, 0.096, 0.147, 0.090, 0.042],
    [ORANGE] + [BLUE] * 4,
)
fig.suptitle(
    'Сколько раз текст запроса бенчмарка встречался в train', x=0.01, ha='left', fontsize=13, fontweight='bold'
)
save(fig, 'bench_query_freq.png')

# 4. Вклад каналов кандидатогенерации (04_retrieve_rank, v4, 6 000 отложенных поисков)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
labels = [
    'BM25 по полям + гео',
    'символьные n-граммы + гео',
    'микрокатегория + гео',
    'e5-base + гео',
    'bge-m3 + гео',
    'e5-large + гео',
    'пул: объединение каналов (потолок)',
    'итог: LightGBM, топ-50',
]
values = [0.5688, 0.5255, 0.6714, 0.9113, 0.9112, 0.9180, 0.9899, 0.9585]
hbar(ax, labels, values, [BLUE] * 6 + [ORANGE] * 2, fmt='{:.3f}', xmax=1.12)
fig.suptitle('Recall@50 каждого канала по отдельности и всей схемы', x=0.01, ha='left', fontsize=13, fontweight='bold')
fig.text(
    0.01,
    0.01,
    'синий — отдельный канал, оранжевый — вся схема; 6 000 отложенных поисков (валидация v1)',
    color=INK2,
    fontsize=9,
)
save(fig, 'channels.png')

# 5. Плотность конкурентов вокруг запроса (06_error_analysis; города с 1–5 тыс. объявлений)
fig, ax = plt.subplots(figsize=(7.5, 2.4))
hbar(
    ax,
    ['запросы бенчмарка', 'валидация: объявление уже в корпусе', 'валидация: объявление добавлено из train'],
    [0.339, 0.326, 0.268],
    [ORANGE, BLUE, BLUE],
    fmt='{:.3f}',
)
fig.suptitle('Похожесть 50-го ближайшего конкурента в том же городе', x=0.01, ha='left', fontsize=13, fontweight='bold')
save(fig, 'competitor_density.png')

# 6. Отправки на бенчмарк (точечный график: разница в тысячных, столбцы от нуля её бы скрыли)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
names = ['v4 — итог', 'v5 — + популярность', 'v7 — валидация v2', 'v6 — валидация v2, 1 нейросеть']
scores = [0.926158, 0.925129, 0.924588, 0.917799]
y = list(range(len(names)))[::-1]
ax.hlines(y, 0.915, scores, color=GRID, linewidth=2)
ax.scatter(scores, y, s=80, color=[ORANGE, BLUE, BLUE, BLUE], zorder=3, edgecolor=SURFACE, linewidth=2)
for yi, s in zip(y, scores):
    ax.text(s + 0.0004, yi, f'{s:.4f}', va='center', color=INK)
ax.set_yticks(y, names)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0.915, 0.93)
ax.set_xlabel('Recall@50 на бенчмарке (ось начинается с 0.915)')
fig.suptitle('Результаты отправок', x=0.01, ha='left', fontsize=13, fontweight='bold')
save(fig, 'submissions.png')

# 7. Карта объявлений корпуса (из 01_eda.ipynb)
nb = nbformat.read(ROOT / 'notebooks' / '01_eda.ipynb', 4)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Объявления корпуса на карте' in cell.source:
        for o in cell.outputs:
            if 'image/png' in o.get('data', {}):
                (OUT / 'map.png').write_bytes(base64.b64decode(o['data']['image/png']))
                print('saved map.png')
