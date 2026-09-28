# Подготовка данных: чистка текста, разбор параметров, таблицы и валидация.

import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import snowballstemmer

QUERY_COLS = [
    'search_query',
    'search_location_id',
    'search_is_delivery_search',
    'search_infm_params_text',
    'search_category',
]
ITEM_COLS = [
    'item_id',
    'item_title_raw',
    'item_description_raw',
    'item_infm_params_text',
    'item_category_id',
    'item_microcat_id',
    'item_price',
    'item_rating',
    'item_rating_reviews_count',
    'item_location_id',
    'item_latitude',
    'item_longitude',
    'item_is_phone_hidden',
    'item_is_message_forbidden',
]


def read_parquet(path, columns=None):
    # Читает parquet и переводит цену и координаты во float.
    df = pq.read_table(path, columns=columns).to_pandas()
    for col in ['item_price', 'item_latitude', 'item_longitude']:
        if col in df:
            df[col] = df[col].astype('float64')
    return df


_TOKEN_RE = re.compile(r'[a-zа-я0-9]+')
_stemmer = snowballstemmer.stemmer('russian')
_stem_cache = {}

# слова, которые есть почти в каждом тексте, в поиске они только мешают
STOPWORDS = set(
    'в во на и с со по для от до за к ко у о об из не или а но же ли бы то это как под над '
    'при без через все мы вы я он она они их его ее'.split()
)


def norm_text(s):
    # Приводит текст к нижнему регистру и оставляет только буквы и цифры.
    if not isinstance(s, str):
        return ''
    s = s.lower().replace('ё', 'е')
    return ' '.join(_TOKEN_RE.findall(s))


def stem(word):
    # Возвращает основу слова, результат запоминается в кэше.
    res = _stem_cache.get(word)
    if res is None:
        res = word if word.isdigit() else _stemmer.stemWord(word)
        _stem_cache[word] = res
    return res


def stem_tokens(s):
    # Разбивает текст на слова, убирает стоп-слова и берёт основы.
    if not isinstance(s, str) or not s:
        return []
    s = s.lower().replace('ё', 'е')
    return [stem(w) for w in _TOKEN_RE.findall(s) if w not in STOPWORDS]


# Параметры объявления записаны одной строкой «ключ значение ключ значение».
# Значение ищем от ключа до следующего ключа из этого списка.
_STOP_KEYS = [
    'Место оказания услуг',
    'Вид услуги',
    'Тип услуги',
    'Тип стоимости',
    'Опыт работы',
    'Работаете',
    'Работа по договору',
    'Марка',
    'Поездка',
    'Онлайн-запись',
    'Рейтинг',
    'Кто оказывает',
    'Срочная',
    'Чем вы',
    'Как вы',
    'Где вы',
    'Куда выезжаете',
    'Транспорт',
    'Гарантия',
    'Время',
    'Дни',
    'Рабочие дни',
    'Бригада',
    'Признак',
    'Минимальн',
    'Берёте',
    'Начальная цена',
    'Стоимость',
    'Продолжительность',
    'Название услуги',
    'Услуга',
    'Предоплата',
    'График работы',
    'Производители',
    'Дополнительно',
    'Инструменты',
    'автосервиса',
    'Предмет или специальность',
    'Онлайн-показ',
    'Поиск по',
]
_NEXT_KEY = '(?=' + '|'.join(' ' + re.escape(k) for k in _STOP_KEYS) + '|$)'


def build_value_vocab(texts, key, min_count=5):
    # Собирает частые значения ключа, длинные идут первыми.
    rx = re.compile(re.escape(key) + r' (.+?)' + _NEXT_KEY)
    cnt = Counter()
    for text in texts:
        if isinstance(text, str):
            for value in rx.findall(text):
                if len(value.split()) <= 6:
                    cnt[value.strip()] += 1
    # если значение пустое, регулярка захватывает следующий ключ, такое выкидываем
    values = [v for v, c in cnt.items() if c >= min_count and v and not v.startswith(tuple(_STOP_KEYS))]
    return sorted(values, key=len, reverse=True)


def make_value_extractor(key, vocab):
    # Делает функцию, которая достаёт из текста значение ключа из словаря.
    rx = re.compile(re.escape(key) + ' (' + '|'.join(map(re.escape, vocab)) + r')(?= |$)')

    def extract(text):
        if not isinstance(text, str):
            return ''
        m = rx.search(text)
        return m.group(1) if m else ''

    return extract


_SERVICE_RE = re.compile(r'(?:Название услуги|Услуга) (.+?)' + _NEXT_KEY)


def extract_services(text, max_n=15):
    # Достаёт названия услуг из прайс-листа в параметрах.
    if not isinstance(text, str) or 'слуг' not in text:
        return []
    out = []
    for s in _SERVICE_RE.findall(text):
        s = s.replace('Своя услуга', '').strip()
        if s and s not in out and len(s) <= 80 and not s.startswith(tuple(_STOP_KEYS)):
            out.append(s)
        if len(out) >= max_n:
            break
    return out


class ParamsParser:
    # Достаёт вид и тип услуги из параметров объявления или фильтров запроса.

    KEYS = {'vid': 'Вид услуги', 'type': 'Тип услуги', 'auto': 'Тип услуги автосервиса'}

    def __init__(self, texts):
        texts = list(texts)
        self.vocab = {}
        self.extractors = {}
        for name, key in self.KEYS.items():
            self.vocab[name] = build_value_vocab(texts, key)
            self.extractors[name] = make_value_extractor(key, self.vocab[name])

    def parse(self, text):
        auto = self.extractors['auto'](text)
        # «Тип услуги автосервиса» тоже начинается с «Тип услуги», поэтому сначала убираем его
        rest = text.replace('Тип услуги автосервиса ' + auto, ' ') if auto else text
        return {
            'vid': self.extractors['vid'](text),
            'type': self.extractors['type'](rest),
            'auto': auto,
        }


def build_items(items, parser, desc_chars=1500):
    # Готовит таблицу объявлений: текстовые поля для поиска и числовые признаки.
    #
    # title - заголовок, struct - вид, тип и названия услуг, params - все параметры,
    # desc - начало описания, dense_text - короткий текст для нейросети.
    df = items.copy()
    parsed = df['item_infm_params_text'].fillna('').map(parser.parse)
    df['vid'] = parsed.map(lambda d: d['vid'])
    df['type'] = parsed.map(lambda d: d['type'])
    df['auto_type'] = parsed.map(lambda d: d['auto'])
    services = df['item_infm_params_text'].map(extract_services)
    df['services'] = services.map(' ; '.join)

    df['title'] = df['item_title_raw'].fillna('')
    df['struct'] = (df['vid'] + ' ; ' + df['type'] + ' ; ' + df['auto_type'] + ' ; ' + df['services']).str.strip(' ;')
    df['params'] = df['item_infm_params_text'].fillna('')
    df['desc'] = df['item_description_raw'].fillna('').str.slice(0, desc_chars)
    # у объявлений из train описание уже обрезано, поэтому полную длину берём
    # из desc_full_len, иначе по длине было бы видно, откуда объявление
    if 'desc_full_len' in df:
        df['desc_len'] = df['desc_full_len']
    else:
        df['desc_len'] = df['item_description_raw'].fillna('').str.len()
    df['dense_text'] = (
        df['title']
        + '. '
        + df['vid']
        + ', '
        + df['type']
        + ' '
        + df['auto_type']
        + '. '
        + services.map(lambda s: ', '.join(s[:8]))
        + '. '
        + df['item_description_raw'].fillna('').str.slice(0, 400)
    )

    # цена -1 значит «не указана», 0 и 1 тоже не настоящие цены
    price = df['item_price'].where(df['item_price'] > 1)
    df['price_log'] = np.log1p(price.clip(upper=1e7))
    df['rating'] = df['item_rating']
    df['reviews_log'] = np.log1p(df['item_rating_reviews_count'].fillna(0))
    # координаты вне России считаем битыми
    lon = df['item_longitude']
    bad_geo = ~df['item_latitude'].between(41, 82) | ~(lon.between(19, 180) | lon.between(-180, -168))
    df.loc[bad_geo, ['item_latitude', 'item_longitude']] = np.nan
    df['not_services'] = (df['item_category_id'] != 114).astype('int8')
    df['phone_hidden'] = df['item_is_phone_hidden'].astype('int8')
    df['msg_forbidden'] = df['item_is_message_forbidden'].astype('int8')

    keep = [
        'item_id',
        'title',
        'struct',
        'params',
        'desc',
        'dense_text',
        'vid',
        'type',
        'auto_type',
        'item_category_id',
        'item_microcat_id',
        'item_location_id',
        'item_latitude',
        'item_longitude',
        'price_log',
        'rating',
        'reviews_log',
        'desc_len',
        'not_services',
        'phone_hidden',
        'msg_forbidden',
    ]
    return df[keep].reset_index(drop=True)


def build_queries(q, parser):
    # Готовит таблицу запросов: чистый текст и разобранные фильтры.
    df = q.copy()
    df['q_norm'] = df['search_query'].map(norm_text)
    params = df['search_infm_params_text'].fillna('')
    parsed = params.map(parser.parse)
    df['f_vid'] = parsed.map(lambda d: d['vid'])
    df['f_type'] = parsed.map(lambda d: d['type'])
    df['f_auto'] = parsed.map(lambda d: d['auto'])
    df['f_rating'] = params.str.contains('Рейтинг пользователя').astype('int8')
    # для нейросети дописываем к запросу значения фильтров
    filt = (df['f_vid'] + ' ' + df['f_type'] + ' ' + df['f_auto']).str.strip()
    df['dense_query'] = np.where(filt != '', df['q_norm'] + ' | ' + filt, df['q_norm'])
    return df


GROUP_KEY = [
    'q_norm',
    'search_location_id',
    'search_infm_params_text',
    'search_category',
    'search_is_delivery_search',
]

# как часто тексты запросов бенчмарка встречаются в train (посчитано в EDA)
BENCH_FREQ_BUCKETS = {'unseen': 0.625, '1': 0.096, '2-5': 0.147, '6-20': 0.090, '>20': 0.042}


def _bucket(n):
    if n <= 0:
        return 'unseen'
    if n == 1:
        return '1'
    if n <= 5:
        return '2-5'
    if n <= 20:
        return '6-20'
    return '>20'


def make_holdout(train, n_groups=6000, seed=42, only_items=None):
    # Откладывает часть поисков из train так, чтобы они были похожи на бенчмарк.
    #
    # Поиск - это текст, локация и фильтры. Для редких текстов из train убираются
    # все строки с этим текстом, для остальных откладывается один поиск.
    # Если задан only_items, берутся только поиски, у которых все выбранные объявления
    # лежат в этом множестве (корпус бенчмарка). Тогда корпус валидации равен корпусу
    # бенчмарка, и условия поиска те же: корпус собран вокруг запросов бенчмарка,
    # у них больше похожих конкурентов, чем у случайного запроса train.
    # Возвращает таблицу отложенных поисков и маску строк train для обучения.
    t = train[train['search_is_delivery_search'] == 0]
    groups = (
        t.groupby(GROUP_KEY, sort=False)
        .agg(
            search_query=('search_query', 'first'),
            item_ids=('item_id', lambda s: list(dict.fromkeys(s))),
            rows=('item_id', 'size'),
        )
        .reset_index()
    )
    text_rows = train.groupby('q_norm').size()
    groups['text_rows'] = groups['q_norm'].map(text_rows)
    groups['rest_rows'] = groups['text_rows'] - groups['rows']

    if only_items is not None:
        groups = groups[groups['item_ids'].map(lambda ids: all(i in only_items for i in ids))]

    # в бенчмарке каждый текст встречается один раз
    groups = groups.sample(frac=1.0, random_state=seed).drop_duplicates('q_norm')

    picked = []
    n_unseen = int(n_groups * BENCH_FREQ_BUCKETS['unseen'])
    rare = groups[groups['text_rows'] <= 3]
    picked.append(rare.sample(min(n_unseen, len(rare)), random_state=seed).assign(bucket='unseen'))
    rest = groups[~groups['q_norm'].isin(picked[0]['q_norm'])].copy()
    rest['bucket'] = rest['rest_rows'].map(_bucket)
    for bucket, share in BENCH_FREQ_BUCKETS.items():
        if bucket == 'unseen':
            continue
        pool = rest[rest['bucket'] == bucket]
        picked.append(pool.sample(min(len(pool), int(n_groups * share)), random_state=seed))
    hold = pd.concat(picked, ignore_index=True)
    hold['query_id'] = [f'val{i:06d}' for i in range(len(hold))]

    unseen_texts = set(hold.loc[hold['bucket'] == 'unseen', 'q_norm'])
    key = pd.MultiIndex.from_frame(train[GROUP_KEY])
    hold_key = pd.MultiIndex.from_frame(hold[GROUP_KEY])
    fit_mask = ~train['q_norm'].isin(unseen_texts).values & ~key.isin(hold_key)
    return hold, fit_mask


def recall_at_k(pred, true, k=50):
    # Считает Recall@k, как в условии задачи.
    vals = []
    for p, t in zip(pred, true):
        t = set(t)
        vals.append(len(set(p[:k]) & t) / len(t) if t else 0.0)
    return float(np.mean(vals))
