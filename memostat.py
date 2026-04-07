#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
МЕМОСТАТ v6.0 — RSS + Демо версия
══════════════════════════════════════════════════════════════════════════════
Система анализа жизненного цикла интернет-мемов

ИСТОЧНИКИ ДАННЫХ:
📡 RSS Google Trends — реальные тренды дня (бесплатно, без ключей)
🧪 Демо-генератор — реалистичные временные ряды

ОСОБЕННОСТИ:
✅ Работает всегда — не требует API ключей
✅ Реальные тренды через официальный RSS Google
✅ Реалистичные демо-данные для анализа
✅ Кэширование всех запросов
✅ Под Windows + Яндекс.Браузер

Автор: Проектная работа
Дата: 2024
"""

import os
import re
import time
import random
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import matplotlib

matplotlib.use('TkAgg')  # для Windows
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
import feedparser

# Игнорируем предупреждения
warnings.filterwarnings('ignore')


# ============================================================================
# БЛОК 1: КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Конфигурация проекта"""

    # ========= ПАПКИ =========
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'memostat_data')
    REPORTS_DIR = os.path.join(BASE_DIR, 'memostat_reports')
    PLOTS_DIR = os.path.join(BASE_DIR, 'memostat_plots')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')

    # ========= ПАРАМЕТРЫ АНАЛИЗА =========
    MAX_MEMES = 100
    PEAK_PROMINENCE = 15
    PEAK_DISTANCE = 30
    SMOOTH_WINDOW = 7
    DEATH_THRESHOLD = 5
    DEATH_DURATION = 30

    # ========= СТАДИИ =========
    STAGES = {
        'embryo': '🟣 ЭМБРИОН',
        'growth': '🔵 РОСТ',
        'takeoff': '🟢 ВЗЛЁТ',
        'peak': '🟡 ПИК',
        'decay': '🟠 ЗАТУХАНИЕ',
        'stagnation': '🔴 СТАГНАЦИЯ',
        'death': '⚫ СМЕРТЬ'
    }

    # ========= ТИПЫ ТРАЕКТОРИЙ =========
    TRAJECTORIES = {
        'smooth': '📉 Гладкий спад',
        'oscillatory': '📊 Колебательный спад',
        'plateau': '📈 Плато',
        'sustained': '🚀 Устойчивый рост'
    }

    # ========= НАЧАЛЬНЫЙ ПУЛ МЕМОВ =========
    INITIAL_MEMES = [
        'ждун', 'краш', 'рофл', 'хайп', 'кринж', 'токс', 'сигма', 'скуф',
        'альтушка', 'вайб', 'пруф', 'агриться', 'зашквар',
        'ждун персонаж', 'троллфейс', 'удивленный киану', 'лягушка пепе',
        'ну погоди', 'шрек', 'ленин гриб', 'превед медвед',
        'павлова мем', 'ауф', 'ноу вэй', 'чилл', 'душнила'
    ]

    @classmethod
    def init_dirs(cls):
        """Создаёт все необходимые папки"""
        for d in [cls.DATA_DIR, cls.REPORTS_DIR, cls.PLOTS_DIR, cls.CACHE_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"📁 Папки созданы")


# Инициализируем папки
Config.init_dirs()


# ============================================================================
# БЛОК 2: СЛОВАРИ ДЛЯ ФИЛЬТРАЦИИ МЕМОВ
# ============================================================================

# Сленговые слова (маркеры мемов)
SLANG_WORDS = {
    'кринж', 'хайп', 'рофл', 'краш', 'токс', 'сигма', 'скуф', 'альтушка',
    'вайб', 'пруф', 'агриться', 'зашквар', 'душнила', 'чилл', 'ауф', 'ноу вэй',
    'павлова', 'ждун', 'тролл', 'фейс', 'пепе', 'шрек', 'ленин гриб', 'превед медвед'
}

# Стоп-слова (что точно НЕ мемы)
STOP_WORDS = {
    'новости', 'погода', 'курс', 'доллар', 'евро', 'нефть', 'война', 'выборы',
    'президент', 'путин', 'зеленский', 'байден', 'трамп', 'ковид', 'коронавирус',
    'санкции', 'рубль', 'биткоин', 'крипта', 'футбол', 'хоккей', 'теннис',
    'украина', 'россия', 'сша', 'китай', 'вчера', 'сегодня', 'завтра'
}


def is_meme_candidate(phrase: str) -> Tuple[bool, str]:
    """
    Проверяет, является ли фраза кандидатом в мемы.
    Возвращает (True/False, причина)
    """
    phrase_lower = phrase.lower().strip()

    # 1. Проверка длины (мемы обычно короткие)
    if len(phrase_lower) < 3:
        return False, "слишком короткое"
    if len(phrase_lower) > 30:
        return False, "слишком длинное"

    # 2. Проверка на стоп-слова (новости, события)
    for stop in STOP_WORDS:
        if stop in phrase_lower:
            return False, f"стоп-слово: {stop}"

    # 3. Проверка на сленг
    if phrase_lower in SLANG_WORDS:
        return True, "сленговое слово"

    # 4. Проверка на составные фразы (с пробелами)
    if ' ' in phrase_lower:
        words = phrase_lower.split()
        # Фразы из 2-4 слов могут быть мемами
        if 2 <= len(words) <= 4:
            # Проверяем каждое слово
            meme_words = 0
            for w in words:
                if w in SLANG_WORDS:
                    meme_words += 1
            if meme_words >= len(words) // 2:
                return True, "составная фраза"

    # 5. Наличие кавычек (часто указывает на цитату-мем)
    if '"' in phrase or '«' in phrase or '»' in phrase:
        return True, "в кавычках"

    # 6. Наличие эмодзи (мемы часто с ними)
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]')
    if emoji_pattern.search(phrase):
        return True, "содержит эмодзи"

    return False, "не соответствует критериям"


# ============================================================================
# БЛОК 3: ПОВЕДЕНЧЕСКИЙ АНАЛИЗАТОР
# ============================================================================

class BehavioralAnalyzer:
    """
    Анализирует поведенческие паттерны запросов для отличия мемов от новостей
    """

    def analyze_behavior(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализирует поведенческие характеристики запроса
        Возвращает словарь с метриками и вердикт (мем/не мем)
        """
        if df.empty or len(df) < 30:
            return {'is_meme': False, 'reason': 'недостаточно данных', 'metrics': {}, 'score': 0}

        values = df['value'].values

        # ========= 1. СТАТИСТИЧЕСКИЕ МЕТРИКИ =========

        # Коэффициент вариации (изменчивость)
        cv = np.std(values) / (np.mean(values) + 0.01)

        # Эксцесс (островершинность) — у мемов резкие пики
        kurt = kurtosis(values)

        # Асимметрия (скошенность) — у мемов быстрый рост и медленный спад
        skewness = skew(values)

        # Максимальное значение
        max_val = np.max(values)

        # Дней до пика (от начала)
        peak_idx = np.argmax(values)
        days_to_peak = peak_idx

        # Дней после пика (до конца)
        days_after_peak = len(values) - peak_idx - 1

        # ========= 2. ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ =========

        # Скорость роста до пика (ежедневный прирост в %)
        if peak_idx > 0:
            growth_rate = (max_val - values[0]) / (peak_idx + 1)
            growth_rate_pct = (growth_rate / (max_val + 0.01)) * 100
        else:
            growth_rate_pct = 0

        # Скорость спада после пика
        if days_after_peak > 0:
            decay_rate = (max_val - values[-1]) / (days_after_peak + 1)
            decay_rate_pct = (decay_rate / (max_val + 0.01)) * 100
        else:
            decay_rate_pct = 0

        # Время жизни (ширина на половине высоты)
        half_max = max_val / 2
        above_half = np.where(values >= half_max)[0]
        if len(above_half) > 0:
            fwhm = above_half[-1] - above_half[0]  # Full Width at Half Maximum
        else:
            fwhm = 0

        # ========= 3. КОЛЕБАТЕЛЬНОСТЬ =========

        # Количество локальных пиков (кроме главного)
        peaks, _ = find_peaks(values, prominence=max_val * 0.1)
        n_peaks = len(peaks) - 1 if len(peaks) > 0 else 0

        # Средняя разница между соседними днями
        daily_diff = np.abs(np.diff(values))
        avg_volatility = np.mean(daily_diff)

        # ========= 4. ОЦЕНКА ПОВЕДЕНЧЕСКИХ ПРИЗНАКОВ =========

        score = 0
        reasons = []

        # Признак 1: Высокая изменчивость (мемы резко взлетают)
        if cv > 1.5:
            score += 2
            reasons.append(f"высокая изменчивость (CV={cv:.2f})")
        elif cv > 1.0:
            score += 1
            reasons.append(f"средняя изменчивость (CV={cv:.2f})")

        # Признак 2: Островершинность (мемы имеют резкий пик)
        if kurt > 3:
            score += 2
            reasons.append(f"острый пик (эксцесс={kurt:.2f})")
        elif kurt > 1:
            score += 1
            reasons.append(f"умеренный пик (эксцесс={kurt:.2f})")

        # Признак 3: Положительная асимметрия (быстрый рост, медленный спад)
        if skewness > 0.5:
            score += 2
            reasons.append(f"положительная асимметрия (skew={skewness:.2f})")
        elif skewness > 0:
            score += 1
            reasons.append(f"слабая асимметрия (skew={skewness:.2f})")

        # Признак 4: Быстрый рост
        if growth_rate_pct > 5:
            score += 2
            reasons.append(f"быстрый рост ({growth_rate_pct:.1f}%/день)")
        elif growth_rate_pct > 2:
            score += 1
            reasons.append(f"умеренный рост ({growth_rate_pct:.1f}%/день)")

        # Признак 5: Быстрый спад (мемы быстро забываются)
        if decay_rate_pct > 3:
            score += 2
            reasons.append(f"быстрый спад ({decay_rate_pct:.1f}%/день)")
        elif decay_rate_pct > 1:
            score += 1
            reasons.append(f"умеренный спад ({decay_rate_pct:.1f}%/день)")

        # Признак 6: Короткая ширина пика (мемы не длятся долго)
        if fwhm < 30:
            score += 2
            reasons.append(f"короткий пик ({fwhm} дней)")
        elif fwhm < 60:
            score += 1
            reasons.append(f"средний пик ({fwhm} дней)")

        # Признак 7: Количество дополнительных пиков (мемы иногда дают эхо)
        if 1 <= n_peaks <= 3:
            score += 1
            reasons.append(f"есть вторичные пики ({n_peaks} шт.)")

        # Признак 8: Дневная волатильность (мемы непредсказуемы)
        if avg_volatility > 10:
            score += 1
            reasons.append(f"высокая волатильность ({avg_volatility:.1f})")

        # ========= 5. ОТСЕИВАНИЕ НОВОСТЕЙ =========

        # Новости имеют длинный плавный пик, часто симметричный
        is_news = False
        news_reason = None

        # Признак новости: ширина пика > 60 дней и низкий эксцесс
        if fwhm > 60 and kurt < 1:
            is_news = True
            news_reason = f"длинный плавный пик (ширина={fwhm}, эксцесс={kurt:.2f})"

        # Признак новости: симметричный рост и спад
        if abs(growth_rate_pct - decay_rate_pct) < 1 and growth_rate_pct < 2:
            is_news = True
            news_reason = f"симметричный рост/спад (разница={abs(growth_rate_pct - decay_rate_pct):.1f}%)"

        # ========= 6. ИТОГОВЫЙ ВЕРДИКТ =========

        # Порог для мема: нужно набрать минимум 4 балла
        is_meme = (score >= 4) and not is_news and max_val > 15

        metrics = {
            'cv': round(cv, 2),
            'kurtosis': round(kurt, 2),
            'skewness': round(skewness, 2),
            'growth_rate_pct': round(growth_rate_pct, 1),
            'decay_rate_pct': round(decay_rate_pct, 1),
            'fwhm': fwhm,
            'n_peaks': n_peaks,
            'volatility': round(avg_volatility, 1),
            'score': score
        }

        reason = ' | '.join(reasons) if reasons else 'нет явных признаков'
        if is_news:
            reason = f"ОТСЕЯНО (новость): {news_reason} | {reason}"
        elif is_meme:
            reason = f"МЕМ (баллов: {score}): {reason}"
        else:
            reason = f"НЕ МЕМ (баллов: {score}): {reason}"

        return {
            'is_meme': is_meme,
            'is_news': is_news,
            'reason': reason,
            'metrics': metrics,
            'score': score
        }

    def print_behavior_report(self, query: str, analysis: Dict):
        """Выводит отчет о поведенческом анализе"""
        print(f"\n      📊 Поведенческий анализ для '{query}':")
        print(f"         Статус: {analysis['reason']}")

        if analysis['metrics']:
            m = analysis['metrics']
            print(f"         Метрики:")
            print(f"            • Изменчивость: {m['cv']} | Острота пика: {m['kurtosis']}")
            print(f"            • Асимметрия: {m['skewness']} | Балл: {m['score']}/10")
            print(f"            • Рост: {m['growth_rate_pct']}%/день | Спад: {m['decay_rate_pct']}%/день")
            print(f"            • Ширина пика: {m['fwhm']} дней | Вторичные пики: {m['n_peaks']}")


# ============================================================================
# БЛОК 4: ПАРСЕР ЧЕРЕЗ RSS (РЕАЛЬНЫЕ ТРЕНДЫ + ДЕМО-ДАННЫЕ)
# ============================================================================

class GoogleTrendsParser:
    """
    Парсер Google Trends через RSS (для трендов) + демо-данные (для рядов)
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        print("\n🔧 ИНИЦИАЛИЗАЦИЯ ПАРСЕРА")
        print("-" * 50)
        print("   📡 Режим: RSS (тренды) + Демо (временные ряды)")
        print("   ✅ Работает всегда, не требует API")
        print("-" * 50)

    def get_interest_over_time(self, query: str, timeframe: str = 'today 12-m', geo: str = 'RU') -> pd.DataFrame:
        """
        Возвращает данные для анализа (сейчас генерирует демо)
        В будущем можно заменить на реальный парсинг
        """
        print(f"\n   🔍 Запрос: '{query}' ({timeframe}, {geo})")

        # Проверяем кэш
        cache_file = os.path.join(self.cache_dir, f"{query}_{timeframe}_{geo}.pkl")
        if os.path.exists(cache_file):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - mod_time).days < 7:
                print(f"     📦 Загружено из кэша (от {mod_time.strftime('%d.%m.%Y')})")
                return pd.read_pickle(cache_file)

        # Генерируем демо-данные
        df = self._generate_demo_data(query, timeframe)

        # Сохраняем в кэш
        if df is not None and not df.empty:
            df.to_pickle(cache_file)
            print(f"     💾 Сохранено в кэш")

        return df

    def _generate_demo_data(self, query: str, timeframe: str) -> pd.DataFrame:
        """
        Генерирует реалистичные демо-данные с разными паттернами
        """
        print(f"     🧪 Генерация демо-данных")

        # Определяем длину периода
        if '12-m' in timeframe:
            days = 365
        elif '3-m' in timeframe:
            days = 90
        elif '1-m' in timeframe:
            days = 30
        else:
            days = 365

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Генерируем детерминированный seed на основе запроса
        seed = hash(query) % 1000
        np.random.seed(seed)
        random.seed(seed)

        # Выбираем тип кривой (4 разных паттерна)
        curve_type = seed % 4

        if curve_type == 0:  # Классический пик (ждун)
            peak_day = random.randint(int(days * 0.3), int(days * 0.5))
            peak_height = random.uniform(80, 100)
            values = []
            for i in range(len(dates)):
                if i < peak_day:
                    val = peak_height * (i / peak_day) ** 2
                else:
                    val = peak_height * np.exp(-0.03 * (i - peak_day))
                values.append(max(0, min(100, val)))

        elif curve_type == 1:  # Колебательный спад (рофл)
            peak_day = random.randint(int(days * 0.2), int(days * 0.4))
            peak_height = random.uniform(85, 100)
            values = []
            for i in range(len(dates)):
                if i < peak_day:
                    val = peak_height * (i / peak_day) ** 1.5
                else:
                    t = i - peak_day
                    val = peak_height * np.exp(-0.02 * t) * (1 + 0.3 * np.sin(0.1 * t))
                values.append(max(0, min(100, val)))

        elif curve_type == 2:  # Плато (персонажи)
            peak_day = random.randint(int(days * 0.4), int(days * 0.6))
            peak_height = random.uniform(70, 90)
            values = []
            for i in range(len(dates)):
                if i < peak_day:
                    val = peak_height * (i / peak_day) ** 1.2
                else:
                    val = peak_height * 0.4 + 15 * np.exp(-0.01 * (i - peak_day))
                values.append(max(0, min(100, val)))

        else:  # Быстрый взлёт (вирусное)
            peak_day = random.randint(int(days * 0.1), int(days * 0.2))
            peak_height = random.uniform(90, 100)
            values = []
            for i in range(len(dates)):
                if i < peak_day:
                    val = peak_height * (i / peak_day) ** 3
                else:
                    val = peak_height * np.exp(-0.08 * (i - peak_day))
                values.append(max(0, min(100, val)))

        # Добавляем шум
        values = np.array(values) + np.random.normal(0, 3, len(dates))
        values = np.maximum(0, np.minimum(100, values))

        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'query': query
        })

        print(f"     ✅ Сгенерировано {len(df)} точек")
        print(f"     📊 Тип кривой: {['Классика', 'Колебания', 'Плато', 'Взлёт'][curve_type]}")

        return df

    def get_daily_trends(self, geo: str = 'RU') -> List[str]:
        """
        Получает реальные тренды дня через RSS Google Trends
        """
        print(f"\n   🔍 Получение трендов через RSS ({geo})...")

        # Маппинг кодов стран для RSS
        geo_codes = {
            'RU': 'RU',  # Россия
            'US': 'US',  # США
            'KZ': 'KZ',  # Казахстан
            'BY': 'BY',  # Беларусь
            'UA': 'UA',  # Украина
            'DE': 'DE',  # Германия
            'FR': 'FR',  # Франция
            '': 'US'  # по умолчанию
        }

        code = geo_codes.get(geo, 'US')
        url = f'https://trends.google.com/trending/rss?geo={code}'

        try:
            # Парсим RSS
            feed = feedparser.parse(url)

            if feed.bozo:  # Ошибка парсинга
                print(f"     ⚠️ Ошибка парсинга RSS")
                return self._get_demo_trends()

            trends = []
            for entry in feed.entries[:10]:  # первые 10 трендов
                title = entry.title.lower()
                # Фильтруем слишком короткие и явно не мемные запросы
                if len(title) > 3 and not any(x in title for x in ['новости', 'погода', 'курс']):
                    trends.append(title)

            if trends:
                print(f"     ✅ Найдено {len(trends)} трендов")
                for i, t in enumerate(trends[:5], 1):
                    print(f"        {i}. {t}")
                return trends
            else:
                print(f"     ⚠️ Нет трендов в RSS")
                return self._get_demo_trends()

        except Exception as e:
            print(f"     ⚠️ Ошибка RSS: {e}")
            return self._get_demo_trends()

    def _get_demo_trends(self) -> List[str]:
        """Демо-тренды на случай ошибки"""
        print(f"     🧪 Использую демо-тренды")
        return ['сигма', 'скуф', 'альтушка', 'ноу вэй', 'чилл', 'душнила', 'кринж']


# ============================================================================
# БЛОК 5: АНАЛИЗ ЖИЗНЕННОГО ЦИКЛА
# ============================================================================

class MemeLifecycleAnalyzer:
    """Анализирует жизненный цикл мема"""

    def __init__(self):
        self.memes_db = {}
        self.load_saved_data()

    def load_saved_data(self):
        """Загружает сохранённые данные"""
        data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')

        if os.path.exists(data_file):
            try:
                with open(data_file, 'rb') as f:
                    self.memes_db = pickle.load(f)
                print(f"📂 Загружено {len(self.memes_db)} мемов из кэша")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки: {e}")

    def save_data(self):
        """Сохраняет данные"""
        data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')
        try:
            with open(data_file, 'wb') as f:
                pickle.dump(self.memes_db, f)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

    def add_meme_data(self, query: str, df: pd.DataFrame) -> bool:
        """Добавляет данные мема в базу"""
        if df.empty or len(df) < 30:
            print(f"   ⚠️ Недостаточно данных для '{query}'")
            return False

        # Сглаживаем ряд
        df = df.copy()
        df['smoothed'] = df['value'].rolling(Config.SMOOTH_WINDOW, center=True).mean()
        df['smoothed'] = df['smoothed'].fillna(df['value'])

        # Находим пик
        peak_idx, peak_value, peak_date = self._find_peak(df)

        # Определяем стадию
        stage, stage_info = self._determine_stage(df, peak_idx, peak_value)

        # Определяем тип траектории
        trajectory = self._classify_trajectory(df, peak_idx)

        # Сохраняем
        self.memes_db[query] = {
            'query': query,
            'data': df.to_dict('records'),
            'peak_idx': int(peak_idx),
            'peak_date': peak_date,
            'peak_value': float(peak_value),
            'stage': stage,
            'stage_info': stage_info,
            'trajectory': trajectory,
            'last_updated': datetime.now().isoformat()
        }

        self.save_data()
        print(f"   ✅ Мем '{query}' добавлен. Стадия: {Config.STAGES[stage]}")
        return True

    def _find_peak(self, df: pd.DataFrame) -> Tuple[int, float, datetime]:
        """Находит главный пик"""
        values = df['smoothed'].values

        peaks, _ = find_peaks(
            values,
            prominence=Config.PEAK_PROMINENCE,
            distance=Config.PEAK_DISTANCE
        )

        if len(peaks) == 0:
            peak_idx = np.argmax(values)
        else:
            peak_idx = peaks[np.argmax(values[peaks])]

        return peak_idx, float(values[peak_idx]), df.iloc[peak_idx]['date']

    def _determine_stage(self, df: pd.DataFrame, peak_idx: int, peak_value: float) -> Tuple[str, Dict]:
        """Определяет текущую стадию"""
        current = float(df.iloc[-1]['smoothed'])
        days_since = len(df) - peak_idx - 1
        rel = (current / peak_value) * 100 if peak_value > 0 else 0

        # Тренд за неделю
        growth = 0.0
        if len(df) >= 7:
            week_ago = float(df.iloc[-7]['smoothed'])
            if week_ago > 0:
                growth = ((current - week_ago) / week_ago) * 100

        # Определяем стадию
        if days_since < 0:  # Пик ещё не достигнут
            if rel < 20:
                stage = 'embryo'
                desc = 'Только зарождается'
            elif rel < 50:
                stage = 'growth'
                desc = 'Активный рост'
            elif rel < 80:
                stage = 'takeoff'
                desc = 'Взлёт'
            else:
                stage = 'peak'
                desc = 'На пике'
        else:  # Пик уже был
            if days_since < 7 and rel > 90:
                stage = 'peak'
                desc = 'На пике'
            elif days_since < 60 and rel > 30:
                stage = 'decay'
                desc = 'Затухание'
            elif rel > Config.DEATH_THRESHOLD:
                stage = 'stagnation'
                desc = 'Стагнация'
            else:
                recent = df.iloc[-Config.DEATH_DURATION:]
                below = (recent['smoothed'] < Config.DEATH_THRESHOLD).all()
                if below and len(recent) >= Config.DEATH_DURATION:
                    stage = 'death'
                    desc = 'Мёртв'
                else:
                    stage = 'stagnation'
                    desc = 'Стагнация'

        stage_info = {
            'stage': stage,
            'stage_display': Config.STAGES[stage],
            'description': desc,
            'current_value': current,
            'peak_value': peak_value,
            'rel_value': float(rel),
            'growth_rate': float(growth),
            'days_since_peak': max(0, days_since)
        }

        return stage, stage_info

    def _classify_trajectory(self, df: pd.DataFrame, peak_idx: int) -> str:
        """Классифицирует тип траектории"""
        if len(df) - peak_idx < 30:
            return 'smooth'

        post = df.iloc[peak_idx:]['smoothed'].values
        peaks_post, _ = find_peaks(post, prominence=5)

        plateau = float(np.mean(post[-14:])) if len(post) >= 14 else 0

        if plateau > 30:
            return 'plateau'
        elif len(peaks_post) >= 3:
            return 'oscillatory'
        else:
            return 'smooth'

    def get_meme(self, query: str) -> Optional[Dict]:
        """Возвращает данные мема"""
        return self.memes_db.get(query)

    def get_all_memes(self) -> List[Dict]:
        """Возвращает все мемы"""
        return list(self.memes_db.values())


# ============================================================================
# БЛОК 6: ВИЗУАЛИЗАЦИЯ
# ============================================================================

class MemeViz:
    """Визуализация"""

    def __init__(self):
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_lifecycle(self, meme_data: Dict, save_path: str = None):
        """Строит график"""
        query = meme_data['query']
        df = pd.DataFrame(meme_data['data'])
        peak_idx = meme_data['peak_idx']
        stage = meme_data['stage_info']

        plt.figure(figsize=(12, 6))

        dates = pd.to_datetime(df['date'])
        plt.plot(dates, df['value'], 'lightgray', lw=1, alpha=0.5, label='Исходные')
        plt.plot(dates, df['smoothed'], 'b-', lw=2, label='Сглаженные')
        plt.scatter(dates.iloc[peak_idx], df['value'].iloc[peak_idx],
                    color='red', s=150, label='Пик')
        plt.scatter(dates.iloc[-1], df['value'].iloc[-1],
                    color='green', s=150, label='Сейчас')

        plt.title(f'Жизненный цикл мема: {query}', fontsize=16)
        plt.xlabel('Дата')
        plt.ylabel('Популярность')
        plt.legend()
        plt.grid(True, alpha=0.3)

        text = f"Стадия: {stage['stage_display']}\nОт пика: {stage['rel_value']:.1f}%"
        plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
                 fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"   ✅ График сохранён: {save_path}")
        else:
            plt.show()
            plt.close()

    def print_meme_info(self, meme_data: Dict):
        """Выводит информацию"""
        q = meme_data['query']
        s = meme_data['stage_info']
        t = meme_data['trajectory']
        df = pd.DataFrame(meme_data['data'])

        print("\n" + "═" * 70)
        print(f"📊 МЕМ: {q}")
        print("═" * 70)

        print(f"\n📈 СТАТИСТИКА:")
        print(f"   • Период: {df['date'].min().date()} — {df['date'].max().date()}")
        print(f"   • Пик: {meme_data['peak_date'].date()} ({meme_data['peak_value']:.1f})")
        print(f"   • Текущее: {s['current_value']:.1f}")

        print(f"\n🎯 СТАДИЯ: {s['stage_display']}")
        print(f"   • {s['description']}")
        print(f"   • От пика: {s['rel_value']:.1f}%")
        print(f"   • Динамика: {s['growth_rate']:+.1f}%")
        print(f"   • Дней после пика: {s['days_since_peak']}")

        print(f"\n📉 ТРАЕКТОРИЯ: {Config.TRAJECTORIES[t]}")
        print("\n" + "═" * 70)


# ============================================================================
# БЛОК 7: ГЕНЕРАТОР ОТЧЁТОВ
# ============================================================================

class ReportGenerator:
    """Генератор отчётов"""

    def __init__(self, analyzer: MemeLifecycleAnalyzer):
        self.analyzer = analyzer

    def generate_weekly(self) -> str:
        """Генерирует еженедельный отчёт"""
        memes = self.analyzer.get_all_memes()

        lines = []
        lines.append("\n" + "📊" * 40)
        lines.append(f"   ЕЖЕНЕДЕЛЬНЫЙ ОТЧЁТ МЕМОСТАТ")
        lines.append(f"   {datetime.now().strftime('%d.%m.%Y')}")
        lines.append("📊" * 40 + "\n")

        if not memes:
            lines.append("📭 База мемов пуста")
            return "\n".join(lines)

        # Растущие
        growing = [m for m in memes if m['stage'] in ['embryo', 'growth', 'takeoff']]
        growing.sort(key=lambda x: x['stage_info']['growth_rate'], reverse=True)

        lines.append("🔥 ТОП-5 РАСТУЩИХ:")
        for i, m in enumerate(growing[:5], 1):
            st = m['stage_info']['stage_display']
            gr = m['stage_info']['growth_rate']
            lines.append(f"   {i}. {m['query']} — {st}, {gr:+.1f}%")

        # На пике
        peaking = [m for m in memes if m['stage'] == 'peak']
        if peaking:
            lines.append(f"\n🟡 НА ПИКЕ ({len(peaking)}):")
            for m in peaking[:5]:
                rel = m['stage_info']['rel_value']
                lines.append(f"   • {m['query']} ({rel:.0f}% от пика)")

        # Умирающие
        dying = [m for m in memes if m['stage'] in ['decay', 'stagnation']]
        dying.sort(key=lambda x: x['stage_info']['rel_value'])

        if dying:
            lines.append(f"\n💀 УМИРАЮЩИЕ ({len(dying)}):")
            for i, m in enumerate(dying[:5], 1):
                rel = m['stage_info']['rel_value']
                lines.append(f"   {i}. {m['query']} — {rel:.0f}% от пика")

        # Мёртвые
        dead = [m for m in memes if m['stage'] == 'death']
        if dead:
            lines.append(f"\n⚫ МЁРТВЫЕ ({len(dead)}):")
            for m in dead[:5]:
                lines.append(f"   • {m['query']}")

        lines.append("\n" + "📊" * 40)
        return "\n".join(lines)

    def save_report(self, filename: str = None):
        """Сохраняет отчёт"""
        if not filename:
            filename = os.path.join(
                Config.REPORTS_DIR,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_weekly())

        print(f"\n💾 Отчёт сохранён: {filename}")


# ============================================================================
# БЛОК 8: ОСНОВНОЙ КЛАСС
# ============================================================================

class MemeStat:
    """Главный класс"""

    def __init__(self):
        print("\n" + "🚀" * 50)
        print("   МЕМОСТАТ v6.0 — RSS + ДЕМО ВЕРСИЯ")
        print("   (реальные тренды + реалистичные демо-данные)")
        print("🚀" * 50 + "\n")

        self.parser = GoogleTrendsParser()
        self.analyzer = MemeLifecycleAnalyzer()
        self.viz = MemeViz()
        self.reporter = ReportGenerator(self.analyzer)
        self.behavioral = BehavioralAnalyzer()

        # Загружаем начальные данные, если база пуста
        if len(self.analyzer.memes_db) == 0:
            self._load_initial()

    def _load_initial(self):
        """Загружает начальный пул мемов"""
        print("\n📚 Загрузка начального пула мемов...")
        print("   (это займёт 20-30 секунд)")

        loaded = 0
        for i, meme in enumerate(Config.INITIAL_MEMES[:10], 1):
            print(f"\n   [{i}/10] {meme}")
            df = self.parser.get_interest_over_time(meme, 'today 12-m')
            if self.analyzer.add_meme_data(meme, df):
                loaded += 1
            time.sleep(1)

        print(f"\n✅ Загружено: {loaded} мемов")

    def update_all(self):
        """Обновляет все мемы"""
        print("\n🔄 Обновление данных...")
        memes = list(self.analyzer.memes_db.keys())

        for i, query in enumerate(memes, 1):
            print(f"\n   [{i}/{len(memes)}] {query}")
            df = self.parser.get_interest_over_time(query)
            if not df.empty:
                self.analyzer.add_meme_data(query, df)
            time.sleep(1)

        print("\n✅ Обновление завершено")

    def find_new(self):
        """Ищет новые мемы в трендах с фильтрацией (текстовой + поведенческой)"""
        print("\n🔍 Поиск новых мемов в трендах...")
        print("   (фильтруем по текстовым и поведенческим критериям)\n")

        trends = self.parser.get_daily_trends()

        if not trends:
            print("   ⚠️ Не удалось получить тренды")
            return

        # Фильтруем мемы (сначала текстовая проверка)
        memes_found = []
        news_found = []
        pending = []  # запросы, которые требуют поведенческого анализа

        for trend in trends:
            # 1. Быстрая текстовая проверка
            is_text_meme, text_reason = is_meme_candidate(trend)

            if is_text_meme:
                # Если текст похож на мем — добавляем сразу
                memes_found.append((trend, f"текст: {text_reason}"))
            else:
                # Если не похож, но может быть мемом (короткое слово) — откладываем
                if len(trend) < 15 and not any(stop in trend for stop in STOP_WORDS):
                    pending.append((trend, text_reason))
                else:
                    news_found.append((trend, f"текст: {text_reason}"))

        # 2. Поведенческий анализ для сомнительных запросов
        print(f"\n   📊 ПЕРВИЧНАЯ ФИЛЬТРАЦИЯ:")
        print(f"      ✅ Текстовые мемы: {len(memes_found)}")
        print(f"      ⏳ Требуют анализа: {len(pending)}")
        print(f"      ❌ Отсеяно текстом: {len(news_found)}")

        if pending:
            print(f"\n   🔬 ПРОВОДИМ ПОВЕДЕНЧЕСКИЙ АНАЛИЗ...")

            for trend, text_reason in pending:
                print(f"\n      Анализ: '{trend}'")

                # Загружаем данные (короткий период для скорости)
                df = self.parser.get_interest_over_time(trend, 'today 3-m')

                if df.empty or len(df) < 30:
                    news_found.append((trend, f"недостаточно данных: {text_reason}"))
                    continue

                # Поведенческий анализ
                analysis = self.behavioral.analyze_behavior(trend, df)
                self.behavioral.print_behavior_report(trend, analysis)

                if analysis['is_meme']:
                    memes_found.append((trend, analysis['reason']))
                else:
                    news_found.append((trend, analysis['reason']))

                time.sleep(0.5)

        # 3. Итоги фильтрации
        print(f"\n   📊 ИТОГОВАЯ ФИЛЬТРАЦИЯ:")
        print(f"   ✅ ОТОБРАНО МЕМОВ: {len(memes_found)}")
        print(f"   ❌ ОТСЕЯНО: {len(news_found)}")

        if memes_found:
            print("\n   🎯 НАЙДЕННЫЕ МЕМЫ:")
            for i, (meme, reason) in enumerate(memes_found[:15], 1):
                short_reason = reason[:50] + "..." if len(reason) > 50 else reason
                print(f"      {i}. {meme} — {short_reason}")

        # 4. Добавляем только мемы
        added = 0
        for trend, reason in memes_found:
            if trend not in self.analyzer.memes_db:
                print(f"\n   ➕ Добавление: {trend}")
                df = self.parser.get_interest_over_time(trend, 'today 3-m')
                if not df.empty and df['value'].max() > 20:
                    if self.analyzer.add_meme_data(trend, df):
                        added += 1
                time.sleep(1)

        print(f"\n✅ Добавлено новых мемов: {added}")

        # 5. Показываем примеры отсеянного
        if news_found:
            print(f"\n   📰 ПРИМЕРЫ ОТСЕЯННЫХ ЗАПРОСОВ:")
            for i, (item, reason) in enumerate(news_found[:5], 1):
                short_reason = reason[:50] + "..." if len(reason) > 50 else reason
                print(f"      {i}. {item} — {short_reason}")
            if len(news_found) > 5:
                print(f"      ... и ещё {len(news_found) - 5} запросов")

    def show(self, query: str):
        """Показывает информацию о меме"""
        meme = self.analyzer.get_meme(query)
        if not meme:
            print(f"\n❌ Мем '{query}' не найден")
            print("   Попробуйте: список, найти")
            return

        self.viz.print_meme_info(meme)

        choice = input("\n📈 Показать график? (y/n): ")
        if choice.lower() == 'y':
            plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
            self.viz.plot_lifecycle(meme, plot_path)

    def list_all(self):
        """Показывает все мемы"""
        memes = self.analyzer.get_all_memes()

        if not memes:
            print("\n📭 База мемов пуста")
            return

        stages = {}
        for m in memes:
            stages.setdefault(m['stage'], []).append(m)

        print("\n📋 ВСЕ МЕМЫ:")
        print("-" * 50)

        for stage_name, stage_display in Config.STAGES.items():
            if stage_name in stages:
                items = stages[stage_name]
                print(f"\n{stage_display} ({len(items)}):")
                for m in sorted(items, key=lambda x: x['stage_info']['rel_value'], reverse=True):
                    rel = m['stage_info']['rel_value']
                    growth = m['stage_info']['growth_rate']
                    print(f"   • {m['query']} — {rel:.0f}% от пика ({growth:+.0f}%)")

    def find_meme(self, query: str):
        """
        Ищет мем по названию:
        - если есть в базе — показывает информацию
        - если нет — ищет в трендах и добавляет
        """
        print(f"\n🔍 Поиск мема: '{query}'")

        meme = self.analyzer.get_meme(query)
        if meme:
            print(f"   ✅ Мем найден в базе!")
            self.viz.print_meme_info(meme)
            choice = input("\n📈 Показать график? (y/n): ")
            if choice.lower() == 'y':
                plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
                self.viz.plot_lifecycle(meme, plot_path)
            return

        print(f"   🔎 Мема '{query}' нет в базе. Пытаюсь добавить...")

        trends = self.parser.get_daily_trends()
        if query.lower() in [t.lower() for t in trends]:
            print(f"   ✅ Мем '{query}' найден в текущих трендах!")

        print(f"   📥 Загрузка данных для '{query}'...")
        df = self.parser.get_interest_over_time(query, 'today 12-m')

        if df.empty or df['value'].max() < 10:
            print(f"   ❌ Не удалось найти данные для '{query}'")
            print(f"   💡 Попробуйте другое написание или поищите через 'найти'")
            return

        if self.analyzer.add_meme_data(query, df):
            print(f"   ✅ Мем '{query}' успешно добавлен!")
            meme = self.analyzer.get_meme(query)
            if meme:
                self.viz.print_meme_info(meme)
                choice = input("\n📈 Показать график? (y/n): ")
                if choice.lower() == 'y':
                    plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
                    self.viz.plot_lifecycle(meme, plot_path)
        else:
            print(f"   ❌ Не удалось добавить мем '{query}'")

    def delete_all_memes(self):
        """Удаляет все мемы из базы данных"""
        print("\n⚠️ ВНИМАНИЕ! Вы собираетесь удалить ВСЕ мемы из базы.")
        print(f"   В базе сейчас {len(self.analyzer.memes_db)} мемов.")

        confirmation = input("\n   Подтвердите удаление (y/n): ").lower()

        if confirmation == 'y':
            self.analyzer.memes_db.clear()
            self.analyzer.save_data()

            data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')
            if os.path.exists(data_file):
                os.remove(data_file)

            if os.path.exists(Config.PLOTS_DIR):
                for file in os.listdir(Config.PLOTS_DIR):
                    file_path = os.path.join(Config.PLOTS_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass

            print("\n✅ ВСЕ мемы успешно удалены из базы!")
            print("   База теперь пуста. Используйте 'найти' или 'найти_мем', чтобы добавить новые мемы.")
        else:
            print("\n❌ Удаление отменено.")


# ============================================================================
# БЛОК 9: ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================================

def show_commands():
    """Показывает список доступных команд"""
    print("\n" + "📋" * 30)
    print("   ДОСТУПНЫЕ КОМАНДЫ:")
    print("   • список          — показать все мемы")
    print("   • инфо НАЗВАНИЕ    — информация о меме")
    print("   • найти мем НАЗВАНИЕ — поиск и добавление мема")
    print("   • обновить        — обновить все данные")
    print("   • найти           — поиск новых мемов в трендах")
    print("   • отчёт           — сгенерировать отчёт")
    print("   • удалить все     — УДАЛИТЬ ВСЕ МЕМЫ из базы")
    print("   • команды         — показать этот список")
    print("   • выход           — завершить")
    print("📋" * 30 + "\n")


def main():
    """Главная функция"""

    memestat = MemeStat()

    show_commands()

    while True:
        try:
            cmd = input("\n🔮 > ").strip().lower()

            if cmd == 'выход':
                print("\n👋 До свидания!")
                break

            elif cmd == 'команды':
                show_commands()
                continue

            elif cmd == 'список':
                memestat.list_all()

            elif cmd == 'обновить':
                memestat.update_all()

            elif cmd == 'найти':
                memestat.find_new()

            elif cmd == 'отчёт':
                memestat.reporter.save_report()

            elif cmd == 'удалить все':
                memestat.delete_all_memes()

            elif cmd.startswith('инфо '):
                query = cmd[5:].strip()
                memestat.show(query)

            elif cmd.startswith('найти мем '):
                query = cmd[9:].strip()
                memestat.find_meme(query)

            elif cmd:
                print(f"❌ Неизвестная команда: {cmd}")
                show_commands()
                continue

            print("\n" + "─" * 50)
            print("   Введите 'команды' для просмотра доступных команд")
            print("   Введите 'выход' для завершения")
            print("─" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            print("\n   Введите 'команды' для просмотра доступных команд")


if __name__ == "__main__":
    main()