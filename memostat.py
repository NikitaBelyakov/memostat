import os
import re
import time
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import feedparser

# Игнорируем предупреждения
warnings.filterwarnings('ignore')


# ============================================================================
# БЛОК 1: КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Конфигурация проекта"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'memostat_data')
    REPORTS_DIR = os.path.join(BASE_DIR, 'memostat_reports')
    PLOTS_DIR = os.path.join(BASE_DIR, 'memostat_plots')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')

    MAX_MEMES = 100
    PEAK_PROMINENCE = 15
    PEAK_DISTANCE = 30
    SMOOTH_WINDOW = 7
    DEATH_THRESHOLD = 5
    DEATH_DURATION = 30

    # НОВЫЕ СТАДИИ ЖИЗНЕННОГО ЦИКЛА
    STAGES = {
        'birth': '🟣 РОЖДЕНИЕ',
        'spread': '🟢 РАСПРОСТРАНЕНИЕ',
        'peak': '🟡 ПИК (ПЛАТО)',
        'fading': '🟠 УГАСАНИЕ',
        'death': '⚫ СМЕРТЬ'
    }

    TRAJECTORIES = {
        'smooth': '📉 Гладкий спад',
        'oscillatory': '📊 Колебательный спад',
        'plateau': '📈 Плато',
        'sustained': '🚀 Устойчивый рост'
    }

    INITIAL_MEMES = [
        'ждун', 'муся', 'альтушка', 'пепе'
    ]

    @classmethod
    def init_dirs(cls):
        for d in [cls.DATA_DIR, cls.REPORTS_DIR, cls.PLOTS_DIR, cls.CACHE_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"📁 Папки созданы")


Config.init_dirs()

# ============================================================================
# БЛОК 2: СЛОВАРИ ДЛЯ ФИЛЬТРАЦИИ МЕМОВ
# ============================================================================

SLANG_WORDS = {
    'кринж', 'хайп', 'рофл', 'краш', 'токс', 'сигма', 'скуф', 'альтушка',
    'вайб', 'пруф', 'агриться', 'зашквар', 'душнила', 'чилл', 'ауф', 'ноу вэй',
    'павлова', 'ждун', 'тролл', 'фейс', 'пепе', 'шрек', 'ленин гриб', 'превед медвед'
}

STOP_WORDS = {
    'новости', 'погода', 'курс', 'доллар', 'евро', 'нефть', 'война', 'выборы',
    'президент', 'путин', 'зеленский', 'байден', 'трамп', 'ковид', 'коронавирус',
    'санкции', 'рубль', 'биткоин', 'крипта', 'футбол', 'хоккей', 'теннис',
    'украина', 'россия', 'сша', 'китай', 'вчера', 'сегодня', 'завтра'
}


def is_meme_candidate(phrase: str) -> Tuple[bool, str]:
    """Локальная проверка, является ли фраза кандидатом в мемы"""
    phrase_lower = phrase.lower().strip()

    if len(phrase_lower) < 3:
        return False, "слишком короткое"
    if len(phrase_lower) > 30:
        return False, "слишком длинное"

    for stop in STOP_WORDS:
        if stop in phrase_lower:
            return False, f"стоп-слово: {stop}"

    if phrase_lower in SLANG_WORDS:
        return True, "сленговое слово"

    if ' ' in phrase_lower:
        words = phrase_lower.split()
        if 2 <= len(words) <= 4:
            meme_words = 0
            for w in words:
                if w in SLANG_WORDS:
                    meme_words += 1
            if meme_words >= len(words) // 2:
                return True, "составная фраза"

    if '"' in phrase or '«' in phrase or '»' in phrase:
        return True, "в кавычках"

    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]')
    if emoji_pattern.search(phrase):
        return True, "содержит эмодзи"

    return False, "не соответствует критериям"


# ============================================================================
# БЛОК 3: ПОВЕДЕНЧЕСКИЙ АНАЛИЗАТОР
# ============================================================================

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BehavioralAnalyzer:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_from_cache()

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty or len(df) < 30:
            return None

        values = df['value'].values

        cv = np.std(values) / (np.mean(values) + 0.01)
        kurt = kurtosis(values)
        skewness = skew(values)
        max_val = np.max(values)
        peak_idx = np.argmax(values)
        if peak_idx > 0:
            growth_rate = ((max_val - values[0]) / (peak_idx + 1) / (max_val + 0.01)) * 100
        else:
            growth_rate = 0
        days_after_peak = len(values) - peak_idx - 1
        if days_after_peak > 0:
            decay_rate = ((max_val - values[-1]) / (days_after_peak + 1) / (max_val + 0.01)) * 100
        else:
            decay_rate = 0
        half_max = max_val / 2
        above_half = np.where(values >= half_max)[0]
        fwhm = above_half[-1] - above_half[0] if len(above_half) > 0 else 0
        peaks, _ = find_peaks(values, prominence=max_val * 0.1)
        n_peaks = len(peaks) - 1 if len(peaks) > 0 else 0

        peak_ratio = max_val / (np.mean(values) + 0.01)

        features = np.array([
            cv, kurt, skewness, growth_rate, decay_rate, fwhm, n_peaks, peak_ratio
        ])

        return features

    def _train_from_cache(self):
        print("\n🎓 ОБУЧЕНИЕ МОДЕЛИ НА ДАННЫХ ИЗ КЭША")
        print("-" * 50)

        cache_files = []
        if os.path.exists(Config.CACHE_DIR):
            cache_files = [f for f in os.listdir(Config.CACHE_DIR) if f.endswith('.pkl')]

        if not cache_files:
            print("   ⚠️ Нет данных в кэше для обучения")
            print("   💡 Модель будет использовать эмпирические пороги")
            self.is_trained = False
            return

        print(f"   📁 Найдено {len(cache_files)} файлов в кэше")

        X = []  # признаки

        for file in cache_files:
            try:
                file_path = os.path.join(Config.CACHE_DIR, file)
                df = pd.read_pickle(file_path)

                if df.empty or len(df) < 30:
                    continue

                features = self.extract_features(df)
                if features is None:
                    continue

                X.append(features)

            except Exception as e:
                continue

        if len(X) < 5:
            print(f"   ⚠️ Недостаточно данных для обучения (нужно минимум 5, получено {len(X)})")
            print("   💡 Модель будет использовать эмпирические пороги")
            self.is_trained = False
            return

        X = np.array(X)

        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.model.fit(X_scaled)

        print(f"   ✅ Модель обучена на {len(X)} примерах мемов")


        self.is_trained = True


    def is_meme_like(self, df: pd.DataFrame) -> Tuple[bool, float, str]:
        features = self.extract_features(df)

        if features is None:
            return False, 0.0, "недостаточно данных"

        if self.is_trained and self.model is not None:
            features_scaled = self.scaler.transform([features])

            prediction = self.model.predict(features_scaled)[0]

            # Получаем степень "нормальности"
            score = self.model.score_samples(features_scaled)[0]
            # Нормализуем score в диапазон 0-1
            normalized_score = 1 / (1 + np.exp(-score))

            is_meme = prediction == 1

            reason = f"модель Isolation Forest (схожесть: {normalized_score:.1%})"
            return is_meme, normalized_score, reason

        return self._predict_empirical(features)

    def _predict_empirical(self, features: np.ndarray) -> Tuple[bool, float, str]:

        cv, kurt, skewness, growth_rate, decay_rate, fwhm, n_peaks, peak_ratio = features

        score = 0
        if cv > 1.5:
            score += 2
        elif cv > 1.0:
            score += 1
        if kurt > 3:
            score += 2
        elif kurt > 1:
            score += 1
        if skewness > 0.5:
            score += 2
        elif skewness > 0:
            score += 1
        if growth_rate > 5:
            score += 2
        elif growth_rate > 2:
            score += 1
        if decay_rate > 3:
            score += 2
        elif decay_rate > 1:
            score += 1
        if fwhm < 30:
            score += 2
        elif fwhm < 60:
            score += 1
        if 1 <= n_peaks <= 3:
            score += 1

        is_news = (fwhm > 60 and kurt < 1) or (abs(growth_rate - decay_rate) < 1 and growth_rate < 2)
        is_meme = (score >= 4) and not is_news and peak_ratio > 1.5

        confidence = min(0.95, score / 10) if is_meme else max(0.05, 1 - score / 10)

        return is_meme, confidence, f"эмпирическая модель (балл: {score}/10)"


# ============================================================================
# БЛОК 4: ПАРСЕР GOOGLE TRENDS
# ============================================================================

class GoogleTrendsParser:
    """Парсер данных Google Trends через pytrends"""

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.pytrends = None
        print("\n🔧 ИНИЦИАЛИЗАЦИЯ ПАРСЕРА")
        print("-" * 50)
        self._init_pytrends()
        print("-" * 50)

    def _init_pytrends(self):
        """Инициализация pytrends"""
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(
                hl='ru-RU',
                tz=180,
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.1,
            )
            print("   ✅ pytrends подключён")
        except ImportError:
            print("   ❌ pytrends не установлен!")
            print("   💡 Установите: pip install pytrends")
            raise
        except Exception as e:
            print(f"   ❌ Ошибка pytrends: {e}")
            raise

    def get_interest_over_time(self, query: str, timeframe: str = 'today 12-m', geo: str = 'RU') -> pd.DataFrame:
        """
        Получает данные из Google Trends через pytrends
        """
        print(f"\n   🔍 Запрос данных: '{query}' ({timeframe}, {geo})")

        # Проверяем кэш
        cache_file = os.path.join(self.cache_dir, f"{query}_{timeframe}_{geo}.pkl")
        if os.path.exists(cache_file):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - mod_time).days < 7:
                print(f"     📦 Загружено из кэша (от {mod_time.strftime('%d.%m.%Y')})")
                return pd.read_pickle(cache_file)

        try:
            # Защита от rate limiting
            time.sleep(2)

            # Строим запрос
            self.pytrends.build_payload(
                [query],
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )

            # Получаем данные
            df = self.pytrends.interest_over_time()

            if df.empty:
                print(f"     ⚠️ Нет данных для '{query}'")
                return pd.DataFrame()

            # Преобразуем в нужный формат
            df = df.reset_index()
            df = df.rename(columns={'date': 'date', query: 'value'})
            df['query'] = query

            # Убираем служебную колонку
            if 'isPartial' in df.columns:
                df = df.drop('isPartial', axis=1)

            print(f"     ✅ Получено {len(df)} точек данных")
            print(f"     📊 Период: {df['date'].min().date()} - {df['date'].max().date()}")
            print(f"     📈 Макс: {df['value'].max():.0f}, мин: {df['value'].min():.0f}")

            # Сохраняем в кэш
            df.to_pickle(cache_file)
            print(f"     💾 Сохранено в кэш")

            return df

        except Exception as e:
            print(f"     ❌ Ошибка pytrends: {e}")
            return pd.DataFrame()

    def get_daily_trends(self, geo: str = 'RU') -> List[str]:
        """
        Получает тренды дня через RSS Google Trends
        """
        print(f"\n   🔍 Получение трендов через RSS ({geo})...")

        geo_codes = {
            'RU': 'RU', 'US': 'US', 'KZ': 'KZ',
            'BY': 'BY', 'UA': 'UA', 'DE': 'DE', 'FR': 'FR', '': 'US'
        }
        code = geo_codes.get(geo, 'US')
        url = f'https://trends.google.com/trending/rss?geo={code}'

        try:
            feed = feedparser.parse(url)

            if feed.bozo:
                print(f"     ⚠️ Ошибка парсинга RSS")
                return self._get_demo_trends()

            trends = []
            for entry in feed.entries[:10]:  # первые 10 трендов
                title = entry.title.lower()
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
        """Только для случая ошибки RSS"""
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
        data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')
        if os.path.exists(data_file):
            try:
                with open(data_file, 'rb') as f:
                    self.memes_db = pickle.load(f)
                print(f"📂 Загружено {len(self.memes_db)} мемов из кэша")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки: {e}")

    def save_data(self):
        data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')
        try:
            with open(data_file, 'wb') as f:
                pickle.dump(self.memes_db, f)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

    def add_meme_data(self, query: str, df: pd.DataFrame) -> bool:
        """Добавляет данные мема в базу"""
        if df.empty or len(df) < 30:
            print(f"   ⚠️ Недостаточно данных для '{query}' (нужно минимум 30 дней)")
            return False

        # Сглаживаем ряд
        df = df.copy()
        df['smoothed'] = df['value'].rolling(Config.SMOOTH_WINDOW, center=True).mean()
        df['smoothed'] = df['smoothed'].fillna(df['value'])

        # Находим пик (максимум сглаженного ряда)
        peak_idx, peak_value, peak_date = self._find_peak(df)

        # Определяем стадию (НОВАЯ ЛОГИКА)
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
        """
        Находит главный пик популярности.
        Просто берём максимальное значение сглаженного ряда.
        """
        values = df['smoothed'].values

        peak_idx = np.argmax(values)
        peak_value = float(values[peak_idx])
        peak_date = df.iloc[peak_idx]['date']

        print(f"     🔍 Пик найден на {peak_idx} дне, значение: {peak_value:.1f}")

        return peak_idx, peak_value, peak_date

    def _determine_stage(self, df: pd.DataFrame, peak_idx: int, peak_value: float) -> Tuple[str, Dict]:
        """
        Определяет стадию жизненного цикла мема.
        Стадии: РОЖДЕНИЕ → РАСПРОСТРАНЕНИЕ → ПИК (ПЛАТО) → УГАСАНИЕ → СМЕРТЬ
        """
        current = float(df.iloc[-1]['smoothed'])
        days_since_peak = len(df) - peak_idx - 1  # дней прошло после пика (отрицательно, если пик ещё не достигнут)
        rel = (current / peak_value) * 100 if peak_value > 0 else 0  # текущее значение относительно пика (%)

        # Тренд за неделю (%)
        growth = 0.0
        if len(df) >= 7:
            week_ago = float(df.iloc[-7]['smoothed'])
            if week_ago > 0:
                growth = ((current - week_ago) / week_ago) * 100

        # ===== ОПРЕДЕЛЕНИЕ СТАДИИ =====

        # Если пик ещё не достигнут (данные заканчиваются до пика)
        if days_since_peak < 0:
            if rel < 20:
                stage = 'birth'
                desc = 'Мем только появился, набирает первые упоминания'
            elif rel < 70:
                stage = 'spread'
                desc = 'Мем активно распространяется, растёт популярность'
            else:
                stage = 'peak'
                desc = 'Мем достиг пика или плато'
        else:
            # Пик уже был
            if days_since_peak < 7 and rel > 80:
                stage = 'peak'
                desc = 'Мем на пике популярности'
            elif days_since_peak < 60 and rel > 25:
                stage = 'fading'
                desc = 'Популярность мема угасает'
            elif rel > Config.DEATH_THRESHOLD:
                stage = 'fading'
                desc = 'Мем угасает, но ещё встречается'
            else:
                # Проверяем, сколько дней значение ниже порога смерти
                recent = df.iloc[-Config.DEATH_DURATION:]
                below_threshold = (recent['smoothed'] < Config.DEATH_THRESHOLD).all()
                if below_threshold and len(recent) >= Config.DEATH_DURATION:
                    stage = 'death'
                    desc = 'Мем мёртв, практически не используется'
                else:
                    stage = 'fading'
                    desc = 'Мем на грани забвения'

        # Формируем информацию о стадии
        stage_info = {
            'stage': stage,
            'stage_display': Config.STAGES[stage],
            'description': desc,
            'current_value': current,
            'peak_value': peak_value,
            'rel_value': float(rel),
            'growth_rate': float(growth),
            'days_since_peak': max(0, days_since_peak)  # только положительные значения
        }

        return stage, stage_info

    def _classify_trajectory(self, df: pd.DataFrame, peak_idx: int) -> str:
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
        return self.memes_db.get(query)

    def get_all_memes(self) -> List[Dict]:
        return list(self.memes_db.values())


# ============================================================================
# БЛОК 6: ВИЗУАЛИЗАЦИЯ
# ============================================================================

class MemeViz:
    def __init__(self):
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_lifecycle(self, meme_data: Dict, save_path: str = None):
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
        plt.ylabel('Популярность (Google Trends)')
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
    def __init__(self, analyzer: MemeLifecycleAnalyzer):
        self.analyzer = analyzer

    def generate_weekly(self) -> str:
        memes = self.analyzer.get_all_memes()
        lines = []
        lines.append("\n" + "📊" * 40)
        lines.append(f"   ЕЖЕНЕДЕЛЬНЫЙ ОТЧЁТ МЕМОСТАТ")
        lines.append(f"   {datetime.now().strftime('%d.%m.%Y')}")
        lines.append("📊" * 40 + "\n")

        if not memes:
            lines.append("📭 База мемов пуста")
            return "\n".join(lines)

        # Распространяющиеся (РОЖДЕНИЕ и РАСПРОСТРАНЕНИЕ)
        spreading = [m for m in memes if m['stage'] in ['birth', 'spread']]
        spreading.sort(key=lambda x: x['stage_info']['growth_rate'], reverse=True)

        lines.append("📈 РАСПРОСТРАНЯЮЩИЕСЯ МЕМЫ:")
        for i, m in enumerate(spreading[:5], 1):
            lines.append(
                f"   {i}. {m['query']} — {m['stage_info']['stage_display']}, {m['stage_info']['growth_rate']:+.1f}%")

        # На пике (ПИК/ПЛАТО)
        peaking = [m for m in memes if m['stage'] == 'peak']
        if peaking:
            lines.append(f"\n🟡 НА ПИКЕ ({len(peaking)}):")
            for m in peaking[:5]:
                lines.append(f"   • {m['query']} ({m['stage_info']['rel_value']:.0f}% от пика)")

        # Угасающие
        fading = [m for m in memes if m['stage'] == 'fading']
        fading.sort(key=lambda x: x['stage_info']['rel_value'])
        if fading:
            lines.append(f"\n🟠 УГАСАЮЩИЕ МЕМЫ ({len(fading)}):")
            for i, m in enumerate(fading[:5], 1):
                lines.append(f"   {i}. {m['query']} — {m['stage_info']['rel_value']:.0f}% от пика")

        # Мёртвые
        dead = [m for m in memes if m['stage'] == 'death']
        if dead:
            lines.append(f"\n⚫ МЁРТВЫЕ МЕМЫ ({len(dead)}):")
            for m in dead[:5]:
                lines.append(f"   • {m['query']}")

        lines.append("\n" + "📊" * 40)
        return "\n".join(lines)

    def save_report(self, filename: str = None):
        if not filename:
            filename = os.path.join(Config.REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_weekly())
        print(f"\n💾 Отчёт сохранён: {filename}")


# ============================================================================
# БЛОК 8: ОСНОВНОЙ КЛАСС
# ============================================================================

class MemeStat:
    def __init__(self):
        print("\n" + "🚀" * 50)
        print("   МЕМОСТАТ v7.0 — РЕАЛЬНЫЕ ДАННЫЕ")
        print("   (RSS тренды + pytrends)")
        print("🚀" * 50 + "\n")

        self.parser = GoogleTrendsParser()
        self.analyzer = MemeLifecycleAnalyzer()
        self.viz = MemeViz()
        self.reporter = ReportGenerator(self.analyzer)
        self.behavioral = BehavioralAnalyzer()

        if len(self.analyzer.memes_db) == 0:
            self._load_initial()

    def _load_initial(self):
        print("\n📚 Загрузка начального пула мемов...")
        print("   (это займёт некоторое время, pytrends делает запросы)")
        loaded = 0
        for i, meme in enumerate(Config.INITIAL_MEMES, 1):
            print(f"\n   [{i}/{len(Config.INITIAL_MEMES)}] {meme}")
            df = self.parser.get_interest_over_time(meme, 'today 12-m')
            if self.analyzer.add_meme_data(meme, df):
                loaded += 1
            time.sleep(2)
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
        """Поиск новых мемов в трендах с использованием ML модели"""
        print("\n🔍 Поиск новых мемов в трендах...")
        print("   (с использованием машинного обучения)\n")

        trends = self.parser.get_daily_trends()
        if not trends:
            print("   ⚠️ Не удалось получить тренды")
            return

        memes_found = []
        news_found = []
        pending = []

        for trend in trends:
            is_text_meme, text_reason = is_meme_candidate(trend)
            if is_text_meme:
                memes_found.append((trend, f"текст: {text_reason}", 0.9))
            else:
                if len(trend) < 15 and not any(stop in trend for stop in STOP_WORDS):
                    pending.append((trend, text_reason))
                else:
                    news_found.append((trend, f"текст: {text_reason}", 0.0))

        print(f"\n   📊 ПЕРВИЧНАЯ ФИЛЬТРАЦИЯ:")
        print(f"      ✅ Текстовые мемы: {len(memes_found)}")
        print(f"      ⏳ Требуют ML анализа: {len(pending)}")
        print(f"      ❌ Отсеяно текстом: {len(news_found)}")

        if pending:
            print(f"\n   🔬 ПРОВОДИМ ML АНАЛИЗ...")
            for trend, text_reason in pending:
                print(f"\n      Анализ: '{trend}'")
                df = self.parser.get_interest_over_time(trend, 'today 3-m')

                if df.empty or len(df) < 30:
                    news_found.append((trend, f"недостаточно данных", 0.0))
                    continue

                # Используем ML модель
                is_meme_like, score, reason = self.behavioral.is_meme_like(df)

                if is_meme_like:
                    memes_found.append((trend, reason, score))
                    print(f"         ✅ ПОХОЖ НА МЕМ (схожесть: {score:.1%})")
                else:
                    news_found.append((trend, reason, score))
                    print(f"         ❌ НЕ ПОХОЖ НА МЕМ (схожесть: {score:.1%})")

        print(f"\n   📊 ИТОГОВАЯ ФИЛЬТРАЦИЯ:")
        print(f"   ✅ ОТОБРАНО: {len(memes_found)}")
        print(f"   ❌ ОТСЕЯНО: {len(news_found)}")

        if memes_found:
            print("\n   🎯 КАНДИДАТЫ В МЕМЫ:")
            for i, (meme, reason, score) in enumerate(memes_found[:15], 1):
                short_reason = reason[:40] + "..." if len(reason) > 40 else reason
                print(f"      {i}. {meme} — {short_reason} (схожесть: {score:.0%})")

        # Добавляем только те, что похожи на мемы
        added = 0
        for trend, reason, score in memes_found:
            if trend not in self.analyzer.memes_db:
                print(f"\n   ➕ Добавление: {trend}")
                df = self.parser.get_interest_over_time(trend, 'today 3-m')
                if not df.empty and df['value'].max() > 20:
                    if self.analyzer.add_meme_data(trend, df):
                        added += 1
                time.sleep(1)

        print(f"\n✅ Добавлено новых кандидатов: {added}")

    def show(self, query: str):
        meme = self.analyzer.get_meme(query)
        if not meme:
            print(f"\n❌ Мем '{query}' не найден")
            return
        self.viz.print_meme_info(meme)
        if input("\n📈 Показать график? (y/n): ").lower() == 'y':
            plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
            self.viz.plot_lifecycle(meme, plot_path)

    def list_all(self):
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
                    print(
                        f"   • {m['query']} — {m['stage_info']['rel_value']:.0f}% от пика ({m['stage_info']['growth_rate']:+.0f}%)")

    def find_meme(self, query: str):
        """
        Ищет мем по названию:
        - если есть в базе — показывает информацию
        - если нет — загружает реальные данные и добавляет
        """
        print(f"\n🔍 Поиск мема: '{query}'")

        # Проверяем, есть ли уже в базе
        meme = self.analyzer.get_meme(query)
        if meme:
            print(f"   ✅ Мем найден в базе!")
            self.viz.print_meme_info(meme)
            choice = input("\n📈 Показать график? (y/n): ")
            if choice.lower() == 'y':
                plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
                self.viz.plot_lifecycle(meme, plot_path)
            return

        # Если нет в базе, загружаем реальные данные
        print(f"   🔎 Мема '{query}' нет в базе. Загружаю реальные данные из Google Trends...")

        # Показываем, есть ли в текущих трендах
        trends = self.parser.get_daily_trends()
        if query.lower() in [t.lower() for t in trends]:
            print(f"   ✅ Мем '{query}' найден в текущих трендах!")

        # Загружаем данные
        print(f"   📥 Загрузка данных для '{query}' (это может занять 5-10 секунд)...")
        df = self.parser.get_interest_over_time(query, 'today 12-m')

        if df.empty or df['value'].max() < 10:
            print(f"   ❌ Не удалось найти данные для '{query}'")
            print(f"   💡 Возможные причины:")
            print(f"      1. Запрос слишком редкий")
            print(f"      2. Ошибка соединения с Google Trends")
            print(f"      3. Попробуйте другое написание")
            return

        # Добавляем в базу
        if self.analyzer.add_meme_data(query, df):
            print(f"\n   ✅ Мем '{query}' успешно добавлен в базу!")

            meme = self.analyzer.get_meme(query)
            if meme:
                self.viz.print_meme_info(meme)
                choice = input("\n📈 Построить график жизненного цикла? (y/n): ")
                if choice.lower() == 'y':
                    plot_path = os.path.join(Config.PLOTS_DIR, f"{query}.png")
                    self.viz.plot_lifecycle(meme, plot_path)
        else:
            print(f"   ❌ Не удалось добавить мем '{query}'")

    def delete_all_memes(self):
        """Удаляет все мемы из базы данных и очищает кэш"""
        print("\n⚠️ ВНИМАНИЕ! Вы собираетесь удалить ВСЕ мемы и очистить кэш.")
        print(f"   В базе сейчас {len(self.analyzer.memes_db)} мемов.")

        cache_files = []
        if os.path.exists(Config.CACHE_DIR):
            cache_files = [f for f in os.listdir(Config.CACHE_DIR) if f.endswith('.pkl')]
        print(f"   В кэше {len(cache_files)} файлов.")

        confirmation = input("\n   Подтвердите удаление (y/n): ").lower()

        if confirmation == 'y':
            self.analyzer.memes_db.clear()
            self.analyzer.save_data()

            data_file = os.path.join(Config.DATA_DIR, 'memes_db.pkl')
            if os.path.exists(data_file):
                os.remove(data_file)
                print("   ✅ База данных удалена")

            if os.path.exists(Config.PLOTS_DIR):
                plots_deleted = 0
                for file in os.listdir(Config.PLOTS_DIR):
                    file_path = os.path.join(Config.PLOTS_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            plots_deleted += 1
                    except:
                        pass
                if plots_deleted > 0:
                    print(f"   ✅ Удалено {plots_deleted} графиков")

            if os.path.exists(Config.CACHE_DIR):
                cache_deleted = 0
                for file in os.listdir(Config.CACHE_DIR):
                    file_path = os.path.join(Config.CACHE_DIR, file)
                    try:
                        if os.path.isfile(file_path) and file.endswith('.pkl'):
                            os.remove(file_path)
                            cache_deleted += 1
                    except:
                        pass
                if cache_deleted > 0:
                    print(f"   ✅ Удалено {cache_deleted} файлов из кэша")

            print("\n✅ ПОЛНОСТЬЮ ОЧИЩЕНО:")
            print("   • База мемов")
            print("   • Кэш Google Trends")
            print("   • Сохранённые графики")
            print("\n   База теперь пуста. Используйте 'найти' или 'найти_мем', чтобы добавить новые мемы.")
        else:
            print("\n❌ Удаление отменено.")


# ============================================================================
# БЛОК 9: ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================================

def show_commands():
    print("\n" + "📋" * 30)
    print("   ДОСТУПНЫЕ КОМАНДЫ:")
    print("   • список          — показать все мемы")
    print("   • инфо НАЗВАНИЕ    — информация о меме")
    print("   • найти мем НАЗВАНИЕ — поиск и добавление мема")
    print("   • обновить        — обновить все данные")
    print("   • найти           — поиск новых мемов в трендах")
    print("   • отчёт           — сгенерировать отчёт")
    print("   • удалить все     — УДАЛИТЬ ВСЕ МЕМЫ")
    print("   • команды         — показать этот список")
    print("   • выход           — завершить")
    print("📋" * 30 + "\n")


def main():
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
                memestat.show(cmd[5:].strip())
            elif cmd.startswith('найти мем '):
                memestat.find_meme(cmd[9:].strip())
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


if __name__ == "__main__":
    main()