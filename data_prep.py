"""
Крок 1. Підготовка даних для лабораторної "Linear Discrimination with SVM".

Що робить цей модуль:
  1. Читає CSV, завантажений з investing.com (GBP/USD Historical Data).
  2. Рахує добову зміну курсу та будує ознаки (features).
  3. Формує цільову змінну (target): +1 якщо наступного дня курс виріс, -1 якщо впав.
  4. Ділить дані на train/test ХРОНОЛОГІЧНО (без перемішування).
  5. Стандартизує ознаки (середнє і відхилення беруться лише з train).
"""

import numpy as np
import pandas as pd


# 1. Завантаження CSV з investing.com
def load_investing_csv(path: str) -> pd.DataFrame:
    """
    Читає файл з investing.com. Колонки там зазвичай такі:
        Date, Price, Open, High, Low, Vol., Change %
    (в україномовній версії назви інші, тому беремо колонки за ПОЗИЦІЄЮ,
    а не за назвою).
    Повертає DataFrame з колонками Date, Close, Open, High, Low,
    відсортований від старих дат до нових.
    """
    raw = pd.read_csv(path)

    df = pd.DataFrame()
    df["Date"] = raw.iloc[:, 0]
    df["Close"] = raw.iloc[:, 1]
    df["Open"] = raw.iloc[:, 2]
    df["High"] = raw.iloc[:, 3]
    df["Low"] = raw.iloc[:, 4]

    # Числа можуть бути рядками (з комами) - приводимо до float
    for col in ["Close", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )

    # Формат дати на investing.com: MM/DD/YYYY. Якщо інакше - pandas спробує вгадати.
    parsed = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = parsed

    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df


# 2-3. Ознаки та цільова змінна
def _feature_frame(df: pd.DataFrame, n_lags: int):
    """
    Спільна частина для бінарної та багатокласової постановки: будує ознаки
    та колонку next_ret (зміна курсу НАСТУПНОГО дня, у %).

    Ознаки описують ТІЛЬКИ минуле (дні t, t-1, ...), а next_ret - що сталося на день t+1.
    Так ми не "підглядаємо" у майбутнє.

      ret_lag0 ... ret_lag{n-1}  - добові зміни курсу (у %) за останні n днів
      ma_ratio                   - Close / середнє Close за 10 днів - 1  (у %)
      volatility                 - стандартне відхилення добових змін за 10 днів
      range_pct                  - (High - Low) / Close  (у %), "розмах" дня
    """
    d = df.copy()

    d["ret"] = d["Close"].pct_change() * 100  # добова зміна, %

    feature_names = []
    for k in range(n_lags):
        name = f"ret_lag{k}"
        d[name] = d["ret"].shift(k)
        feature_names.append(name)

    d["ma_ratio"] = (d["Close"] / d["Close"].rolling(10).mean() - 1) * 100
    d["volatility"] = d["ret"].rolling(10).std()
    d["range_pct"] = (d["High"] - d["Low"]) / d["Close"] * 100
    feature_names += ["ma_ratio", "volatility", "range_pct"]

    d["next_ret"] = d["ret"].shift(-1)  # зміна ЗАВТРА відносно сьогодні
    return d, feature_names


def build_dataset(df: pd.DataFrame, n_lags: int = 5):
    """
    Бінарна постановка: мітка +1, якщо наступного дня курс виріс, -1, якщо впав.
    Дні, коли курс не змінився (next_ret == 0), прибираються.
    """
    d, feature_names = _feature_frame(df, n_lags)
    d["target"] = np.where(d["next_ret"] > 0, 1, -1)

    d = d.dropna(subset=feature_names + ["next_ret"])
    d = d[d["next_ret"] != 0].reset_index(drop=True)

    X = d[feature_names].to_numpy(dtype=float)
    y = d["target"].to_numpy(dtype=int)
    dates = d["Date"].to_numpy()
    return X, y, feature_names, dates


def build_dataset_multiclass(df: pd.DataFrame, n_lags: int = 5):
    """
    Багатокласова постановка, крок 1: повертає ознаки X та "сиру" зміну курсу наступного дня next_ret.
    Самі класи формуються пізніше функцією make_quantile_classes, ПІСЛЯ поділу на train/test.
    Нульові дні тут не прибираються: вони природно потрапляють у "середні" класи.
    """
    d, feature_names = _feature_frame(df, n_lags)
    d = d.dropna(subset=feature_names + ["next_ret"]).reset_index(drop=True)

    X = d[feature_names].to_numpy(dtype=float)
    next_ret = d["next_ret"].to_numpy(dtype=float)
    dates = d["Date"].to_numpy()
    return X, next_ret, feature_names, dates


def make_quantile_classes(ret_train, ret_test, n_classes: int = 3):
    """
    Багатокласова постановка, крок 2: ділимо дні на n_classes класів за величиною зміни курсу.
    Межі беруться як квантилі ТІЛЬКИ по train (щоб тест не впливав на розмітку).
    Клас 0 - найсильніше падіння, клас n_classes-1 - найсильніше зростання.
    Приклад для 3 класів: нижня третина змін = "вниз", середня = "боковик", верхня = "вгору".
    Повертає (y_train, y_test, edges), де edges - межі класів у відсотках.
    """
    qs = np.arange(1, n_classes) / n_classes
    edges = np.quantile(ret_train, qs)
    y_train = np.digitize(ret_train, edges)
    y_test = np.digitize(ret_test, edges)
    return y_train, y_test, edges


# 4. Хронологічний поділ
def time_split(X, y, train_ratio: float = 0.8):
    """Перші train_ratio днів - навчання, решта - тест. БЕЗ перемішування."""
    n_train = int(len(X) * train_ratio)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# 5. Стандартизація
def standardize(X_train, X_test):
    """
    Приводимо кожну ознаку до середнього 0 і відхилення 1.
    Середнє та відхилення рахуємо ТІЛЬКИ по train і застосовуємо до обох наборів,
    інакше інформація з тесту "просочиться" у навчання (data leakage).
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # захист від ділення на нуль
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
