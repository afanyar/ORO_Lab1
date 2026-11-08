"""
Крок 3. ROC-крива та AUC, написані вручну на numpy.

Ідея ROC-кривої:
  Класифікатор видає бал (decision_function = w.x + b). Рішення "вгору" приймається, якщо бал >= порогу.
  Змінюючи поріг від дуже високого до дуже низького, ми отримуємо різні пари:
      TPR = TP / P   - частка справжніх "вгору", які ми знайшли (чутливість)
      FPR = FP / N   - частка справжніх "вниз", які ми помилково назвали "вгору"
  Крива в координатах (FPR, TPR) і є ROC-кривою.

  * Ідеальний класифікатор: крива йде у верхній лівий кут, AUC = 1.
  * Випадкове вгадування: діагональ від (0,0) до (1,1), AUC = 0.5.
"""

import numpy as np


def roc_curve_manual(y_true, scores):
    """
    y_true - мітки {-1, +1} (позитивний клас = +1, тобто "вгору").
    scores - бали класифікатора (чим більший, тим впевненіше "вгору").
    Повертає масиви (fpr, tpr, thresholds).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    # Сортуємо об'єкти за балом від найбільшого до найменшого
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    positive = (y_true[order] > 0).astype(float)

    # Якщо поріг опускається до i-го об'єкта, то всі об'єкти до нього (включно) названі "вгору"
    tp = np.cumsum(positive)        # скільки з них справді "вгору"
    fp = np.cumsum(1.0 - positive)  # скільки з них насправді "вниз"

    # Поріг можна ставити тільки МІЖ різними значеннями балів (якщо бали однакові, їх не розділити)
    distinct = np.r_[np.where(np.diff(scores_sorted) != 0)[0], len(scores_sorted) - 1]

    tpr = np.r_[0.0, tp[distinct] / tp[-1]]
    fpr = np.r_[0.0, fp[distinct] / fp[-1]]
    thresholds = np.r_[np.inf, scores_sorted[distinct]]
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """Площа під кривою методом трапецій."""
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_bootstrap_ci(y_true, scores, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """
    Довірчий інтервал для AUC методом бутстрепу: багато разів перевибираємо тестові дні
    (з поверненням) і дивимося, наскільки сильно "стрибає" AUC.
    Повертає (нижня межа, верхня межа).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:  # у вибірці лишився один клас - AUC не визначений
            continue
        fpr, tpr, _ = roc_curve_manual(yb, scores[idx])
        aucs.append(auc_trapezoid(fpr, tpr))
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))
