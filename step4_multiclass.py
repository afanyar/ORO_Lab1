"""
Крок 4 (бонус): багатокласова класифікація - "multi class championship".

Запуск:  python step4_multiclass.py
(потрібні data_prep.py, svm_custom.py, svm_multiclass.py, roc_utils.py і CSV-файл поруч)

Що робимо:
  * ділимо дні на N_CLASSES класів за величиною зміни курсу наступного дня
    (для 3 класів: "вниз" / "боковик" / "вгору");
  * навчаємо власний SVM "один проти всіх" та готовий sklearn для порівняння;
  * рахуємо accuracy, матрицю помилок, ROC-криві (по кожному класу проти решти) та AUC.

Кількість класів змінюється однією змінною N_CLASSES нижче.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from data_prep import build_dataset_multiclass, load_investing_csv, make_quantile_classes, standardize, time_split
from roc_utils import auc_trapezoid, roc_curve_manual
from svm_multiclass import OneVsRestSVM

CSV_PATH = "GBP_USD Historical Data.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

N_CLASSES = 3   # <- кількість класів
LAMBDA = 0.01   # сила регуляризації, як на попередніх кроках


def class_names(n_classes):
    if n_classes == 3:
        return ["Вниз", "Нейтрально", "Вгору"]
    names = [f"Клас {k}" for k in range(n_classes)]
    names[0] += " (сильне падіння)"
    names[-1] += " (сильне зростання)"
    return names


def macro_auc_ovr(y_true, scores, n_classes):
    """Середнє AUC по класах, кожен клас проти решти (власна реалізація ROC)."""
    aucs = []
    for k in range(n_classes):
        fpr, tpr, _ = roc_curve_manual(np.where(y_true == k, 1, -1), scores[:, k])
        aucs.append(auc_trapezoid(fpr, tpr))
    return aucs, float(np.mean(aucs))


def bootstrap_macro_auc_ci(y_true, scores, n_classes, n_boot=500, seed=42):
    """95% інтервал для macro-AUC бутстрепом (перевибірка тестових днів з поверненням)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < n_classes:
            continue
        vals.append(macro_auc_ovr(y_true[idx], scores[idx], n_classes)[1])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def plot_confusion(ax, cm, names, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


def main():
    names_cls = class_names(N_CLASSES)

    # --- Дані ---
    df = load_investing_csv(CSV_PATH)
    X, next_ret, feat_names, dates = build_dataset_multiclass(df, n_lags=5)
    X_train, X_test, r_train, r_test = time_split(X, next_ret, train_ratio=0.8)
    y_train, y_test, edges = make_quantile_classes(r_train, r_test, N_CLASSES)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    print(f"Класів: {N_CLASSES}. Межі класів за зміною курсу наступного дня (%, по train):")
    print("  " + ", ".join(f"{e:+.4f}" for e in edges))
    print(f"{'Клас':<28}{'train':>8}{'test':>8}")
    for k in range(N_CLASSES):
        print(f"{names_cls[k]:<28}{np.sum(y_train == k):>8}{np.sum(y_test == k):>8}")

    majority = int(np.bincount(y_train).argmax())
    baseline = accuracy_score(y_test, np.full_like(y_test, majority))
    print(f"\nБазовий рівень (завжди '{names_cls[majority]}'): accuracy = {baseline:.4f}; "
          f"випадкове вгадування: {1 / N_CLASSES:.4f}")

    # --- Моделі ---
    my_ovr = OneVsRestSVM(N_CLASSES, learning_rate=0.1, lambda_param=LAMBDA, n_iters=2000).fit(X_train_s, y_train)
    C = 1.0 / (LAMBDA * len(X_train_s))
    sk_ovr = OneVsRestClassifier(SVC(kernel="linear", C=C, class_weight="balanced")).fit(X_train_s, y_train)

    models = {"Власний SVM (OvR)": my_ovr, "sklearn SVC (OvR)": sk_ovr}
    all_scores, all_cms = {}, {}

    for title, model in models.items():
        y_pred = model.predict(X_test_s)
        scores = model.decision_function(X_test_s)
        all_scores[title] = scores

        print("\n" + "=" * 64)
        print(title.upper() + " (test)")
        print("=" * 64)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, labels=list(range(N_CLASSES)),
                                    target_names=names_cls, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))
        all_cms[title] = cm
        print("Матриця помилок (рядки - справжній клас, стовпці - передбачений):")
        print(cm)

    # --- ROC / AUC по класах (клас проти решти) ---
    print("\n" + "=" * 64)
    print("AUC по класах (клас проти решти), test")
    print("=" * 64)
    print(f"{'Модель':<22}" + "".join(f"{n[:12]:>14}" for n in names_cls) + f"{'macro':>10}   95% інтервал macro")
    curves = {}
    for title, scores in all_scores.items():
        aucs, macro = macro_auc_ovr(y_test, scores, N_CLASSES)
        lo, hi = bootstrap_macro_auc_ci(y_test, scores, N_CLASSES)
        print(f"{title:<22}" + "".join(f"{a:>14.4f}" for a in aucs) + f"{macro:>10.4f}   [{lo:.3f}; {hi:.3f}]")

        # Перевірка: наш AUC має збігатися з sklearn для кожного класу
        ok = all(np.isclose(aucs[k], roc_auc_score((y_test == k).astype(int), scores[:, k]))
                 for k in range(N_CLASSES))
        print(f"{'':<22}збігається з sklearn по всіх класах: {ok}")
        curves[title] = [(*roc_curve_manual(np.where(y_test == k, 1, -1), scores[:, k])[:2], aucs[k])
                         for k in range(N_CLASSES)]

    # --- Графік 7: матриці помилок ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, cm) in zip(axes, all_cms.items()):
        plot_confusion(ax, cm, names_cls, title)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/7_confusion_matrices.png", dpi=150)

    # --- Графік 8: ROC-криві по класах ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (title, cl) in zip(axes, curves.items()):
        for k, (fpr, tpr, auc) in enumerate(cl):
            ax.plot(fpr, tpr, linewidth=2, label=f"{names_cls[k]} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Випадкове вгадування")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{title}: ROC, клас проти решти")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/8_roc_multiclass.png", dpi=150)

    print(f"\nГрафіки збережено: {FIG_DIR}/7_confusion_matrices.png, {FIG_DIR}/8_roc_multiclass.png")
    plt.show()


if __name__ == "__main__":
    main()
