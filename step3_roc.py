"""
Крок 3: ROC-криві та AUC для власного й готового SVM.

Запуск:  python step3_roc.py

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC

from data_prep import build_dataset, load_investing_csv, standardize, time_split
from roc_utils import auc_bootstrap_ci, auc_trapezoid, roc_curve_manual
from svm_custom import CustomSVM

CSV_PATH = "GBP_USD Historical Data.csv" 
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

LAMBDA = 0.01  # те саме значення, що й на кроці 2


def main():
    # --- Дані та моделі (як на кроці 2) ---
    df = load_investing_csv(CSV_PATH)
    X, y, names, dates = build_dataset(df, n_lags=5)
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    my_svm = CustomSVM(learning_rate=0.1, lambda_param=LAMBDA, n_iters=2000).fit(X_train_s, y_train)
    C = 1.0 / (LAMBDA * len(X_train_s))
    sk_svm = SVC(kernel="linear", C=C, class_weight="balanced").fit(X_train_s, y_train)

    # --- Бали (decision_function) для обох моделей ---
    # ВАЖЛИВО: у ROC подаємо бали, а не передбачені мітки +1/-1!
    scores = {
        "Власний SVM": my_svm.decision_function(X_test_s),
        "sklearn SVC": sk_svm.decision_function(X_test_s),
    }
    scores_train = {
        "Власний SVM": my_svm.decision_function(X_train_s),
        "sklearn SVC": sk_svm.decision_function(X_train_s),
    }

    # --- ROC і AUC ---
    curves = {}
    print(f"{'Модель':<14}{'AUC (вручну)':>14}{'AUC (sklearn)':>15}{'AUC на train':>14}   95% інтервал AUC на test")
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve_manual(y_test, s)
        auc_manual = auc_trapezoid(fpr, tpr)
        auc_sk = roc_auc_score(y_test, s)  # перевірка
        fpr_tr, tpr_tr, _ = roc_curve_manual(y_train, scores_train[name])
        auc_train = auc_trapezoid(fpr_tr, tpr_tr)
        lo, hi = auc_bootstrap_ci(y_test, s)
        curves[name] = (fpr, tpr, auc_manual)
        print(f"{name:<14}{auc_manual:>14.4f}{auc_sk:>15.4f}{auc_train:>14.4f}   [{lo:.3f}; {hi:.3f}]")

        # Додаткова перевірка: наша крива має збігатися з sklearn
        # (drop_intermediate=False, щоб sklearn не викидав "зайві" точки на прямих ділянках)
        fpr_sk, tpr_sk, _ = roc_curve(y_test, s, drop_intermediate=False)
        same = len(fpr) == len(fpr_sk) and np.allclose(fpr, fpr_sk) and np.allclose(tpr, tpr_sk)
        print(f"{'':<14}крива збігається з sklearn: {same}")

    # --- Графік 6: ROC-криві обох моделей ---
    fig, ax = plt.subplots(figsize=(6, 6))
    styles = {"Власний SVM": ("tab:blue", "-", 2.5), "sklearn SVC": ("tab:orange", "--", 1.8)}
    for name, (fpr, tpr, auc) in curves.items():
        color, ls, lw = styles[name]
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Випадкове вгадування (AUC = 0.5)")
    ax.set_xlabel("FPR (частка хибних 'вгору')")
    ax.set_ylabel("TPR (частка знайдених 'вгору')")
    ax.set_title("ROC-криві (тестова вибірка)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/6_roc_curves.png", dpi=150)
    print(f"\nГрафік збережено: {FIG_DIR}/6_roc_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
