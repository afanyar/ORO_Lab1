"""
Крок 2: навчання власного та готового SVM і порівняння точності.

Запуск:  python step2_train.py

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from data_prep import build_dataset, load_investing_csv, standardize, time_split
from svm_custom import CustomSVM

CSV_PATH = "GBP_USD Historical Data.csv" 
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

LAMBDA = 0.01  # сила регуляризації власного SVM


def report(title, y_true, y_pred):
    """Друкує accuracy, звіт по класах і матрицю помилок."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nЗвіт по класах:")
    print(classification_report(y_true, y_pred, labels=[-1, 1], target_names=["Вниз (-1)", "Вгору (+1)"], zero_division=0))
    print("Матриця помилок (рядки - справжній клас, стовпці - передбачений; порядок: -1, +1):")
    print(confusion_matrix(y_true, y_pred, labels=[-1, 1]))
    return acc


def main():
    # --- Дані (з кроку 1) ---
    df = load_investing_csv(CSV_PATH)
    X, y, names, dates = build_dataset(df, n_lags=5)
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    # --- Базовий рівень: що буде, якщо завжди вгадувати найчастіший клас ---
    majority = 1 if np.mean(y_train == 1) >= 0.5 else -1
    baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority))
    print(f"Базовий рівень (завжди клас {majority:+d}): accuracy = {baseline_acc:.4f}")

    # --- Власний SVM ---
    my_svm = CustomSVM(learning_rate=0.1, lambda_param=LAMBDA, n_iters=2000)
    my_svm.fit(X_train_s, y_train)
    acc_train = accuracy_score(y_train, my_svm.predict(X_train_s))
    print(f"\nВласний SVM: фінальна втрата J = {my_svm.loss_history_[-1]:.4f}, accuracy на train = {acc_train:.4f}")
    report("ВЛАСНИЙ SVM (test)", y_test, my_svm.predict(X_test_s))

    # --- Готовий SVM з sklearn ---
    # C у sklearn і lambda у нашій формулі пов'язані так: C = 1 / (lambda * n).
    # Беремо відповідне C, щоб порівняння було чесним (однакова "сила регуляризації").
    C = 1.0 / (LAMBDA * len(X_train_s))
    sk_svm = SVC(kernel="linear", C=C, class_weight="balanced")  # balanced - те саме, що balanced=True у власному SVM
    sk_svm.fit(X_train_s, y_train)
    report("ГОТОВИЙ SVM, sklearn SVC (test)", y_test, sk_svm.predict(X_test_s))

    # --- Порівняння ваг: якщо реалізація правильна, вони мають бути близькими ---
    print("\nПорівняння знайдених ваг (власний vs sklearn):")
    print(f"{'ознака':<12}{'власний':>10}{'sklearn':>10}")
    for name, w_my, w_sk in zip(names, my_svm.w, sk_svm.coef_[0]):
        print(f"{name:<12}{w_my:>10.4f}{w_sk:>10.4f}")
    print(f"{'b (зсув)':<12}{my_svm.b:>10.4f}{sk_svm.intercept_[0]:>10.4f}")

    # --- Графік збіжності ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(my_svm.loss_history_)
    ax.set_xlabel("Крок градієнтного спуску")
    ax.set_ylabel("Функція втрат J")
    ax.set_title("Збіжність власного SVM")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/5_loss_convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
