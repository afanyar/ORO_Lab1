"""
Крок 1: запуск підготовки даних і візуалізація їх розподілу.

Запуск:  python step1_explore.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from data_prep import build_dataset, load_investing_csv, standardize, time_split

CSV_PATH = "GBP_USD Historical Data.csv" 
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_UP = "tab:green"
COLOR_DOWN = "tab:red"


def main():
    # --- Дані ---
    df = load_investing_csv(CSV_PATH)
    print(f"Завантажено днів: {len(df)}  ({df['Date'].min().date()} - {df['Date'].max().date()})")

    X, y, names, dates = build_dataset(df, n_lags=5)
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=0.8)
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)

    print(f"Об'єктів: {len(X)}, ознак: {X.shape[1]}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Частка 'вгору' у train: {np.mean(y_train == 1):.3f}, у test: {np.mean(y_test == 1):.3f}")

    # --- Графік 1: баланс класів ---
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Train", "Test"]
    up = [np.sum(y_train == 1), np.sum(y_test == 1)]
    down = [np.sum(y_train == -1), np.sum(y_test == -1)]
    x = np.arange(2)
    ax.bar(x - 0.2, up, 0.4, color=COLOR_UP, label="Вгору (+1)")
    ax.bar(x + 0.2, down, 0.4, color=COLOR_DOWN, label="Вниз (-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Кількість днів")
    ax.set_title("Баланс класів")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/1_class_balance.png", dpi=150)

    # --- Графік 2: гістограми ознак за класами ---
    n_feat = X.shape[1]
    cols = 4
    rows = int(np.ceil(n_feat / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = axes.ravel()
    for i, name in enumerate(names):
        axes[i].hist(X_train[y_train == 1, i], bins=30, alpha=0.6, color=COLOR_UP, label="Вгору")
        axes[i].hist(X_train[y_train == -1, i], bins=30, alpha=0.6, color=COLOR_DOWN, label="Вниз")
        axes[i].set_title(name)
    for j in range(n_feat, len(axes)):
        axes[j].axis("off")
    axes[0].legend()
    fig.suptitle("Розподіл ознак (train) за класами", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/2_feature_histograms.png", dpi=150)

    # --- Графік 3: діаграма розсіювання двох ознак ---
    i, j = names.index("ret_lag0"), names.index("ret_lag1")
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, text in [(1, COLOR_UP, "Вгору"), (-1, COLOR_DOWN, "Вниз")]:
        m = y_train == label
        ax.scatter(X_train_s[m, i], X_train_s[m, j], s=8, alpha=0.5, color=color, label=text)
    ax.set_xlabel("ret_lag0 (стандартизована)")
    ax.set_ylabel("ret_lag1 (стандартизована)")
    ax.set_title("Дві ознаки: чи видно межу між класами?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/3_scatter_two_features.png", dpi=150)

    # --- Графік 4: проєкція всіх ознак на 2D через PCA ---
    # PCA руками через SVD: центруємо дані, беремо дві головні осі
    _, _, Vt = np.linalg.svd(X_train_s - X_train_s.mean(axis=0), full_matrices=False)
    Z = X_train_s @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, text in [(1, COLOR_UP, "Вгору"), (-1, COLOR_DOWN, "Вниз")]:
        m = y_train == label
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.5, color=color, label=text)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA-проєкція ознак на площину")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/4_pca_projection.png", dpi=150)

    print(f"Графіки збережено у папку '{FIG_DIR}'")
    plt.show()


if __name__ == "__main__":
    main()
