"""
Крок 4. Багатокласовий SVM методом "один проти всіх" (one-vs-rest, OvR).

SVM за своєю природою розділяє лише два класи. Щоб працювати з K класами:
  1. Для кожного класу k навчаємо окремий бінарний SVM: "клас k" (+1) проти "усі інші" (-1).
  2. Отримуємо K балів для кожного об'єкта (по одному від кожного бінарного SVM).
  3. Передбачаємо той клас, чий бал найбільший (argmax).

"""

import numpy as np

from svm_custom import CustomSVM


class OneVsRestSVM:
    def __init__(self, n_classes: int, learning_rate: float = 0.1, lambda_param: float = 0.01,
                 n_iters: int = 2000, balanced: bool = True):
        self.n_classes = n_classes
        self.models = [
            CustomSVM(learning_rate=learning_rate, lambda_param=lambda_param,
                      n_iters=n_iters, balanced=balanced)
            for _ in range(n_classes)
        ]

    def fit(self, X, y):
        """y - цілі мітки класів 0 ... K-1."""
        y = np.asarray(y)
        for k, model in enumerate(self.models):
            y_binary = np.where(y == k, 1, -1)  # клас k проти решти
            model.fit(X, y_binary)
        return self

    def decision_function(self, X):
        """Матриця балів розміром (кількість об'єктів) x K: стовпець k - бал k-го бінарного SVM."""
        return np.column_stack([m.decision_function(X) for m in self.models])

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)
