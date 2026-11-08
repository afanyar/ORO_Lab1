"""
Крок 2. Власна реалізація лінійного SVM (soft margin).

Мінімізуємо функцію втрат:

    J(w, b) = (lambda/2) * ||w||^2  +  (1/n) * sum_i max(0, 1 - y_i * (w.x_i + b))
              (перший доданок - регуляризація, другий - hinge loss, штраф за помилки)

  * перший доданок змушує вектор w бути коротким, а це і є широкий відступ (margin);
  * другий доданок штрафує точки, які опинилися всередині відступу або з неправильного боку.

Мінімізуємо звичайним градієнтним спуском: на кожному кроці рахуємо градієнт
по ВСІХ об'єктах одразу.
"""

import numpy as np


class CustomSVM:
    def __init__(self, learning_rate: float = 0.1, lambda_param: float = 0.01, n_iters: int = 2000,
                 balanced: bool = True):
        self.learning_rate = learning_rate  # довжина кроку градієнтного спуску
        self.lambda_param = lambda_param    # сила регуляризації (більше lambda - ширший відступ, більше помилок)
        self.n_iters = n_iters              # кількість кроків спуску
        self.balanced = balanced            # True - помилки на рідшому класі штрафуються сильніше (див. fit)
        self.w = None
        self.b = 0.0
        self.loss_history_ = []             # значення J на кожному кроці (для графіка збіжності)

    def _to_pm1(self, y):
        """Мітки приводимо до {-1, +1}, як того вимагають формули."""
        return np.where(np.asarray(y) > 0, 1.0, -1.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = self._to_pm1(y)

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history_ = []

        # Ваги об'єктів. Якщо balanced=True, кожен клас вносить однаковий внесок у втрату,
        # навіть якщо днів "вгору" і "вниз" різна кількість. Інакше на зашумлених даних SVM
        # часто просто завжди передбачає найчастіший клас.
        if self.balanced:
            sample_w = np.where(y_ > 0, n_samples / (2 * np.sum(y_ > 0)), n_samples / (2 * np.sum(y_ < 0)))
        else:
            sample_w = np.ones(n_samples)

        for _ in range(self.n_iters):
            margins = y_ * (X @ self.w + self.b)   # y_i * (w.x_i + b) для всіх i одразу
            violated = margins < 1                 # точки, що порушують відступ (дають ненульовий hinge)

            # Значення функції втрат (для контролю, що спуск працює)
            hinge = (sample_w * np.maximum(0.0, 1.0 - margins)).mean()
            self.loss_history_.append(0.5 * self.lambda_param * (self.w @ self.w) + hinge)

            # Градієнти
            sw = sample_w[violated]
            grad_w = self.lambda_param * self.w - ((sw * y_[violated])[:, None] * X[violated]).sum(axis=0) / n_samples
            grad_b = -(sw * y_[violated]).sum() / n_samples

            # Крок проти градієнта
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

        return self

    def decision_function(self, X):
        """Бал w.x + b: знак - клас, модуль - наскільки далеко точка від межі. Саме його використаємо для ROC."""
        return X @ self.w + self.b

    def predict(self, X):
        """Передбачені мітки: +1 (вгору) або -1 (вниз)."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
