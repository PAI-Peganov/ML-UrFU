import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


np.random.seed(42)
X = np.sort(np.random.rand(20))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, len(X))
X_plot = np.linspace(0, 1, 100)


def plot_ridge(alpha):
    # Создаем полиномиальную регрессию высокой степени
    model = make_pipeline(PolynomialFeatures(degree=12), Ridge(alpha=alpha))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green',
             linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='blue', linewidth=2,
             label=f'L2 Регуляризация (alpha={alpha:.2e})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("Как L2-штраф усмиряет модель")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_lasso(alpha):
    # Создаем полиномиальную регрессию с L1 штрафом
    # Увеличим max_iter
    model = make_pipeline(PolynomialFeatures(degree=12),
                          Lasso(alpha=alpha, max_iter=100000))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green',
             linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='purple', linewidth=2,
             label=f'L1 Регуляризация (alpha={alpha:.2e})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("Lasso (L1): Зануление лишних коэффициентов")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_elastic_net(alpha, l1_ratio=0.5):
    # l1_ratio=1 это Lasso, l1_ratio=0 это Ridge
    model = make_pipeline(PolynomialFeatures(degree=12),
                          ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                     max_iter=100000))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green',
             linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='orange', linewidth=2,
             label=f'ElasticNet (alpha={alpha:.2e}, l1_ratio={l1_ratio})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("ElasticNet: Баланс между отбором и сглаживанием")
    plt.grid(True, alpha=0.3)
    plt.show()
