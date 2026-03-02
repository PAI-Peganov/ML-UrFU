import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def plot_linear_regression_training(n_iterations=50, lr=0.1, n_points=100):
    """
    Генерирует данные, обучает линейную регрессию и возвращает анимацию процесса.
    """
    # 1. Генерация данных
    np.random.seed(42)
    X = 2 * np.random.rand(n_points, 1)
    y = 4 + 3 * X + np.random.randn(n_points, 1)

    # Подготовка матриц (X_b содержит столбец единиц для bias)
    X_b = np.c_[np.ones((n_points, 1)), X]
    theta = np.random.randn(2, 1)

    # 2. Градиентный спуск с сохранением истории
    history_theta = []
    for _ in range(n_iterations):
        gradients = 2/n_points * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr * gradients
        history_theta.append(theta.copy())

    # 3. Настройка графики
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
    line, = ax.plot([], [], color='red', lw=3, label='Модель')

    # Текстовое поле для параметров
    text_params = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Обучение: Градиентный спуск (lr={lr})')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        text_params.set_text('')
        return line, text_params

    def animate(i):
        curr_theta = history_theta[i]

        # Точки для отрисовки прямой
        x_range = np.array([[0], [2]])
        x_range_b = np.c_[np.ones((2, 1)), x_range]
        y_predict = x_range_b.dot(curr_theta)

        line.set_data(x_range.flatten(), y_predict.flatten())

        # Обновление текста: theta[0] - intercept, theta[1] - coef
        text_params.set_text(
            f'Итерация: {i+1}\n'
            f'Intercept: {curr_theta[0][0]:.4f}\n'
            f'Coef: {curr_theta[1][0]:.4f}'
        )
        return line, text_params

    ani = FuncAnimation(fig, animate, frames=len(history_theta),
                        init_func=init, blit=True, interval=150)

    plt.close(fig) # Предотвращаем появление лишнего статического графика
    return HTML(ani.to_jshtml())


def plot_sgd_training(n_epochs=50, lr=0.1, n_points=100, batch_size=1):
    """
    Стохастический градиентный спуск (SGD)
    """
    # 1. Генерация данных
    np.random.seed(42)
    X = 2 * np.random.rand(n_points, 1)
    y = 4 + 3 * X + np.random.randn(n_points, 1)

    # Подготовка матриц
    X_b = np.c_[np.ones((n_points, 1)), X]
    theta = np.random.randn(2, 1)

    # 2. SGD с сохранением истории
    history_theta = []
    n_samples = n_points

    for epoch in range(n_epochs):
        # Перемешиваем данные в начале каждой эпохи
        indices = np.random.permutation(n_samples)
        X_b_shuffled = X_b[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            # Берем мини-батч
            X_batch = X_b_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Градиент на батче
            gradients = 2/len(X_batch) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta = theta - lr * gradients

        history_theta.append(theta.copy())

    # 3. Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
    line, = ax.plot([], [], color='red', lw=3, label='Модель')

    text_params = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'SGD: lr={lr}, batch_size={batch_size}')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        text_params.set_text('')
        return line, text_params

    def animate(i):
        curr_theta = history_theta[i]
        x_range = np.array([[0], [2]])
        x_range_b = np.c_[np.ones((2, 1)), x_range]
        y_predict = x_range_b.dot(curr_theta)

        line.set_data(x_range.flatten(), y_predict.flatten())
        text_params.set_text(
            f'Эпоха: {i+1}\n'
            f'Intercept: {curr_theta[0][0]:.4f}\n'
            f'Coef: {curr_theta[1][0]:.4f}'
        )
        return line, text_params

    ani = FuncAnimation(fig, animate, frames=len(history_theta),
                        init_func=init, blit=True, interval=200)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_momentum_training(n_epochs=50, lr=0.1, momentum=0.9, n_points=100):
    """
    SGD с моментумом (Momentum)
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n_points, 1)
    y = 4 + 3 * X + np.random.randn(n_points, 1)

    X_b = np.c_[np.ones((n_points, 1)), X]
    theta = np.random.randn(2, 1)
    velocity = np.zeros_like(theta)  # Накопленная скорость

    history_theta = []

    for epoch in range(n_epochs):
        # Перемешиваем данные
        indices = np.random.permutation(n_points)
        X_b_shuffled = X_b[indices]
        y_shuffled = y[indices]

        for i in range(n_points):
            X_i = X_b_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]

            # Градиент на одном примере
            gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)

            # Обновление с моментумом
            velocity = momentum * velocity - lr * gradients
            theta = theta + velocity

        history_theta.append(theta.copy())

    # Визуализация (аналогично предыдущей функции)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
    line, = ax.plot([], [], color='red', lw=3, label='Модель')

    text_params = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Momentum: lr={lr}, momentum={momentum}')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        text_params.set_text('')
        return line, text_params

    def animate(i):
        curr_theta = history_theta[i]
        x_range = np.array([[0], [2]])
        x_range_b = np.c_[np.ones((2, 1)), x_range]
        y_predict = x_range_b.dot(curr_theta)

        line.set_data(x_range.flatten(), y_predict.flatten())
        text_params.set_text(
            f'Эпоха: {i+1}\n'
            f'Intercept: {curr_theta[0][0]:.4f}\n'
            f'Coef: {curr_theta[1][0]:.4f}'
        )
        return line, text_params

    ani = FuncAnimation(fig, animate, frames=len(history_theta),
                        init_func=init, blit=True, interval=200)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_nesterov_training(n_epochs=50, lr=0.1, momentum=0.9, n_points=100):
    """
    Nesterov Accelerated Gradient (NAG)
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n_points, 1)
    y = 4 + 3 * X + np.random.randn(n_points, 1)

    X_b = np.c_[np.ones((n_points, 1)), X]
    theta = np.random.randn(2, 1)
    velocity = np.zeros_like(theta)

    history_theta = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_points)
        X_b_shuffled = X_b[indices]
        y_shuffled = y[indices]

        for i in range(n_points):
            X_i = X_b_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]

            # Nesterov: сначала делаем шаг в направлении накопленной скорости
            theta_lookahead = theta + momentum * velocity

            # Градиент в "заглядывающей вперед" точке
            gradients = 2 * X_i.T.dot(X_i.dot(theta_lookahead) - y_i)

            # Обновление
            velocity = momentum * velocity - lr * gradients
            theta = theta + velocity

        history_theta.append(theta.copy())

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
    line, = ax.plot([], [], color='red', lw=3, label='Модель')

    text_params = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Nesterov: lr={lr}, momentum={momentum}')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        text_params.set_text('')
        return line, text_params

    def animate(i):
        curr_theta = history_theta[i]
        x_range = np.array([[0], [2]])
        x_range_b = np.c_[np.ones((2, 1)), x_range]
        y_predict = x_range_b.dot(curr_theta)

        line.set_data(x_range.flatten(), y_predict.flatten())
        text_params.set_text(
            f'Эпоха: {i+1}\n'
            f'Intercept: {curr_theta[0][0]:.4f}\n'
            f'Coef: {curr_theta[1][0]:.4f}'
        )
        return line, text_params

    ani = FuncAnimation(fig, animate, frames=len(history_theta),
                        init_func=init, blit=True, interval=200)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_adagrad_training(n_epochs=50, lr=0.5, n_points=100, epsilon=1e-8):
    """
    AdaGrad (Adaptive Gradient)
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n_points, 1)
    y = 4 + 3 * X + np.random.randn(n_points, 1)

    X_b = np.c_[np.ones((n_points, 1)), X]
    theta = np.random.randn(2, 1)
    cache = np.zeros_like(theta)  # Накопление квадратов градиентов

    history_theta = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_points)
        X_b_shuffled = X_b[indices]
        y_shuffled = y[indices]

        for i in range(n_points):
            X_i = X_b_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]

            # Градиент
            gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)

            # AdaGrad: накапливаем квадраты градиентов
            cache += gradients ** 2

            # Адаптивное обновление (разные learning rates для разных параметров)
            theta = theta - lr * gradients / (np.sqrt(cache) + epsilon)

        history_theta.append(theta.copy())

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
    line, = ax.plot([], [], color='red', lw=3, label='Модель')

    text_params = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'AdaGrad: lr={lr}')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        text_params.set_text('')
        return line, text_params

    def animate(i):
        curr_theta = history_theta[i]
        x_range = np.array([[0], [2]])
        x_range_b = np.c_[np.ones((2, 1)), x_range]
        y_predict = x_range_b.dot(curr_theta)

        line.set_data(x_range.flatten(), y_predict.flatten())
        text_params.set_text(
            f'Эпоха: {i+1}\n'
            f'Intercept: {curr_theta[0][0]:.4f}\n'
            f'Coef: {curr_theta[1][0]:.4f}'
        )
        return line, text_params

    ani = FuncAnimation(fig, animate, frames=len(history_theta),
                        init_func=init, blit=True, interval=200)
    plt.close(fig)
    return HTML(ani.to_jshtml())
