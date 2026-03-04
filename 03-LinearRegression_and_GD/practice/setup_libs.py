import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sns
from matplotlib import patches as p
from matplotlib import pyplot as plt
from IPython.display import Image
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import BaggingRegressor  # Вот наш бэггинг регрессор
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
# отключим всякие предупреждения Anaconda
import warnings
import numpy as np
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6


def get_grid(data, border=1., step=.005):  # получаем все точки плоскости
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step),
                       np.arange(y_min, y_max, step))


def plot_model(X_train, y_train, clf, title=None, proba=False, produce_data_point=lambda x, y: (x, y)):
    xx, yy = get_grid(X_train)  # получаем все точки плоскости
    plt.figure(figsize=(7, 7))
    # предсказываем значения для каждой точки плоскости

    data_to_predict = np.array([produce_data_point(x, y)
                               for x, y in zip(xx.ravel(), yy.ravel())])
    if proba:  # нужно ли предсказывать вероятности
        predicted = clf.predict_proba(data_to_predict)[:, 1].reshape(xx.shape)
    else:
        predicted = clf.predict(data_to_predict).reshape(xx.shape)

    # Отрисовка плоскости
    ax = plt.gca()
    ax.pcolormesh(xx, yy, predicted, cmap='spring')

    # Отрисовка точек

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
               s=100, cmap='spring', edgecolors='b')
    colors = ['purple', 'yellow', 'orange']
    patches = []
    for yi in np.unique(y_train):
        patches.append(
            p.Patch(color=colors[int(yi)], label='$y_{pred}=$'+str(int(yi))))
    ax.legend(handles=patches)
    plt.title(title)
    return clf


def plot_radial_3d_decision_boundary(model, X_scaled, y, title="3D Decision Boundary"):
    """
    Визуализирует разделяющую плоскость логистической регрессии в 3D.
    X_scaled должен содержать 3 признака: [x1, x2, r]
    """
    # 1. Параметры сетки
    mesh_size = 40
    margin = 0.5

    x_min, x_max = X_scaled[:, 0].min() - margin, X_scaled[:, 0].max() + margin
    y_min, y_max = X_scaled[:, 1].min() - margin, X_scaled[:, 1].max() + margin

    x_range = np.linspace(x_min, x_max, mesh_size)
    y_range = np.linspace(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(x_range, y_range)

    # 2. Уравнение плоскости: w1*x + w2*y + w3*r + b = 0  =>  r = -(w1*x + w2*y + b) / w3
    w = model.coef_[0]
    b = model.intercept_[0]

    # Чтобы избежать деления на ноль, если w3 вдруг 0
    zz = -(w[0] * xx + w[1] * yy + b) / (w[2] + 1e-9)

    # 3. Создание фигуры
    fig = go.Figure()

    # Точки данных
    fig.add_trace(go.Scatter3d(
        x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
        mode='markers',
        marker=dict(size=3, color=y, colorscale='Portland', opacity=0.7),
        name='Данные'
    ))

    # Разделяющая плоскость
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='Greys', opacity=0.5, showscale=False,
        name='Разделяющая плоскость'
    ))

    # Настройка осей и вида
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X1 (scaled)',
            yaxis_title='X2 (scaled)',
            zaxis_title='R (radial feature)',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

# Пример вызова:
# plot_radial_3d_decision_boundary(lr_new, X_new_test_scaled, y_test)
