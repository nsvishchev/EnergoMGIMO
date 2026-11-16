import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor

# ============================================
# 2. Параметры
# ============================================

CONTAMINATION = 0.02      # ожидаемая доля аномалий
FIT_SAMPLE_SIZE = 10000   # размер общей обучающей выборки (поставьте большое число, чтобы учить на всём train)
SEED = 42

# ============================================
# 3. Загрузка данных ETTm1
# ============================================

url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"

df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["date"])

features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
data = df[features].copy()

print("Количество пропусков:")
print(data.isnull().sum())

# Разделение по времени: первые 70% в обучение, остальное в тест
split = int(len(data) * 0.7)

train_raw = data.iloc[:split].copy()
test_raw = data.iloc[split:].copy()

# Пропуски заполняем медианами обучающей части
medians = train_raw.median()
train_raw = train_raw.fillna(medians)
test_raw = test_raw.fillna(medians)

# ============================================
# 4. Нормализация и единая обучающая выборка
# ============================================

scaler = StandardScaler()
X_train = scaler.fit_transform(train_raw)
X_test = scaler.transform(test_raw)

# Контроль дрейфа: средние тестовых признаков в единицах стандартного
# отклонения train. Значения заметно больше 0.5 по модулю говорят о сдвиге режима.
print("\nСдвиг средних test относительно train (в σ):")
print(pd.Series(X_test.mean(axis=0), index=features).round(2).to_string())

# Общая обучающая выборка, которую теперь реально используют ВСЕ три метода
rng = np.random.default_rng(SEED)
sample_size = min(FIT_SAMPLE_SIZE, len(X_train))
sample_indices = rng.choice(len(X_train), size=sample_size, replace=False)
X_fit = X_train[sample_indices]

# ============================================
# 5. Isolation Forest
# ============================================

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION,
    random_state=SEED
)

start = time.perf_counter()
isolation_forest.fit(X_fit)
if_pred = isolation_forest.predict(X_test)
if_score = -isolation_forest.score_samples(X_test)   # больше = аномальнее
if_time = time.perf_counter() - start

if_anomalies = if_pred == -1

# ============================================
# 6. One-Class SVM
# ============================================

one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=CONTAMINATION
)

start = time.perf_counter()
one_class_svm.fit(X_fit)
svm_pred = one_class_svm.predict(X_test)
svm_score = -one_class_svm.decision_function(X_test)  # больше = аномальнее
svm_time = time.perf_counter() - start

svm_anomalies = svm_pred == -1

# ============================================
# 7. Autoencoder
# ============================================

autoencoder = MLPRegressor(
    hidden_layer_sizes=(4, 2, 4),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=SEED
)

# Время считаем так же, как у остальных: обучение + оценка теста
start = time.perf_counter()
autoencoder.fit(X_fit, X_fit)
X_reconstructed = autoencoder.predict(X_test)
ae_score = np.mean((X_test - X_reconstructed) ** 2, axis=1)   # ошибка восстановления
ae_time = time.perf_counter() - start

# Порог по обучающей выборке (в замер времени не входит, у других методов
# калибровка порога делается внутри fit)
train_error = np.mean((X_fit - autoencoder.predict(X_fit)) ** 2, axis=1)
threshold = np.percentile(train_error, 100 * (1 - CONTAMINATION))

ae_anomalies = ae_score > threshold

# ============================================
# 8. Сравнение результатов
# ============================================

names = ["Isolation Forest", "One-Class SVM", "Autoencoder"]
native_masks = {
    "Isolation Forest": if_anomalies,
    "One-Class SVM": svm_anomalies,
    "Autoencoder": ae_anomalies,
}
scores = {
    "Isolation Forest": if_score,
    "One-Class SVM": svm_score,
    "Autoencoder": ae_score,
}
times = {
    "Isolation Forest": if_time,
    "One-Class SVM": svm_time,
    "Autoencoder": ae_time,
}

results = pd.DataFrame({
    "Метод": names,
    "Аномалии": [native_masks[n].sum() for n in names],
    "Доля аномалий, %": [native_masks[n].mean() * 100 for n in names],
    "Время работы, сек": [times[n] for n in names],
})

print(f"\nСравнение алгоритмов (собственные пороги, ожидали {CONTAMINATION * 100:.0f}%):")
print(results.round(3).to_string(index=False))


def top_mask(score, q=1 - CONTAMINATION):
    """Отмечает долю q·100% самых аномальных точек ТЕСТА."""
    return score >= np.quantile(score, q)


# Режим 2: одинаковая доля срабатываний в тесте у всех методов
top_masks = {n: top_mask(scores[n]) for n in names}


def overlap_report(masks, title):
    print(f"\n{title}")

    # Попарные пересечения (на диагонали число срабатываний метода)
    inter = pd.DataFrame(index=names, columns=names, dtype=int)
    jacc = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            i = (masks[a] & masks[b]).sum()
            u = (masks[a] | masks[b]).sum()
            inter.loc[a, b] = i
            jacc.loc[a, b] = i / u if u else np.nan

    print("Число общих срабатываний:")
    print(inter.to_string())
    print("Индекс Жаккара:")
    print(jacc.round(2).to_string())

    all_three = np.logical_and.reduce([masks[n] for n in names])
    print("Обнаружено всеми тремя методами:", all_three.sum())


overlap_report(native_masks, "Пересечения: собственные пороги методов")
overlap_report(top_masks, f"Пересечения: у всех топ {CONTAMINATION * 100:.0f}% теста")

# ============================================
# 9. Визуализация
# ============================================

dates = df["date"].iloc[split:].values
ot = test_raw["OT"].values

style = {
    "Isolation Forest": ("orange", "o"),
    "One-Class SVM": ("green", "x"),
    "Autoencoder": ("red", "^"),
}


def plot_methods(masks, title):
    """Три панели (по одной на метод) + панель моментов срабатывания."""
    all_three = np.logical_and.reduce([masks[n] for n in names])

    fig, axes = plt.subplots(
        4, 1,
        figsize=(17, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 1.6]}
    )

    y_min, y_max = ot.min(), ot.max()
    margin = 0.05 * (y_max - y_min)

    for ax, name in zip(axes[:3], names):
        color, marker = style[name]
        mask = masks[name]
        ax.plot(dates, ot, color="steelblue", linewidth=0.8,
                label="Температура масла OT")
        ax.scatter(dates[mask], ot[mask], color=color, s=25, marker=marker,
                   label=f"{name}: {mask.sum()} ({mask.mean() * 100:.1f}%)")
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_ylabel("OT")
        ax.set_title(name, fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)

    ax = axes[3]
    rows = [(n, masks[n], style[n][0]) for n in names]
    rows.append(("Все три метода", all_three, "black"))

    for i, (name, mask, color) in enumerate(rows):
        ax.scatter(dates[mask], np.full(mask.sum(), i),
                   color=color, marker="|", s=250)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("Моменты срабатывания по методам", fontsize=12)
    ax.set_xlabel("Дата")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle(title, fontsize=14)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


plot_methods(
    native_masks,
    "Методы обнаружения аномалий ETTm1: собственные пороги"
)
plot_methods(
    top_masks,
    f"Методы обнаружения аномалий ETTm1: одинаковая доля, топ {CONTAMINATION * 100:.0f}% теста"
)

# Ошибка восстановления автоэнкодера и порог
fig, ax = plt.subplots(figsize=(17, 4.5))
ax.plot(dates, ae_score, color="gray", linewidth=0.7,
        label="Ошибка восстановления (test)")
ax.axhline(threshold, color="red", linestyle="--",
           label=f"Порог (перцентиль {100 * (1 - CONTAMINATION):.0f} по train)")
ax.set_yscale("log")
ax.set_ylabel("MSE (лог. шкала)")
ax.set_xlabel("Дата")
ax.set_title("Autoencoder: ошибка восстановления и порог", fontsize=13)
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
fig.autofmt_xdate(rotation=30)
fig.tight_layout()

plt.show()