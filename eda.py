Скрипт:
1. Загружает данные о транзакциях и исторических курсах валют.
2. Конвертирует суммы транзакций в USD.
3. Выполняет базовый анализ данных (статистика, распределения, корреляции).
4. Строит графики и сохраняет их в папку 'plots'.
5. Формирует список продуктовых и технических гипотез.
6. Сохраняет отчёт в текстовый файл 'report.txt'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== Настройки отображения =====
pd.set_option("display.max_columns", None)  # Показывать все колонки при выводе DataFrame
sns.set(style="whitegrid", palette="muted", font_scale=1.1)  # Красивый стиль графиков

# ===== 1. Загрузка данных =====
print("[INFO] Загрузка данных...")
# Загружаем транзакции
transactions = pd.read_parquet("transaction_fraud_data.parquet")
# Загружаем курсы валют
exchange_rates = pd.read_parquet("historical_currency_exchange.parquet")

# ===== 2. Предобработка =====
print("[INFO] Предобработка данных...")

# Преобразуем дату и время транзакции в формат datetime и выделим дату отдельно
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
transactions["date"] = transactions["timestamp"].dt.date

# Преобразуем дату в курсах валют и выделим её в том же формате
exchange_rates["date"] = pd.to_datetime(exchange_rates["date"]).dt.date

# Объединяем таблицы по дате, чтобы к каждой транзакции добавить курс валют
merged = transactions.merge(exchange_rates, on="date", how="left")

# Функция для перевода суммы в USD по соответствующему курсу
def convert_to_usd(row):
    rate = row.get(row["currency"], None)  # Берём колонку с названием валюты
    if pd.notnull(rate) and rate != 0:
        return row["amount"] / rate
    return None  # Если нет курса, возвращаем None

# Применяем функцию к каждой строке
merged["amount_usd"] = merged.apply(convert_to_usd, axis=1)

# Удаляем строки, где не удалось рассчитать сумму в USD
merged = merged.dropna(subset=["amount_usd"])

# ===== 3. Общая статистика =====
total_transactions = len(merged)
fraud_count = merged["is_fraud"].sum()
fraud_percent = fraud_count / total_transactions * 100
unique_customers = merged["customer_id"].nunique()
unique_vendors = merged["vendor"].nunique()

# ===== 4. Создание папки для графиков =====
os.makedirs("plots", exist_ok=True)

# ===== 5. Построение графиков =====

# 1. Распределение сумм транзакций
plt.figure(figsize=(8, 5))
sns.histplot(merged["amount_usd"], bins=50, kde=True)
plt.title("Распределение сумм транзакций (USD)")
plt.xlabel("Сумма в USD")
plt.ylabel("Количество транзакций")
plt.tight_layout()
plt.savefig("plots/amount_distribution.png")
plt.close()

# 2. Топ-10 стран по количеству транзакций
top_countries = merged["country"].value_counts().head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Топ-10 стран по количеству транзакций")
plt.xlabel("Количество")
plt.ylabel("Страна")
plt.tight_layout()
plt.savefig("plots/top_countries.png")
plt.close()

# 3. Доля мошенничества по категориям вендоров
fraud_by_vendor = merged.groupby("vendor_category")["is_fraud"].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=fraud_by_vendor.values * 100, y=fraud_by_vendor.index)
plt.title("Доля мошенничества по категориям вендоров (%)")
plt.xlabel("Доля мошеннических операций (%)")
plt.ylabel("Категория")
plt.tight_layout()
plt.savefig("plots/fraud_by_vendor.png")
plt.close()

# 4. Транзакции по часам суток
merged["hour"] = merged["timestamp"].dt.hour
plt.figure(figsize=(8, 5))
sns.countplot(x="hour", data=merged)
plt.title("Распределение транзакций по часам суток")
plt.xlabel("Час")
plt.ylabel("Количество транзакций")
plt.tight_layout()
plt.savefig("plots/transactions_by_hour.png")
plt.close()

# 5. Корреляция числовых признаков
numeric_cols = merged.select_dtypes(include=["float64", "int64"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols, annot=False, cmap="coolwarm")
plt.title("Корреляция числовых признаков")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ===== 6. Генерация гипотез =====
product_hypotheses = [
    "Увеличить лимиты проверок для транзакций в ночное время, когда уровень мошенничества выше.",
    "Ввести дополнительные проверки для категорий с высокой долей мошенничества (например, путешествия, развлечения).",
    "Предлагать клиентам уведомления о крупных транзакциях в нестандартное время.",
    "Внедрить гео-проверку для операций за пределами страны клиента.",
    "Использовать машинное обучение для оценки риска в реальном времени."
]

technical_hypotheses = [
    "Добавить больше временных признаков (день недели, праздничный день, сезон).",
    "Интегрировать поведенческие биометрические данные для аутентификации.",
    "Ввести кластеризацию клиентов по типичному паттерну трат.",
    "Обогащать данные о вендорах из внешних источников.",
    "Разрабатывать ансамблевые модели для улучшения точности выявления мошенничества."
]

# ===== 7. Сохранение отчёта =====
with open("report.txt", "w", encoding="utf-8") as f:
    f.write("=== Общая статистика ===\n")
    f.write(f"Всего транзакций: {total_transactions}\n")
    f.write(f"Количество мошеннических: {fraud_count} ({fraud_percent:.2f}%)\n")
    f.write(f"Уникальных клиентов: {unique_customers}\n")
    f.write(f"Уникальных вендоров: {unique_vendors}\n\n")
    
    f.write("=== Продуктовые гипотезы ===\n")
    for h in product_hypotheses:
        f.write(f"- {h}\n")
    f.write("\n=== Технические гипотезы ===\n")
    for h in technical_hypotheses:
        f.write(f"- {h}\n")

print("[INFO] Анализ завершён! Графики сохранены в папку 'plots', отчёт — в 'report.txt'")
