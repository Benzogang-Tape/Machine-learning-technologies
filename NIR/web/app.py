"""
Streamlit-приложение для НИР "Прогнозирование отклика клиента на банковскую
маркетинговую кампанию" (UCI Bank Marketing).

Запуск:
    streamlit run web/app.py

Возможности:
- Подбор гиперпараметров GradientBoosting в интерфейсе — модель перестраивается
  при изменении любого слайдера.
- Ввод характеристик клиента — приложение возвращает вероятность отклика
  и решение классификатора при заданном пороге.
- Отображение метрик и confusion matrix на тестовой выборке для текущей модели.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop(columns=["duration"])
    df["pdays_999"] = (df["pdays"] == 999).astype(int)
    median_pdays = df.loc[df["pdays"] != 999, "pdays"].median()
    df["pdays"] = df["pdays"].replace(999, median_pdays)
    df["is_contacted_before"] = (df["previous"] > 0).astype(int)
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    X = df.drop(columns=["y"])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


@st.cache_resource(show_spinner="Обучение модели...")
def train_model(n_estimators: int, learning_rate: float, max_depth: int):
    df = load_data()
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("prep", build_preprocessor(df)),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    return pipe, X_test, y_test, proba_test


def hyperparameters_block() -> tuple[int, float, int, float]:
    st.sidebar.header("Гиперпараметры модели")
    n_estimators = st.sidebar.slider(
        "n_estimators (число деревьев бустинга)",
        min_value=50, max_value=500, value=200, step=50,
        help="Больше деревьев — выше качество и время обучения.",
    )
    learning_rate = st.sidebar.select_slider(
        "learning_rate",
        options=[0.01, 0.03, 0.05, 0.1, 0.2],
        value=0.1,
        help="Скорость обучения. Меньше — нужно больше деревьев.",
    )
    max_depth = st.sidebar.slider(
        "max_depth (глубина каждого дерева)",
        min_value=2, max_value=8, value=3, step=1,
    )
    st.sidebar.markdown("---")
    threshold = st.sidebar.slider(
        "Порог классификации",
        min_value=0.05, max_value=0.95, value=0.30, step=0.05,
        help="Вероятности >= порога относятся к классу 'yes'.",
    )
    return n_estimators, learning_rate, max_depth, threshold


def client_input_block(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Параметры клиента")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("age", 17, 100, 40)
        job = st.selectbox("job", sorted(df["job"].unique()))
        marital = st.selectbox("marital", sorted(df["marital"].unique()))
        education = st.selectbox("education", sorted(df["education"].unique()))
        default = st.selectbox("default", sorted(df["default"].unique()))
        housing = st.selectbox("housing", sorted(df["housing"].unique()))
        loan = st.selectbox("loan", sorted(df["loan"].unique()))

    with col2:
        contact = st.selectbox("contact", sorted(df["contact"].unique()))
        month = st.selectbox("month", sorted(df["month"].unique()))
        day_of_week = st.selectbox("day_of_week", sorted(df["day_of_week"].unique()))
        campaign = st.slider("campaign (контактов в кампании)", 1, 50, 2)
        pdays = st.slider("pdays (дней с прошлого контакта)", 0, 30, 6)
        previous = st.slider("previous (предыдущих контактов)", 0, 10, 0)
        poutcome = st.selectbox("poutcome", sorted(df["poutcome"].unique()))

    with col3:
        st.markdown("**Макроэкономические индикаторы**")
        emp_var_rate = st.slider("emp.var.rate", -3.5, 1.5, 1.1, step=0.1)
        cons_price_idx = st.slider("cons.price.idx", 92.0, 95.0, 93.99, step=0.01)
        cons_conf_idx = st.slider("cons.conf.idx", -51.0, -26.0, -36.4, step=0.1)
        euribor3m = st.slider("euribor3m", 0.5, 5.5, 4.86, step=0.01)
        nr_employed = st.slider("nr.employed", 4960.0, 5230.0, 5191.0, step=1.0)

    row = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "pdays_999": int(previous == 0),
        "is_contacted_before": int(previous > 0),
    }
    return pd.DataFrame([row])


def metrics_block(y_test, proba_test, threshold: float):
    pred = (proba_test >= threshold).astype(int)
    metrics = {
        "ROC AUC": roc_auc_score(y_test, proba_test),
        "PR AUC": average_precision_score(y_test, proba_test),
        "F1 (yes)": f1_score(y_test, pred),
        "Recall (yes)": recall_score(y_test, pred),
        "Precision (yes)": precision_score(y_test, pred, zero_division=0),
    }
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.3f}")

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(f"Confusion matrix (порог = {threshold:.2f})")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Bank Marketing: NIR", layout="wide")
    st.title("Прогнозирование отклика клиента на банковскую маркетинговую кампанию")
    st.markdown(
        "Демонстрационное приложение к НИР по дисциплине «Технологии машинного "
        "обучения». Модель — GradientBoosting. Перестраивается при изменении "
        "гиперпараметров в левой панели."
    )

    n_estimators, learning_rate, max_depth, threshold = hyperparameters_block()
    pipe, X_test, y_test, proba_test = train_model(n_estimators, learning_rate, max_depth)

    st.subheader("Качество модели на тестовой выборке")
    metrics_block(y_test, proba_test, threshold)

    st.markdown("---")
    df_full = load_data()
    sample = client_input_block(df_full)

    if st.button("Получить прогноз", type="primary"):
        proba = pipe.predict_proba(sample)[0, 1]
        verdict = "yes — клиент скорее откроет вклад" if proba >= threshold else "no — клиент скорее не откроет вклад"
        st.success(f"Вероятность отклика: **{proba:.3f}**  →  **{verdict}**")
        st.caption(f"Порог классификации: {threshold:.2f}")


if __name__ == "__main__":
    main()
