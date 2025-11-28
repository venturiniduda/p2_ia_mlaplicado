# -*- coding: utf-8 -*-
"""
Projeto Final - Machine Learning Aplicado
Regressão para prever DIFDATA (diferença em dias entre data de nascimento
e data de recebimento original da DN) usando o dataset SINASC_2025.

Este script cobre:
- EDA
- Pré-processamento
- Modelagem (3 modelos)
- Otimização de hiperparâmetros com GridSearchCV (5 folds)
- Avaliação no conjunto de teste
"""

# ==============================
# 1. IMPORTAÇÕES
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Para warnings mais limpos
import warnings
warnings.filterwarnings("ignore")


# ==============================
# 2. CONFIGURAÇÕES GERAIS
# ==============================

# Caminho do CSV (ajuste se necessário)
DATA_PATH = "dados/SINASC_2025.csv"

# Separador do arquivo
SEP = ";"

# Opcional: limitar quantidade de linhas para teste / desenvolvimento
# Coloque None para usar o dataset inteiro
N_ROWS = None  # ex.: 200000 para reduzir

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ==============================
# 3. CARREGAMENTO DO DATASET
# ==============================

print("Carregando dataset...")
df = pd.read_csv(DATA_PATH, sep=SEP, nrows=N_ROWS)
print(f"Dataset carregado com formato: {df.shape}")
print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())


# ==============================
# 4. ANÁLISE EXPLORATÓRIA (EDA)
# ==============================

# 4.1 Estatísticas descritivas básicas
print("\nEstatísticas descritivas (numéricas):")
print(df.describe().T)

# Verificando a variável alvo DIFDATA
if "DIFDATA" not in df.columns:
    raise ValueError("A coluna 'DIFDATA' não foi encontrada no dataset!")

print("\nDescrição da variável alvo DIFDATA:")
print(df["DIFDATA"].describe())

# 4.2 Distribuição de DIFDATA (histograma e boxplot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["DIFDATA"].dropna(), bins=50, kde=True)
plt.title("Distribuição de DIFDATA (histograma)")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["DIFDATA"].dropna())
plt.title("Boxplot de DIFDATA")

plt.tight_layout()
plt.show()

# 4.3 Checagem de valores faltantes
print("\nPercentual de valores faltantes por coluna:")
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print(missing_pct)

# 4.4 Exemplo de análise de algumas features numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# vamos pegar algumas colunas numéricas de interesse, se existirem
colunas_exemplo_num = [c for c in ["IDADEMAE", "PESO", "IDADEPAI", "SEMAGESTAC"] if c in numeric_cols]

for col in colunas_exemplo_num:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=40, kde=True)
    plt.title(f"Distribuição de {col}")
    plt.show()

# 4.5 Matriz de correlação (para um subconjunto de colunas numéricas)
# Para não ficar absurdo, vamos limitar a umas 15 variáveis numéricas
num_sample_cols = numeric_cols[:15]

plt.figure(figsize=(12, 10))
corr = df[num_sample_cols].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de correlação (amostra de variáveis numéricas)")
plt.show()

# 4.6 Detecção simples de outliers em DIFDATA
Q1 = df["DIFDATA"].quantile(0.25)
Q3 = df["DIFDATA"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df["DIFDATA"] < limite_inferior) | (df["DIFDATA"] > limite_superior)]
print(f"\nTotal de outliers em DIFDATA (IQR rule): {len(outliers)}")


# ==============================
# 5. PRÉ-PROCESSAMENTO
# ==============================

# 5.1 Separar X e y
TARGET_COL = "DIFDATA"
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# 5.2 Remover colunas de data brutas (opcional, para evitar alta cardinalidade e vazamento direto)
date_cols = [col for col in X.columns if col.startswith("DT")]
print("\nColunas de data detectadas e removidas do modelo:", date_cols)
X = X.drop(columns=date_cols)

# 5.3 Identificar tipos de variáveis
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nTotal de features numéricas:", len(numeric_features))
print("Total de features categóricas:", len(categorical_features))

# 5.4 Construir transformações para numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==============================
# 6. DIVISÃO TREINO / TESTE
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTamanho do treino: {X_train.shape}, teste: {X_test.shape}")


# ==============================
# 7. DEFINIÇÃO DOS MODELOS E GRIDSEARCH
# ==============================

# 3 modelos de regressão distintos
modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
}

# Pipelines: pré-processamento + regressor
pipelines = {
    nome: Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", modelo)
    ])
    for nome, modelo in modelos.items()
}

# Grades de hiperparâmetros
param_grids = {
    "LinearRegression": {
        # LinearRegression não tem muitos hiperparâmetros relevantes no sklearn
        # podemos deixar sem grid, mas para cumprir o requisito, deixamos algo simples
        # (por exemplo, normalização interna on/off, mas já escalamos antes, então é simbólico)
        "regressor__fit_intercept": [True, False]
    },
    "RandomForest": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    },
    "GradientBoosting": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
        "regressor__subsample": [1.0, 0.8]
    }
}

# Dicionário para salvar os melhores modelos e resultados
best_models = {}
resultados_grid = {}

# Loop de GridSearchCV para cada modelo
for nome_modelo, pipeline in pipelines.items():
    print(f"\n===== Treinando e otimizando modelo: {nome_modelo} =====")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[nome_modelo],
        cv=5,                        # 5 folds conforme requisito
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print(f"Melhores hiperparâmetros para {nome_modelo}:")
    print(grid.best_params_)

    best_models[nome_modelo] = grid.best_estimator_
    resultados_grid[nome_modelo] = grid.cv_results_


# ==============================
# 8. AVALIAÇÃO NO CONJUNTO DE TESTE
# ==============================

def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


tabela_resultados = []

for nome_modelo, modelo in best_models.items():
    print(f"\n===== Avaliando modelo {nome_modelo} no conjunto de teste =====")
    y_pred = modelo.predict(X_test)

    mae, mse, rmse, r2 = calcular_metricas(y_test, y_pred)
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    tabela_resultados.append({
        "Modelo": nome_modelo,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Converter tabela de resultados em DataFrame para visualização
df_resultados = pd.DataFrame(tabela_resultados).sort_values(by="MAE")
print("\nComparação de desempenho dos modelos (ordenado por MAE):")
print(df_resultados)


# ==============================
# 9. GRÁFICOS DE AVALIAÇÃO DO MELHOR MODELO
# ==============================

# Escolher o melhor modelo pela métrica MAE (menor é melhor)
melhor_linha = df_resultados.sort_values(by="MAE").iloc[0]
melhor_modelo_nome = melhor_linha["Modelo"]
melhor_modelo = best_models[melhor_modelo_nome]

print(f"\nMelhor modelo selecionado: {melhor_modelo_nome}")

y_pred_best = melhor_modelo.predict(X_test)
residuos = y_test - y_pred_best

# 9.1 Gráfico de dispersão: valores reais vs previstos
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.3)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")  # linha 45°
plt.xlabel("DIFDATA real")
plt.ylabel("DIFDATA previsto")
plt.title(f"Valores reais vs previstos - {melhor_modelo_nome}")
plt.tight_layout()
plt.show()

# 9.2 Distribuição dos erros (resíduos)
plt.figure(figsize=(6, 4))
sns.histplot(residuos, bins=50, kde=True)
plt.xlabel("Erro (y_real - y_previsto)")
plt.title(f"Distribuição dos resíduos - {melhor_modelo_nome}")
plt.tight_layout()
plt.show()

print("\nScript finalizado.")
