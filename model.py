# -*- coding: utf-8 -*-
"""
Projeto Final - Machine Learning Aplicado
Regressão para prever DIFDATA (diferença em dias entre data de nascimento
e data de recebimento original da DN) usando o dataset SINASC_2025.

Este script cobre:
- EDA
- Pré-processamento (remoção de colunas, remoção de NaN, transformação num/cat)
- Modelagem (3 modelos)
- Otimização de hiperparâmetros com GridSearchCV (5 folds)
- Avaliação no conjunto de teste
- Gráficos de desempenho (dispersão, resíduos, comparação entre modelos, barras de erro)
- Matriz de confusão (após categorizar o atraso em faixas)
"""

# ==============================
# 1. IMPORTAÇÕES
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tableone import TableOne

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")


# ==============================
# 2. CONFIGURAÇÕES GERAIS
# ==============================

# Caminho do CSV (dentro da pasta "dados")
DATA_PATH = os.path.join("dados", "SINASC_2025.csv")

SEP = ";"          # separador do CSV
N_ROWS = None     
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ==============================
# 3. CARREGAMENTO DO DATASET
# ==============================

print("Carregando dataset...")
df = pd.read_csv(DATA_PATH, sep=SEP, nrows=N_ROWS)
df = df.iloc[:, 1:]
print(f"Dataset carregado com formato: {df.shape}")

print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())


# ==============================
# 3.1 REMOÇÃO DE LINHAS COM NaN/NULL
# ==============================

print("\nRemovendo linhas com valores NaN/null...")
df = df.dropna()
print(f"Dataset após remoção de NaN: {df.shape}")

print("\nPercentual de valores faltantes após dropna (deve ser 0):")
print((df.isna().mean() * 100).sort_values(ascending=False))

# ==============================
# 4. ANÁLISE EXPLORATÓRIA (EDA)
# ==============================

print("\nEstatísticas descritivas (numéricas):")
print(df.describe().T)

# Verificando a variável alvo DIFDATA
if "DIFDATA" not in df.columns:
    raise ValueError("A coluna 'DIFDATA' não foi encontrada no dataset!")

print("\nDescrição da variável alvo DIFDATA:")
print(df["DIFDATA"].describe())

# 4.1 Distribuição de DIFDATA (histograma e boxplot)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(df["DIFDATA"], bins=50, kde=True, color="#1f77b4")
plt.title("Distribuição de DIFDATA", fontsize=12, fontweight="bold")
plt.xlabel("DIFDATA (dias)")
plt.ylabel("Frequência")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["DIFDATA"], color="#ff7f0e")
plt.title("Boxplot de DIFDATA", fontsize=12, fontweight="bold")
plt.xlabel("DIFDATA (dias)")

plt.suptitle("Análise exploratória da variável alvo DIFDATA\n"
             "Mostra concentração de valores e presença de outliers.",
             fontsize=10)
plt.tight_layout()
plt.show()

# 4.2 Matriz de correlação (subconjunto de numéricas)
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
num_sample_cols = numeric_cols_all[:15]

plt.figure(figsize=(12, 10))
corr = df[num_sample_cols].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação (subconjunto de variáveis numéricas)", fontsize=14, fontweight="bold")
plt.suptitle("Ajuda a identificar relações lineares entre variáveis numéricas.\n"
             "Correlação fraca com DIFDATA indica natureza não linear do problema.",
             fontsize=10)
plt.tight_layout()
plt.show()

# 4.3 Detecção simples de outliers em DIFDATA (regra do IQR)
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

# 5.2 Remover colunas de data brutas (DT...)
date_cols = [col for col in X.columns if col.startswith("DT")]
print("\nColunas de data detectadas e removidas do modelo:", date_cols)
X = X.drop(columns=date_cols)

# 5.3 Remover colunas irrelevantes para o modelo
cols_irrelevantes = [
    "contador",
    "GESTACAO",
    "PARTO",
    "APGAR1",
    "APGAR5",
    "RACACOR",
    "PESO",
    "CODESTAB",
    "ORIGEM",
    "CODCART",
    "NUMREGCART",
    "NUMEROLOTE",
    "VERSAOSIST",
    "QTDPARTNOR",
    "QTDPARTCES",
    "SEMAGESTAC",
    "TPMETESTIM",
    "CONSPRENAT",
    "MESPRENAT",
    "TPAPRESENT",
    "STTRABPART",
    "STCESPARTO",
    "TPROBSON",
    "STDNEPIDEM",
    "STDNNOVA",
    "CODMUNCART",
    "RACACORN"
]

cols_irrelevantes_presentes = [c for c in cols_irrelevantes if c in X.columns]
print("\nColunas irrelevantes removidas do modelo:", cols_irrelevantes_presentes)
X = X.drop(columns=cols_irrelevantes_presentes)

print("\nInformações do dataset:")
print(df.info())

# 5.4 Identificar tipos de variáveis (numéricas x categóricas)
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nTotal de features numéricas:", len(numeric_features))
print("Total de features categóricas:", len(categorical_features))

# 5.4.1 Garantir que todas as variáveis categóricas sejam strings
print("\nConvertendo colunas categóricas para string para o OneHotEncoder...")
if len(categorical_features) > 0:
    X[categorical_features] = X[categorical_features].astype(str)

# 5.5 Construir transformações para numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # redundante com dropna, mas seguro
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
# 6. TABLE ONE (resumo da amostra)
# ==============================

# Selecionar colunas para incluir na Table One
columns_table1 = numeric_features + categorical_features

# Definir quais são categóricas
categorical_table1 = categorical_features

print("\nGerando Table One (resumo estatístico da amostra)...")

table1 = TableOne(
    df,
    columns=columns_table1,
    categorical=categorical_table1,
    groupby=None,
    missing=False
)

# Converter para DataFrame para exibir e salvar bonitinho
table1_df = table1.tableone.reset_index()

print("\n===== TABLE ONE – Resumo da Amostra =====\n")
print(table1_df.to_string(index=False))  # exibe como tabela alinhada no console

# Exportar para CSV já formatado
table1_df.to_csv("table_one.csv", sep=";", index=False, encoding="utf-8-sig")

print("\nTable One gerada e salva como table_one.csv")

# ==============================
# 7. DIVISÃO TREINO / TESTE
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTamanho do treino: {X_train.shape}, teste: {X_test.shape}")


# ==============================
# 8. DEFINIÇÃO DOS MODELOS E GRIDSEARCH
# ==============================

modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
}

pipelines = {
    nome: Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", modelo)
    ])
    for nome, modelo in modelos.items()
}

param_grids = {
    "LinearRegression": {
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

best_models = {}
resultados_grid = {}
cv_resumo = []   # para guardar resumo da validação cruzada

for nome_modelo, pipeline in pipelines.items():
    print(f"\n===== Treinando e otimizando modelo: {nome_modelo} =====")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[nome_modelo],
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print(f"Melhores hiperparâmetros para {nome_modelo}:")
    print(grid.best_params_)

    best_models[nome_modelo] = grid.best_estimator_
    resultados_grid[nome_modelo] = grid.cv_results_

    # Resumo da validação cruzada (para barras de erro)
    best_index = grid.best_index_
    mean_mae_cv = -grid.cv_results_["mean_test_score"][best_index]  # MAE positivo
    std_mae_cv = grid.cv_results_["std_test_score"][best_index]

    cv_resumo.append({
        "Modelo": nome_modelo,
        "MAE_CV_medio": mean_mae_cv,
        "MAE_CV_desvio": std_mae_cv
    })


# ==============================
# 9. AVALIAÇÃO NO CONJUNTO DE TESTE
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

df_resultados = pd.DataFrame(tabela_resultados).sort_values(by="MAE")
print("\nComparação de desempenho dos modelos (ordenado por MAE):")
print(df_resultados)

# Unir resultados de teste com resumo da validação cruzada
df_cv = pd.DataFrame(cv_resumo)
df_cv = df_cv.merge(df_resultados[["Modelo"]], on="Modelo", how="right")


# ==============================
# 9.1 GRÁFICOS COMPARATIVOS ENTRE MODELOS
# ==============================

# Gráfico comparativo de MAE no conjunto de teste
plt.figure(figsize=(8, 5))
plt.bar(df_resultados["Modelo"], df_resultados["MAE"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])

plt.title("Comparação do Erro Médio Absoluto (MAE) entre Modelos", fontsize=14, fontweight="bold")
plt.suptitle("MAE calculado no conjunto de teste.\nValores menores indicam melhor desempenho preditivo.",
             fontsize=10)

plt.xlabel("Modelo", fontsize=12)
plt.ylabel("MAE (dias)", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico comparativo de R² no conjunto de teste
plt.figure(figsize=(8, 5))
plt.bar(df_resultados["Modelo"], df_resultados["R2"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])

plt.title("Comparação do Coeficiente de Determinação (R²) entre Modelos", fontsize=14, fontweight="bold")
plt.suptitle("R² calculado no conjunto de teste.\nValores próximos de 1 indicam maior explicação "
             "da variabilidade de DIFDATA.", fontsize=10)

plt.xlabel("Modelo", fontsize=12)
plt.ylabel("R²", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico de barras com barras de erro (MAE na validação cruzada)
plt.figure(figsize=(8, 5))
plt.bar(
    df_cv["Modelo"],
    df_cv["MAE_CV_medio"],
    yerr=df_cv["MAE_CV_desvio"],
    capsize=5,
    color=["#1f77b4", "#ff7f0e", "#2ca02c"]
)

plt.title("Erro Médio Absoluto (MAE) na Validação Cruzada (5 folds)", fontsize=14, fontweight="bold")
plt.suptitle("Barras de erro representam o desvio padrão do MAE entre os folds.\n"
             "Quanto menores as barras, mais estável é o desempenho do modelo.",
             fontsize=10)

plt.xlabel("Modelo", fontsize=12)
plt.ylabel("MAE médio na validação cruzada (dias)", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 10. GRÁFICOS DE AVALIAÇÃO DO MELHOR MODELO
# ==============================

melhor_linha = df_resultados.sort_values(by="MAE").iloc[0]
melhor_modelo_nome = melhor_linha["Modelo"]
melhor_modelo = best_models[melhor_modelo_nome]

print(f"\nMelhor modelo selecionado: {melhor_modelo_nome}")

y_pred_best = melhor_modelo.predict(X_test)
residuos = y_test - y_pred_best

# 9.1 Gráfico de dispersão: valores reais vs previstos (melhorado)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color="#1f77b4", edgecolors="k")

min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Linha de acerto perfeito")

plt.title(f"Comparação entre Valores Reais e Previstos — {melhor_modelo_nome}",
          fontsize=14, fontweight="bold")
plt.suptitle("Cada ponto representa uma previsão do modelo.\n"
             "Pontos próximos da linha vermelha indicam previsões mais precisas.",
             fontsize=10)

plt.xlabel("DIFDATA (valor real)", fontsize=12)
plt.ylabel("DIFDATA (valor previsto)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9.2 Distribuição dos resíduos (melhorado)
plt.figure(figsize=(8, 5))
sns.histplot(residuos, bins=50, kde=True, color="#2ca02c")

plt.title(f"Distribuição dos Erros (Resíduos) — {melhor_modelo_nome}",
          fontsize=14, fontweight="bold")
plt.suptitle("Mostra como os erros do modelo estão distribuídos.\n"
             "O ideal é que estejam concentrados próximos de zero.",
             fontsize=10)

plt.xlabel("Erro (valor real - previsto)", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 11. MATRIZ DE CONFUSÃO (CATEGORIZANDO O ATRASO)
# ==============================

# Função para transformar DIFDATA contínuo em faixas de atraso
def categorizar_atraso(dias):
    if dias <= 5:
        return "Baixo (≤ 5 dias)"
    elif dias <= 20:
        return "Médio (6 a 20 dias)"
    else:
        return "Alto (> 20 dias)"

# Transformar valores reais e previstos em classes
y_test_cls = y_test.apply(categorizar_atraso)
y_pred_best_cls = pd.Series(y_pred_best, index=y_test.index).apply(categorizar_atraso)

labels = ["Baixo (≤ 5 dias)", "Médio (6 a 20 dias)", "Alto (> 20 dias)"]
cm = confusion_matrix(y_test_cls, y_pred_best_cls, labels=labels)

print("\nMatriz de confusão (atraso em faixas):")
print(cm)

print("\nRelatório de classificação por faixas de atraso:")
print(classification_report(y_test_cls, y_pred_best_cls, target_names=labels))

# Matriz de Confusão Melhorada
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(values_format="d", cmap="Blues", xticks_rotation=45)

plt.title(f"Matriz de Confusão — Classificação das Faixas de Atraso ({melhor_modelo_nome})",
          fontsize=14, fontweight="bold")
plt.suptitle("Compara as categorias de atraso reais versus previstas.\n"
             "A diagonal principal indica acertos; fora da diagonal são erros.",
             fontsize=10)

plt.xlabel("Classe Predita", fontsize=12)
plt.ylabel("Classe Real", fontsize=12)
plt.tight_layout()
plt.show()

# ==============================
# 12. EXPORTAR RESULTADOS PARA CSV
# ==============================

print("\nGerando arquivo CSV com resultados de previsão...")

resultado_csv = pd.DataFrame({
    "DIFDATA_REAL": y_test,
    "DIFDATA_PREVISTO": y_pred_best,
    "ERRO": y_test - y_pred_best,
    "CLASSE_REAL": y_test_cls,
    "CLASSE_PREVISTA": y_pred_best_cls
})

# Caminho do arquivo de saída
csv_path = "resultados_predicoes.csv"

resultado_csv.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

print(f"Arquivo CSV gerado com sucesso: {csv_path}")

print("\nScript finalizado.")
