"""
MÓDULO DE TREINAMENTO - Breast Cancer (UCI)

Treina classificadores para prever o atributo Class
(no-recurrence-events | recurrence-events) e salva o de MELHOR ACURÁCIA.

Etapas (nesta ordem):
  1. Carregar e validar os dados  ("?" vira valor ausente; domínios conferidos com a UCI)
  2. Análise exploratória         (balanceamento, moda, associação atributo x Class)
  3. Separar treino e teste       (estratificado, semente fixa)
  4. Pré-processar dentro de um Pipeline (imputação pela moda + one-hot),
     assim nada do teste "vaza" para o treino
  5. Comparar vários modelos pela acurácia (validação cruzada aninhada, só no treino)
  6. Ajustar os hiperparâmetros do MELHOR modelo (GridSearchCV) e avaliá-lo no teste
  7. Salvar modelo, metadados, relatório e gráficos

Uso:  python treinamento.py
"""
import json
from datetime import datetime
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from scipy.stats import binomtest, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
CAMINHO_DADOS = RAIZ / "data" / "breast-cancer.csv"
PASTA_MODELO = RAIZ / "modelo"
PASTA_RESULTADOS = RAIZ / "resultados"
CAMINHO_MODELO = PASTA_MODELO / "modelo_final.joblib"
CAMINHO_METADADOS = PASTA_MODELO / "metadados.json"

ALVO = "Class"
SEMENTE = 42
TAMANHO_TESTE = 0.2

# Domínio de cada atributo (documentação da UCI). Os valores de breast-quad
# usam "_" no CSV da prova; "-" (como na UCI) também é aceito e convertido.
DOMINIOS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "menopause": ["lt40", "ge40", "premeno"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                   "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20",
                  "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
    "node-caps": ["yes", "no"],
    "deg-malig": [1, 2, 3],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"],
}
ATRIBUTOS = list(DOMINIOS)                    # mesma ordem do CSV
OPCIONAIS = ["node-caps", "breast-quad"]      # únicos com valores ausentes na base

CLASSES = {
    "no-recurrence-events": "sem indicativo de recorrência do câncer de mama",
    "recurrence-events": "indicativo de RECORRÊNCIA do câncer de mama",
}


def _para_json(obj):
    """Converte escalares do numpy (int64, float64...) para tipos nativos ao gravar o JSON."""
    return obj.item() if hasattr(obj, "item") else str(obj)


def normalizar_valor(atributo, valor):
    """Padroniza um valor informado (espaços, maiúsculas, '-' em breast-quad, int em deg-malig)."""
    if pd.isna(valor):
        return None
    if atributo == "deg-malig":
        numero = float(valor)
        if numero != int(numero):
            raise ValueError(f"deg-malig deve ser inteiro, recebido {valor!r}")
        return int(numero)
    valor = str(valor).strip().lower()
    if atributo == "breast-quad":
        valor = valor.replace("-", "_")
    return valor


# ----------------------------------------------------------------------------
# 1) Dados
# ----------------------------------------------------------------------------
def carregar_dados(caminho=CAMINHO_DADOS):
    """Lê o CSV, trata '?' como ausente e confere se os valores estão nos domínios da UCI."""
    df = pd.read_csv(caminho, na_values=["?"])
    faltando = [c for c in ATRIBUTOS + [ALVO] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    for col in ATRIBUTOS:
        df[col] = df[col].map(lambda v, c=col: normalizar_valor(c, v))
        invalidos = set(df[col].dropna()) - set(DOMINIOS[col])
        if invalidos:
            raise ValueError(f"Valores fora do domínio em '{col}': {sorted(invalidos, key=str)}")
    return df[ATRIBUTOS + [ALVO]]


# ----------------------------------------------------------------------------
# 2) Análise exploratória
# ----------------------------------------------------------------------------
def analise_exploratoria(df, mostrar):
    mostrar("1) DADOS")
    mostrar(f"   Registros: {len(df)} | Atributos: {len(ATRIBUTOS)} | Alvo: {ALVO}")

    ausentes = df[ATRIBUTOS].isna().sum()
    ausentes = ausentes[ausentes > 0]
    texto = ", ".join(f"{c} = {n} ({n / len(df):.1%})" for c, n in ausentes.items())
    mostrar(f'   Valores ausentes ("?" no CSV): {texto or "nenhum"}')
    mostrar("   -> imputados pela moda DENTRO do pipeline (usando apenas dados de treino).")

    contagem = df[ALVO].value_counts()
    mostrar("\n   Distribuição da classe:")
    for classe, n in contagem.items():
        mostrar(f"     {classe:22s} {n:4d}  ({n / len(df):.1%})")
    baseline = contagem.max() / len(df)
    mostrar(f"   Baseline (sempre prever a classe majoritária): {baseline:.1%} de acurácia")
    mostrar("   Base desbalanceada: class_weight='balanced' entra como hiperparâmetro; "
            "a escolha final é por acurácia.")

    mostrar("\n2) MODA E ASSOCIAÇÃO COM A CLASSE (teste qui-quadrado, alfa = 5%)")
    linhas = []
    for col in ATRIBUTOS:
        moda = df[col].mode()[0]
        tabela = pd.crosstab(df[col], df[ALVO])
        qui2, p, _, _ = chi2_contingency(tabela)
        cramer = sqrt(qui2 / (tabela.values.sum() * (min(tabela.shape) - 1)))
        linhas.append({"atributo": col, "moda": moda,
                       "freq. da moda": f"{(df[col] == moda).mean():.1%}",
                       "qui2": round(qui2, 2), "p-valor": round(p, 4),
                       "Cramér V": round(cramer, 2),
                       "associado?": "sim" if p < 0.05 else "não"})
    mostrar(pd.DataFrame(linhas).to_string(index=False))
    mostrar("   (categorias raras têm frequência esperada baixa; leia o p-valor como aproximação)")
    return baseline


# ----------------------------------------------------------------------------
# 4) Pré-processamento e modelos
# ----------------------------------------------------------------------------
def montar_pipeline(classificador):
    """Pipeline = imputação pela moda + one-hot (todos os atributos são categóricos) + modelo.

    Não há variável quantitativa contínua: deg-malig (1 a 3) é um grau, tratado
    como categoria. Dados categóricos não são normalizados, e one-hot já deixa
    todas as colunas na mesma escala (0/1).
    """
    pre = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([("pre", ColumnTransformer([("categorias", pre, ATRIBUTOS)])),
                     ("modelo", classificador)])


def definir_modelos():
    """nome -> (classificador, grade de hiperparâmetros)."""
    peso = [None, "balanced"]
    return {
        "Baseline (classe majoritária)": (DummyClassifier(strategy="most_frequent"), {}),
        "Regressão Logística": (
            LogisticRegression(max_iter=5000),
            {"modelo__C": [0.01, 0.1, 1, 10], "modelo__class_weight": peso}),
        "KNN": (
            KNeighborsClassifier(),
            {"modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
             "modelo__weights": ["uniform", "distance"]}),
        "Árvore de Decisão": (
            DecisionTreeClassifier(random_state=SEMENTE),
            {"modelo__max_depth": [2, 3, 4, 5, None],
             "modelo__min_samples_leaf": [1, 5, 10],
             "modelo__class_weight": peso}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=SEMENTE),
            {"modelo__max_depth": [3, 5, None],
             "modelo__min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEMENTE),
            {"modelo__n_estimators": [50, 100],
             "modelo__learning_rate": [0.05, 0.1],
             "modelo__max_depth": [1, 2, 3]}),
        "SVM": (
            SVC(),
            {"modelo__C": [0.1, 1, 10], "modelo__kernel": ["linear", "rbf"]}),
        "Naive Bayes": (
            BernoulliNB(),
            {"modelo__alpha": [0.1, 0.5, 1.0, 2.0], "modelo__fit_prior": [True, False]}),
    }


def comparar_modelos(X_treino, y_treino, mostrar):
    """Compara os modelos por validação cruzada ANINHADA (só com os dados de treino).

    Laço externo: mede a acurácia. Laço interno (GridSearchCV): escolhe os hiperparâmetros
    usando apenas o treino de cada divisão externa. Assim a nota de cada modelo não é
    inflada pelo ajuste de hiperparâmetros e o conjunto de teste continua intocado.
    """
    externo = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)
    interno = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    resultados = []
    for nome, (clf, grade) in definir_modelos().items():
        busca = GridSearchCV(montar_pipeline(clf), grade, cv=interno, scoring="accuracy")
        notas = cross_val_score(busca, X_treino, y_treino, cv=externo, scoring="accuracy", n_jobs=-1)
        resultados.append({"modelo": nome, "acuracia": notas.mean(), "desvio": notas.std()})
        mostrar(f"   {nome:30s} {notas.mean():.1%} +/- {notas.std():.1%}")
    return pd.DataFrame(resultados)


def ajustar_melhor(nome, X_treino, y_treino):
    """Ajusta os hiperparâmetros do modelo escolhido (GridSearchCV) e treina com todo o treino."""
    classificador, grade = definir_modelos()[nome]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)
    busca = GridSearchCV(montar_pipeline(classificador), grade, cv=cv, scoring="accuracy", n_jobs=-1)
    return busca.fit(X_treino, y_treino)


# ----------------------------------------------------------------------------
# 6) Avaliação do melhor modelo (única vez em que o teste é usado)
# ----------------------------------------------------------------------------
def avaliar_melhor(modelo, X_teste, y_teste, baseline, mostrar):
    previsto = modelo.predict(X_teste)
    acertos = int((previsto == y_teste).sum())
    ic = binomtest(acertos, len(y_teste)).proportion_ci(confidence_level=0.95, method="wilson")
    classes = list(modelo.classes_)

    mostrar(f"\n   Acurácia no teste: {acertos}/{len(y_teste)} = {acertos / len(y_teste):.1%} "
            f"(IC 95%: {ic.low:.1%} a {ic.high:.1%}; baseline {baseline:.1%})")
    mostrar("\n   Matriz de confusão (linhas = real, colunas = previsto):")
    matriz = pd.DataFrame(confusion_matrix(y_teste, previsto, labels=classes),
                          index=[f"real: {c}" for c in classes],
                          columns=[f"prev: {c}" for c in classes])
    mostrar(matriz.to_string())
    mostrar("\n   Precisão / revocação / F1 por classe:")
    relatorio = classification_report(y_teste, previsto, digits=3, zero_division=0)
    mostrar("   " + relatorio.replace("\n", "\n   ").rstrip())
    return {"acertos_teste": acertos, "n_teste": len(y_teste), "ic95": [ic.low, ic.high],
            "relatorio": classification_report(y_teste, previsto, output_dict=True, zero_division=0),
            "matriz_confusao": matriz.values.tolist(), "classes": classes}, previsto


# ----------------------------------------------------------------------------
# 7) Gráficos
# ----------------------------------------------------------------------------
def salvar_graficos(tabela, baseline, y_teste, previsto, classes):
    from matplotlib.figure import Figure  # sem pyplot: não abre janela

    ordem = tabela.sort_values("acuracia")
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.barh(ordem["modelo"], ordem["acuracia"], xerr=ordem["desvio"], color="#4c78a8")
    ax.axvline(baseline, color="gray", linestyle="--", label=f"Baseline {baseline:.1%}")
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("Acurácia (validação cruzada aninhada, média +/- desvio)")
    ax.set_title("Comparação dos modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "comparacao_modelos.png", dpi=120)

    fig = Figure(figsize=(5.5, 4.5))
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(y_teste, previsto, labels=classes, ax=ax,
                                            cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão (teste)")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticklabels(classes, rotation=15)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=120)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    PASTA_MODELO.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    linhas = []

    def mostrar(texto=""):
        print(texto)
        linhas.append(texto)

    mostrar("=" * 72)
    mostrar(" TREINAMENTO - Breast Cancer (UCI) | alvo: Class")
    mostrar("=" * 72 + "\n")

    df = carregar_dados()
    baseline = analise_exploratoria(df, mostrar)

    X, y = df[ATRIBUTOS], df[ALVO]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TAMANHO_TESTE, stratify=y, random_state=SEMENTE)
    mostrar(f"\n3) TREINO / TESTE (estratificado, semente {SEMENTE}): "
            f"{len(X_treino)} treino | {len(X_teste)} teste")

    mostrar("\n4) COMPARAÇÃO DOS MODELOS - acurácia por validação cruzada aninhada "
            "(5 x 3 externa, 5 interna, só no treino)")
    tabela = comparar_modelos(X_treino, y_treino, mostrar)

    # Melhor modelo = maior acurácia média na validação cruzada aninhada.
    # O conjunto de teste NÃO participa da escolha (só avalia o modelo final).
    candidatos = tabela[~tabela["modelo"].str.startswith("Baseline")]
    melhor = candidatos.sort_values(["acuracia", "desvio"], ascending=[False, True]).iloc[0]
    nome = melhor["modelo"]

    ranking = tabela.sort_values(["acuracia", "desvio"], ascending=[False, True]).copy()
    ranking["acuracia"] = ranking["acuracia"].map("{:.1%}".format)
    ranking["desvio"] = ranking["desvio"].map("{:.1%}".format)
    mostrar("\n5) RANKING (maior acurácia primeiro)")
    mostrar(ranking.rename(columns={"modelo": "Modelo", "acuracia": "Acurácia",
                                    "desvio": "Desvio"}).to_string(index=False))

    mostrar("\n" + "=" * 72)
    mostrar(f" MELHOR MODELO: {nome}")
    mostrar("=" * 72)
    busca = ajustar_melhor(nome, X_treino, y_treino)
    modelo = busca.best_estimator_
    parametros = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
    mostrar(f"   Acurácia na validação cruzada aninhada: {melhor['acuracia']:.1%} "
            f"+/- {melhor['desvio']:.1%}")
    mostrar(f"   Hiperparâmetros escolhidos (GridSearchCV no treino): {parametros}")
    aval, previsto = avaliar_melhor(modelo, X_teste, y_teste, baseline, mostrar)

    # Salvar artefatos
    joblib.dump(modelo, CAMINHO_MODELO)
    salvar_graficos(tabela, baseline, y_teste, previsto, aval["classes"])
    tabela.to_csv(PASTA_RESULTADOS / "comparacao_modelos.csv", index=False, encoding="utf-8")

    metadados = {
        "base": "UCI Breast Cancer (id 14)",
        "data_treinamento": datetime.now().isoformat(timespec="seconds"),
        "versao_scikit_learn": sklearn.__version__,
        "semente": SEMENTE,
        "tamanho_teste": TAMANHO_TESTE,
        "melhor_modelo": nome,
        "hiperparametros": parametros,
        "acuracia_validacao_cruzada_aninhada": float(melhor["acuracia"]),
        "acuracia_teste": aval["acertos_teste"] / aval["n_teste"],
        "acuracia_baseline": float(baseline),
        "avaliacao": aval,
        "atributos": ATRIBUTOS,
        "opcionais": OPCIONAIS,
        "dominios": DOMINIOS,
        "moda_treino": {c: X_treino[c].mode()[0] for c in ATRIBUTOS},
        "interpretacao_classes": CLASSES,
    }
    CAMINHO_METADADOS.write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False, default=_para_json), encoding="utf-8")

    mostrar("\n6) ARQUIVOS GERADOS")
    for caminho in (CAMINHO_MODELO, CAMINHO_METADADOS,
                    PASTA_RESULTADOS / "comparacao_modelos.csv",
                    PASTA_RESULTADOS / "comparacao_modelos.png",
                    PASTA_RESULTADOS / "matriz_confusao.png",
                    PASTA_RESULTADOS / "relatorio_treinamento.txt"):
        mostrar(f"   {caminho.relative_to(RAIZ)}")
    (PASTA_RESULTADOS / "relatorio_treinamento.txt").write_text(
        "\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
