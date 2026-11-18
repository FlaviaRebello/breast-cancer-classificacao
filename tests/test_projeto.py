"""Testes automáticos dos módulos de treinamento e inferência.  Rode com:  pytest"""
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import inferencia
import treinamento

RAIZ = treinamento.RAIZ

PACIENTE = {"age": "40-49", "menopause": "premeno", "tumor-size": "25-29", "inv-nodes": "0-2",
            "node-caps": "no", "deg-malig": 2, "breast": "left", "breast-quad": "left_low",
            "irradiat": "no"}


@pytest.fixture(scope="module")
def modelo():
    """(pipeline, metadados) salvos por treinamento.py (se faltarem, são gerados)."""
    return inferencia.carregar_modelo()


def rodar_cli(*argumentos):
    """Executa inferencia.py como um usuário faria, no terminal."""
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(RAIZ / "inferencia.py"), *argumentos],
                          capture_output=True, text=True, encoding="utf-8", env=ambiente)


# ---------------------------------------------------------------- dados
def test_dados_carregam_e_ausentes_viram_nan():
    df = treinamento.carregar_dados()
    assert df.shape == (286, 10)
    assert df["node-caps"].isna().sum() == 8
    assert df["breast-quad"].isna().sum() == 1
    assert set(df["Class"]) == set(treinamento.CLASSES)


def test_valor_fora_do_dominio_e_rejeitado(tmp_path):
    texto = treinamento.CAMINHO_DADOS.read_text(encoding="utf-8").replace("50-59", "55-59x", 1)
    csv_ruim = tmp_path / "ruim.csv"
    csv_ruim.write_text(texto, encoding="utf-8")
    with pytest.raises(ValueError, match="fora do domínio"):
        treinamento.carregar_dados(csv_ruim)


def test_pipeline_aceita_ausente_e_categoria_nao_vista_no_treino():
    df = treinamento.carregar_dados()
    X, y = df[treinamento.ATRIBUTOS], df[treinamento.ALVO]
    pipeline = treinamento.montar_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X.iloc[:100], y.iloc[:100])

    novo = X.iloc[[0]].copy()
    novo["node-caps"] = np.nan          # ausente -> imputado pela moda do treino
    novo["age"] = "20-29"               # faixa rara: pode não existir nas 100 primeiras linhas
    assert pipeline.predict(novo)[0] in treinamento.CLASSES


# ---------------------------------------------------------------- modelo salvo
def test_metadados_consistentes_com_o_modelo(modelo):
    pipeline, metadados = modelo
    assert metadados["atributos"] == treinamento.ATRIBUTOS
    assert set(pipeline.classes_) == set(treinamento.CLASSES)
    assert 0 < metadados["acuracia_teste"] <= 1
    assert 0 < metadados["acuracia_validacao_cruzada_aninhada"] <= 1


# ---------------------------------------------------------------- inferência (função)
def test_prever_retorna_classe_valida(modelo):
    resultado = inferencia.prever(PACIENTE, *modelo)
    assert resultado["classe"] in treinamento.CLASSES
    assert resultado["indicativo"] == treinamento.CLASSES[resultado["classe"]]


def test_probabilidades_somam_1(modelo):
    resultado = inferencia.prever(PACIENTE, *modelo)
    assert sum(resultado["probabilidades"].values()) == pytest.approx(1.0)


def test_campos_opcionais_podem_faltar(modelo):
    paciente = {k: v for k, v in PACIENTE.items() if k not in treinamento.OPCIONAIS}
    resultado = inferencia.prever(paciente, *modelo)
    assert set(resultado["imputados"]) == set(treinamento.OPCIONAIS)


def test_valor_invalido_e_campo_obrigatorio_geram_erro(modelo):
    with pytest.raises(ValueError, match="'age': valor inválido"):
        inferencia.prever({**PACIENTE, "age": "100-109"}, *modelo)
    with pytest.raises(ValueError, match="'deg-malig': valor inválido"):
        inferencia.prever({**PACIENTE, "deg-malig": 2.5}, *modelo)
    sem_idade = {k: v for k, v in PACIENTE.items() if k != "age"}
    with pytest.raises(ValueError, match="'age' é obrigatório"):
        inferencia.prever(sem_idade, *modelo)


def test_formas_equivalentes_de_digitar_dao_a_mesma_previsao(modelo):
    """Maiúsculas, espaços, '-' em breast-quad e deg-malig em texto não mudam o resultado."""
    variante = {**PACIENTE, "menopause": " PREMENO ", "breast-quad": "Left-Low", "deg-malig": "2"}
    assert inferencia.prever(variante, *modelo)["classe"] == inferencia.prever(PACIENTE, *modelo)["classe"]


# ---------------------------------------------------------------- inferência (terminal)
@pytest.mark.parametrize("exemplo", ["paciente_1.json", "paciente_2.json"])
def test_cli_com_arquivo_de_exemplo(exemplo):
    resultado = rodar_cli("--arquivo", str(RAIZ / "exemplos" / exemplo))
    assert resultado.returncode == 0, resultado.stderr
    assert "Class prevista" in resultado.stdout


def test_cli_com_argumentos():
    resultado = rodar_cli("--age", "40-49", "--menopause", "premeno", "--tumor-size", "25-29",
                          "--inv-nodes", "0-2", "--deg-malig", "2", "--breast", "left",
                          "--irradiat", "no")
    assert resultado.returncode == 0, resultado.stderr
    assert "não informado" in resultado.stdout       # node-caps e breast-quad omitidos


def test_cli_valor_invalido_retorna_erro():
    resultado = rodar_cli("--arquivo", str(RAIZ / "exemplos" / "paciente_1.json"), "--age", "x")
    assert resultado.returncode == 2
    assert "valor inválido" in resultado.stderr
