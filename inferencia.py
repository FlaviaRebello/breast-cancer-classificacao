"""
MÓDULO DE INFERÊNCIA - Breast Cancer (UCI)

Recebe os dados de uma NOVA PACIENTE e indica o atributo Class
(no-recurrence-events | recurrence-events), usando o modelo salvo por treinamento.py.
O pré-processamento (imputação e one-hot) já está dentro do modelo salvo, então
a entrada é a paciente "crua", exatamente como aparece no CSV.

Formas de uso:
  python inferencia.py                               # modo interativo (pergunta cada atributo)
  python inferencia.py --arquivo exemplos/paciente_1.json
  python inferencia.py --age 40-49 --menopause premeno --tumor-size 25-29 --inv-nodes 0-2 \
                       --node-caps no --deg-malig 2 --breast left --breast-quad left_low --irradiat no

Como biblioteca:
  from inferencia import prever
  prever({"age": "40-49", "menopause": "premeno", ...})

Projeto educacional: não substitui avaliação médica.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import treinamento
from treinamento import (ATRIBUTOS, CAMINHO_METADADOS, CAMINHO_MODELO, CLASSES, DOMINIOS,
                         OPCIONAIS, normalizar_valor)


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def _modelo_utilizavel():
    """True se o modelo salvo existe e foi gerado com esta versão do scikit-learn."""
    if not (CAMINHO_MODELO.exists() and CAMINHO_METADADOS.exists()):
        return False
    metadados = json.loads(CAMINHO_METADADOS.read_text(encoding="utf-8"))
    return metadados.get("versao_scikit_learn") == sklearn.__version__


def carregar_modelo():
    """Carrega (modelo, metadados). Se o modelo faltar ou for de outra versão, treina de novo."""
    if not _modelo_utilizavel():
        print("Modelo ausente ou gerado com outra versão do scikit-learn: "
              "treinando novamente (leva alguns instantes)...\n", file=sys.stderr)
        treinamento.main()
        print("\n" + "-" * 72 + "\n", file=sys.stderr)
    modelo = joblib.load(CAMINHO_MODELO)
    metadados = json.loads(CAMINHO_METADADOS.read_text(encoding="utf-8"))
    return modelo, metadados


# ----------------------------------------------------------------------------
# Validação da paciente
# ----------------------------------------------------------------------------
def _vazio(valor):
    return valor is None or (isinstance(valor, float) and np.isnan(valor)) \
        or (isinstance(valor, str) and valor.strip() in ("", "?"))


def validar_atributo(atributo, bruto):
    """Valida UM atributo. Devolve o valor padronizado (np.nan se opcional e vazio) ou levanta ValueError."""
    aceitos = ", ".join(map(str, DOMINIOS[atributo]))
    if _vazio(bruto):
        if atributo in OPCIONAIS:
            return np.nan                       # o modelo imputa pela moda do treino
        raise ValueError(f"'{atributo}' é obrigatório. Valores aceitos: {aceitos}")
    try:
        valor = normalizar_valor(atributo, bruto)
    except (TypeError, ValueError):
        valor = None
    if valor not in DOMINIOS[atributo]:
        raise ValueError(f"'{atributo}': valor inválido {bruto!r}. Valores aceitos: {aceitos}")
    return valor


def validar_paciente(paciente):
    """Confere e padroniza os dados. Levanta ValueError listando TODOS os problemas."""
    erros, limpo = [], {}

    desconhecidos = sorted(set(paciente) - set(ATRIBUTOS))
    if desconhecidos:
        erros.append(f"Atributos desconhecidos: {desconhecidos}. Use: {ATRIBUTOS}")

    for atributo in ATRIBUTOS:
        try:
            limpo[atributo] = validar_atributo(atributo, paciente.get(atributo))
        except ValueError as erro:
            erros.append(str(erro))

    if erros:
        raise ValueError("\n".join(erros))
    return limpo


# ----------------------------------------------------------------------------
# Inferência
# ----------------------------------------------------------------------------
def prever(paciente, modelo=None, metadados=None):
    """Prevê a Class de uma paciente (dict atributo -> valor).

    Retorna um dict com: classe, indicativo, probabilidades (se o modelo oferecer),
    paciente (dados padronizados) e imputados (atributos ausentes preenchidos pela moda).
    """
    if modelo is None or metadados is None:
        modelo, metadados = carregar_modelo()

    dados = validar_paciente(paciente)
    entrada = pd.DataFrame([dados], columns=ATRIBUTOS)

    classe = str(modelo.predict(entrada)[0])
    resultado = {
        "classe": classe,
        "indicativo": CLASSES[classe],
        "paciente": dados,
        "imputados": {a: metadados["moda_treino"][a] for a in ATRIBUTOS if _vazio(dados[a])},
    }
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(entrada)[0]
        resultado["probabilidades"] = {str(c): float(p) for c, p in zip(modelo.classes_, probabilidades)}
    return resultado


def formatar_resultado(resultado, metadados):
    linhas = ["=" * 72, " INFERÊNCIA - nova paciente", "=" * 72, " Dados informados:"]
    for atributo, valor in resultado["paciente"].items():
        if atributo in resultado["imputados"]:
            valor = f"(não informado -> moda do treino: {resultado['imputados'][atributo]})"
        linhas.append(f"   {atributo:12s}: {valor}")

    linhas += ["", f" Class prevista : {resultado['classe']}",
               f" Indicativo     : {resultado['indicativo']}"]
    if "probabilidades" in resultado:
        texto = " | ".join(f"{c} {p:.1%}" for c, p in resultado["probabilidades"].items())
        linhas.append(f" Probabilidades : {texto}")
    linhas += ["",
               f" Modelo: {metadados['melhor_modelo']} | acurácia no teste: "
               f"{metadados['acuracia_teste']:.1%} (baseline {metadados['acuracia_baseline']:.1%})",
               " Aviso: projeto educacional, não substitui avaliação médica."]
    return "\n".join(linhas)


# ----------------------------------------------------------------------------
# Entrada de dados (argumentos, arquivo ou interativo)
# ----------------------------------------------------------------------------
def montar_parser():
    parser = argparse.ArgumentParser(
        description="Prevê o atributo Class (recorrência) de uma nova paciente.",
        epilog="Sem argumentos, o programa pergunta cada atributo (modo interativo).",
        allow_abbrev=False)
    parser.add_argument("--arquivo", help="JSON com os dados da paciente (ex.: exemplos/paciente_1.json)")
    for atributo in ATRIBUTOS:
        aceitos = ", ".join(map(str, DOMINIOS[atributo]))
        opcional = " (opcional)" if atributo in OPCIONAIS else ""
        parser.add_argument(f"--{atributo}", dest=atributo, help=f"valores: {aceitos}{opcional}")
    return parser


def perguntar_paciente():
    """Modo interativo: pergunta cada atributo até receber um valor válido."""
    print("Informe os dados da paciente (digite o valor exatamente como listado).\n")
    paciente = {}
    for atributo in ATRIBUTOS:
        aceitos = " | ".join(map(str, DOMINIOS[atributo]))
        dica = "  [Enter = não informado]" if atributo in OPCIONAIS else ""
        while True:
            resposta = input(f"{atributo}  [{aceitos}]{dica}: ")
            try:
                validar_atributo(atributo, resposta)
            except ValueError as erro:
                print("   ->", erro)
                continue
            paciente[atributo] = resposta
            break
    print()
    return paciente


def main(argv=None):
    args = montar_parser().parse_args(argv)
    try:
        if args.arquivo:
            paciente = json.loads(Path(args.arquivo).read_text(encoding="utf-8"))
        else:
            paciente = {}
        paciente.update({a: v for a, v in vars(args).items() if a in ATRIBUTOS and v is not None})

        if not paciente:
            paciente = perguntar_paciente()

        modelo, metadados = carregar_modelo()
        resultado = prever(paciente, modelo, metadados)
    except FileNotFoundError as erro:
        print(f"Erro: arquivo não encontrado: {erro.filename}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as erro:           # JSON inválido, valor inválido, erro de leitura
        print(f"Erro: {erro}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nEntrada interrompida.", file=sys.stderr)
        return 1

    print(formatar_resultado(resultado, metadados))
    return 0


if __name__ == "__main__":
    sys.exit(main())
