# Previsão de recorrência de câncer de mama — UCI Breast Cancer

Avaliação de Data Science: classificador treinado na base **Breast Cancer** (UCI, id 14), com dois módulos.

| Módulo | Arquivo | O que faz |
|---|---|---|
| Treinamento | `treinamento.py` | Executa todo o processo, compara os modelos pela acurácia, ajusta e avalia o **melhor** e salva o modelo |
| Inferência | `inferencia.py` | Recebe os dados de uma **nova paciente** e devolve o atributo **Class** previsto |

> **Sobre o significado de `Class`:** na base, `Class` indica se houve **recorrência** do câncer de mama
> (`recurrence-events`) ou não (`no-recurrence-events`) em pacientes já tratadas. Não é "tem / não tem câncer".
> Projeto educacional; não substitui avaliação médica.

## 1. A base

- Fonte: Instituto de Oncologia de Ljubljana (Zwitter & Soklic, 1988) — 286 pacientes, 9 atributos categóricos + `Class`.
- Classes: `no-recurrence-events` 201 (70,3%) e `recurrence-events` 85 (29,7%) → **baseline** (sempre prever a majoritária) = 70,3% de acurácia.
- Valores ausentes (`?` no CSV): `node-caps` (8) e `breast-quad` (1).

## 2. Estrutura do repositório

```
├── data/breast-cancer.csv          base de dados
├── treinamento.py                  módulo de treinamento
├── inferencia.py                   módulo de inferência
├── modelo/
│   ├── modelo_final.joblib         pipeline completo (pré-processamento + melhor modelo)
│   └── metadados.json              modelo escolhido, métricas, domínio de cada atributo
├── resultados/
│   ├── relatorio_treinamento.txt   tudo que o treinamento imprimiu
│   ├── comparacao_modelos.csv / .png
│   └── matriz_confusao.png
├── exemplos/                       duas pacientes de exemplo (JSON)
├── tests/test_projeto.py           testes automáticos (pytest)
└── requirements.txt
```

## 3. Como executar

```powershell
python -m venv .venv
.venv\Scripts\Activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python treinamento.py             # treina e salva o modelo (o modelo já acompanha o repositório)
python inferencia.py              # modo interativo: pergunta cada atributo
python inferencia.py --arquivo exemplos/paciente_1.json
python inferencia.py --age 40-49 --menopause premeno --tumor-size 25-29 --inv-nodes 0-2 --node-caps no --deg-malig 2 --breast left --breast-quad left_low --irradiat no
pytest                            # testes automáticos
```

O treinamento leva de alguns segundos a poucos minutos, conforme a máquina (usa todos os núcleos).
Se o modelo salvo estiver ausente ou tiver sido gerado com outra versão do scikit-learn, a inferência treina novamente sozinha.

## 4. Módulo de treinamento — o que ele faz

1. **Carga e validação.** `?` vira valor ausente; todo valor é conferido com o domínio descrito na UCI.
2. **Análise exploratória.** Balanceamento das classes, **moda** de cada atributo e teste **qui-quadrado** atributo × `Class` (α = 5%).
   Associados à classe: `inv-nodes`, `node-caps`, `deg-malig` e `irradiat`; `tumor-size` fica no limite (p = 0,056);
   `age`, `menopause`, `breast` e `breast-quad` não mostram associação.
3. **Treino/teste.** 80/20, estratificado, semente 42 (228 treino / 58 teste). O teste é usado **uma única vez**, no final.
4. **Pré-processamento dentro do `Pipeline`:** imputação pela **moda** + **one-hot**. Todos os atributos são categóricos
   (`deg-malig`, de 1 a 3, é um grau e foi tratado como categoria), então não há o que normalizar: one-hot já deixa tudo em 0/1.
   Por estar no pipeline, o pré-processamento é refeito só com os dados de treino de cada divisão (sem vazamento) e é aplicado
   exatamente igual na inferência.
5. **Modelos comparados:** baseline, Regressão Logística, KNN, Árvore de Decisão, Random Forest, Gradient Boosting, SVM e Naive Bayes,
   cada um com sua grade de hiperparâmetros. A base é desbalanceada, então `class_weight='balanced'` (Regressão Logística e Árvore)
   e `fit_prior=False` (Naive Bayes, prior uniforme) entram como hiperparâmetros a serem testados.
6. **Comparação por validação cruzada aninhada** (só no treino): o laço interno (`GridSearchCV`) escolhe os hiperparâmetros e o
   externo (5 × 3) mede a acurácia. Assim a nota não é inflada pelo ajuste, e modelos com grades maiores não saem favorecidos.
   **Melhor modelo = maior acurácia média.** O teste não participa da escolha.
7. **Melhor modelo:** ajuste final dos hiperparâmetros e avaliação no teste (acurácia, IC 95% de Wilson, matriz de confusão, precisão/revocação/F1).
8. **Artefatos:** modelo, metadados, relatório, CSV e gráficos (pasta `resultados/`).

## 5. Resultados (semente 42, scikit-learn 1.9.1)

| Modelo | Acurácia (CV aninhada) | Desvio |
|---|---|---|
| **Árvore de Decisão** | **76,0%** | 4,1% |
| Gradient Boosting | 75,3% | 4,0% |
| Random Forest | 74,2% | 4,3% |
| KNN | 74,1% | 4,0% |
| SVM | 73,4% | 4,5% |
| Naive Bayes | 71,8% | 4,9% |
| Baseline (classe majoritária) | 70,2% | 0,8% |
| Regressão Logística | 69,5% | 5,4% |

**Modelo escolhido: Árvore de Decisão** (`max_depth=2`, `min_samples_leaf=1`, sem `class_weight`).

- Validação cruzada aninhada: **76,0%** ± 4,1%
- Teste: **70,7%** (41 de 58), IC 95% de 58,0% a 80,8% (baseline: 70,3%)

| Teste (58 pacientes) | previsto: no-recurrence | previsto: recurrence |
|---|---|---|
| real: no-recurrence (41) | 38 | 3 |
| real: recurrence (17) | 14 | 3 |

**O que a árvore aprendeu** (4 folhas): prevê `recurrence-events` apenas quando `inv-nodes` ≥ 3 (fora da faixa 0-2) **e** `deg-malig` = 3
(24 das 31 pacientes de treino nessa folha tiveram recorrência); nos demais casos prevê `no-recurrence-events`.
É uma regra simples e coerente com fatores prognósticos conhecidos do câncer de mama: mais linfonodos comprometidos e maior grau de malignidade.

**Leitura honesta dos números**

- Na validação cruzada o melhor modelo supera o baseline em cerca de 6 pontos, mas no teste (58 pacientes) fica praticamente empatado com ele.
  Cada paciente a mais ou a menos muda a acurácia em 1,7 ponto, e o intervalo de confiança é largo.
- Entre os cinco primeiros modelos a diferença é de ~2,6 pontos, menor que o desvio da validação cruzada (~4 pontos): não há vencedor estatisticamente claro.
  É uma base pequena e de sinal fraco (o baseline sozinho já acerta 70%).
- Como a base é desbalanceada e a prova pede ênfase em acurácia, o modelo acerta bem a classe majoritária
  e tem **revocação baixa para `recurrence-events` (17,6% no teste)**. Para uso real, essa métrica seria o foco.

## 6. Módulo de inferência

Entrada: os 9 atributos da paciente, exatamente como no CSV (`node-caps` e `breast-quad` são opcionais; se faltarem, o modelo usa a moda do treino).
O programa valida cada valor (listando os aceitos), aceita maiúsculas/minúsculas e `left-low` ou `left_low`, e **não grava** os dados informados.

```
$ python inferencia.py --arquivo exemplos/paciente_2.json
 Class prevista : recurrence-events
 Indicativo     : indicativo de RECORRÊNCIA do câncer de mama
 Probabilidades : no-recurrence-events 22.6% | recurrence-events 77.4%

$ python inferencia.py --arquivo exemplos/paciente_1.json
 Class prevista : no-recurrence-events
 Indicativo     : sem indicativo de recorrência do câncer de mama
 Probabilidades : no-recurrence-events 76.4% | recurrence-events 23.6%
```

Também pode ser usado como biblioteca: `from inferencia import prever; prever({"age": "40-49", ...})`.

## 7. Testes

`pytest` executa 13 testes: carga e validação dos dados, pipeline com valor ausente e categoria nunca vista, consistência do modelo salvo,
previsão, probabilidades, validação de entradas inválidas e execução pelo terminal (arquivo JSON e argumentos).

## 8. Governança de dados (LGPD)

- Base pública (CC BY 4.0), sem identificadores diretos das pacientes.
- Dado de saúde é **dado pessoal sensível** (Lei 13.709/2018, art. 5º, II): a inferência processa a entrada só em memória, sem gravar nem enviar.
- Modelo apenas educacional; qualquer decisão clínica cabe a um profissional de saúde.

## 9. Referência

Zwitter, M. & Soklic, M. (1988). *Breast Cancer* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51P4M
