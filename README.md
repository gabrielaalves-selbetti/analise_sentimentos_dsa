# Análise de Sentimentos em Reviews de Produtos

Projeto desenvolvido ao longo do curso **Fundamentos de Python** da [Data Science Academy (DSA)](https://www.datascienceacademy.com.br/), no capítulo dedicado a **Machine Learning** — Mini Projeto 6.

## Objetivo

Construir um modelo de classificação de texto capaz de identificar o sentimento de reviews de produtos como **positivo** ou **negativo**, cobrindo todas as etapas de um pipeline real de Machine Learning: da exploração dos dados até o deploy do modelo.

## Dataset

| Campo | Descrição |
|---|---|
| `review_id` | Identificador único do review |
| `texto_review` | Texto escrito pelo cliente |
| `sentimento` | Rótulo da classe (`positivo` / `negativo`) |

- 500 registros originais, 488 após remoção de valores nulos
- Divisão treino/teste: 75% / 25% (estratificada)

## Pipeline

```
Texto bruto → Limpeza → TF-IDF → StandardScaler → Regressão Logística → Previsão
```

1. **EDA** — distribuição de classes e análise de dados ausentes
2. **Limpeza de texto** — remoção de acentos, lowercasing, eliminação de caracteres especiais
3. **Engenharia de atributos** — vetorização TF-IDF e codificação do rótulo alvo
4. **Otimização de hiperparâmetros** — `GridSearchCV` com validação cruzada de 5 folds (72 combinações testadas)
5. **Avaliação** — acurácia, relatório de classificação e matriz de confusão
6. **Deploy** — serialização com `joblib` e função de inferência para novos reviews

## Resultado

| Métrica | Valor |
|---|---|
| Acurácia | **81,15%** |
| F1-Score (Negativo) | 0,80 |
| F1-Score (Positivo) | 0,82 |

Melhores hiperparâmetros encontrados pelo GridSearchCV:

```python
{
    'tfidf__max_features': 500,
    'tfidf__ngram_range': (1, 1),
    'logreg__C': 0.1,
    'logreg__penalty': 'l1',
    'logreg__max_iter': 5000
}
```

## Estrutura do Projeto

```
analise_sentimentos_dsa/
├── app.ipynb                    # Notebook principal com todo o pipeline
├── dataset.csv                  # Dataset de reviews utilizado no projeto
├── modelo_sentimento_v1.joblib  # Modelo treinado e serializado
├── requirements.txt             # Dependências do projeto
```

## Como Executar

**1. Crie e ative o ambiente virtual:**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

**2. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**3. Abra o notebook:**

```bash
jupyter notebook app.ipynb
```

## Dependências Principais

- `scikit-learn` — pipeline de ML, TF-IDF, Regressão Logística, GridSearchCV
- `pandas` / `numpy` — manipulação e análise de dados
- `matplotlib` / `seaborn` — visualizações
- `joblib` — serialização do modelo

## Sobre o Curso

Este projeto faz parte do curso **Fundamentos de Python** da **Data Science Academy**, que cobre desde os fundamentos da linguagem até aplicações práticas de Machine Learning e Ciência de Dados.
