# Trabalho de Redes Neurais Artificiais: Previsão de Adesão a Depósito à Prazo

## 1. Visão Geral do Projeto

Este projeto da disciplina de Inteligência Artificial (2025) da UNISC aplica Redes Neurais Artificiais (RNAs) para prever a adesão de clientes de um banco português a um depósito a prazo. Para isso, utilizamos o dataset "Bank Marketing" do UCI Machine Learning Repository, com dados de campanhas de marketing por telefone.

## 2. Problema a Ser Resolvido

O desafio é um problema de **classificação binária**: prever se um cliente irá aderir (`yes`) ou não (`no`) a um depósito a prazo, com base em suas informações e histórico de contato.

## 3. Dataset Utilizado

O dataset "Bank Marketing"  possui as seguintes características:
* **Fonte:** UCI Machine Learning Repository.
* **Origem:** Campanha de marketing direto de uma instituição bancária portuguesa.
* **Instâncias:** 45.211.
* **Colunas (Features):** 16, além da variável alvo (`y`).
* **Período dos Dados:** Maio de 2008 a Novembro de 2010.

**Descrição Detalhada das Colunas:**

**Dados do Cliente Bancário:**
1.  **age** (Idade): Numérico. Idade do cliente.
2.  **job** (Profissão): Categórico. Tipo de trabalho (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`).
3.  **marital** (Estado Civil): Categórico (`married`, `divorced`, `single`, `unknown`).
4.  **education** (Educação): Categórico (`basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown`).
5.  **default** (Inadimplência): Binário (`yes`, `no`).
6.  **balance** (Saldo): Numérico. Saldo médio anual da conta em euros.
7.  **housing** (Financiamento Habitacional): Binário (`yes`, `no`).
8.  **loan** (Empréstimo Pessoal): Binário (`yes`, `no`).

**Dados do Último Contato da Campanha Atual:**
9.  **contact** (Tipo de Contato): Categórico (`cellular`, `telephone`, `unknown`).
10. **day** (Dia do Contato): Numérico (1 a 31).
11. **month** (Mês do Contato): Categórico (`jan`, `feb`, ..., `dec`).
12. **duration** (Duração do Contato): Numérico (em segundos). **Nota:** Esta variável não está disponível antes do contato e deve ser usada com cautela para previsões pré-contato.

**Outros Atributos da Campanha:**
13. **campaign** (Número de Contatos): Numérico. Contatos realizados durante esta campanha para o cliente.
14. **pdays** (Dias Desde Último Contato): Numérico. Dias desde o último contato em campanha anterior (-1 se nunca contatado).
15. **previous** (Contatos Anteriores): Numérico. Contatos realizados antes desta campanha.
16. **poutcome** (Resultado da Campanha Anterior): Categórico (`success`, `failure`, `other`, `unknown`).

**Variável Alvo (Target):**
17. **y** (Adesão ao Depósito): Binário (`yes`, `no`).

## 4. Tratamento de Dados Realizado

Para preparar os dados para a Rede Neural, realizamos os seguintes tratamentos:

*   **Encoding de Variáveis Categóricas:** Utilizamos **One-Hot Encoding** para converter variáveis categóricas em formato numérico, evitando a criação de uma ordem artificial entre as categorias.
*   **Normalização das Variáveis Numéricas:** As variáveis numéricas foram normalizadas com **StandardScaler** para que nenhuma delas domine o treinamento por ter uma escala maior.
*   **Tratamento de Valores Desconhecidos (`"unknown"`):** Os valores `"unknown"` foram mantidos como uma categoria separada no processo de One-Hot Encoding.
*   **Balanceamento de Classes:** O dataset é **desbalanceado**. Para corrigir isso e evitar que o modelo favorecesse a classe majoritária, aplicamos a técnica **SMOTE (Synthetic Minority Over-sampling Technique)** no conjunto de treino, criando dados sintéticos da classe com menos amostras.
*   **Mapeamento da Variável Alvo:** A variável alvo `y` foi mapeada de 'no' para `0` e 'yes' para `1`.

## 5. Modelo de RNA Escolhido

Escolhemos um **Perceptron Multicamadas (MLP)**, um tipo de rede neural capaz de aprender padrões complexos, ideal para este problema de classificação.

**Parâmetros do Modelo (Exemplo de Configuração Inicial):**
* **`hidden_layer_sizes`**: `(100, 50)` - Duas camadas ocultas, com 100 e 50 neurônios, respectivamente.
* **`activation`**: `'relu'` - Função de ativação retificada linear (ReLU).
* **`solver`**: `'adam'` - Otimizador Adam, conhecido por sua eficiência em problemas de grande escala.
* **`learning_rate_init`**: `0.001` - Taxa de aprendizado inicial.
* **`max_iter`**: `200` - Número máximo de iterações (épocas) de treinamento.
* **`random_state`**: `42` - Para reprodutibilidade dos resultados.
* **`verbose`**: `True` - Para exibir o progresso do treinamento.

A escolha final dos parâmetros do MLP foi baseada na análise de métricas de desempenho, como:
*   **Acurácia:** Proporção de previsões corretas.
*   **Matriz de Confusão:** Representação visual dos verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
*   **Relatório de Classificação:** Contém `precision`, `recall` e `F1-score` para cada classe, sendo crucial para datasets desbalanceados.

## 6. Etapas Desenvolvidas

1.  **Carregamento e Análise dos Dados:** Importamos o dataset e analisamos suas características.
2.  **Pré-processamento:**
    *   Identificação de colunas numéricas e categóricas.
    *   Criação de um `ColumnTransformer` para aplicar `StandardScaler` e `OneHotEncoder`.
    *   Mapeamento da variável alvo `y` para 0 e 1.
3.  **Divisão dos Dados:** Dividimos os dados em 80% para treino e 20% para teste, mantendo a proporção das classes com o parâmetro `stratify`.
4.  **Balanceamento de Classes:** Aplicação de `SMOTE` no conjunto de treinamento para gerar amostras sintéticas da classe minoritária.
5.  **Treinamento do Modelo:** Criamos e treinamos o `MLPClassifier` com os dados tratados.
6.  **Avaliação do Modelo:**
    *   Realização de previsões no conjunto de teste.
    *   Cálculo e exibição do erro (`loss`) e do número de iterações.
    *   Geração da acurácia, matriz de confusão e relatório de classificação.
    *   Visualização da matriz de confusão.
7.  **Simulação:** Criamos uma função para testar o modelo com novos dados, simulando um caso de uso real.

## 7. Estrutura dos Scripts: `treinamento_manual.py` vs. `grid_search.py`

O projeto foi dividido em dois scripts principais:

*   **`treinamento_manual.py`**:
    *   **Propósito**: Treinar e avaliar um modelo `MLPClassifier` com **hiperparâmetros fixos**.
    *   **Funcionalidade**: Realiza todo o pipeline de pré-processamento, balanceamento, treinamento e avaliação do modelo.
    *   **Uso Ideal**: Útil para entender o fluxo do projeto e para testes rápidos com uma configuração específica.

*   **`grid_search.py`**:
    *   **Propósito**: Otimizar o modelo, **buscando os melhores hiperparâmetros** com `GridSearchCV`.
    *   **Funcionalidade**: Testa sistematicamente múltiplas combinações de hiperparâmetros, encontra a melhor, e então treina e avalia o modelo final.
    *   **Uso Ideal**: Abordagem recomendada para encontrar um modelo mais robusto e com melhor desempenho.

Resumindo: `treinamento_manual.py` é uma implementação direta para testes rápidos, enquanto `grid_search.py` é a abordagem mais completa para otimizar a performance do modelo.

## 8. Ferramentas e Bibliotecas Utilizadas

* **Python:** Linguagem de programação principal.
* **Scikit-learn:** Para pré-processamento de dados (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `train_test_split`), construção do modelo (`MLPClassifier`) e avaliação (`accuracy_score`, `confusion_matrix`, `classification_report`).
* **Pandas:** Para manipulação e análise de dados (estruturas `DataFrame`).
* **Numpy:** Para operações numéricas.
* **Matplotlib e Seaborn:** Para visualização de dados (e.g., matriz de confusão).
* **Imbalanced-learn (imblearn):** Para o balanceamento de classes (`SMOTE`).

## 9. Principal Desafio Encontrado Durante o Desenvolvimento

Os principais desafios encontrados foram:

*   **Tratamento de Dados:** Lidar com os valores `"unknown"` nas variáveis categóricas e escolher a melhor forma de tratá-los (One-Hot Encoding).
*   **Balanceamento de Classes:** O grande desbalanceamento da variável alvo (`y`) exigiu o uso de SMOTE para que o modelo não aprendesse a prever apenas a classe majoritária.
*   **Variável `duration`:** A variável `duration` é um forte preditor, mas só é conhecida *após* a ligação. Isso é um ponto de atenção, pois o objetivo ideal seria prever a adesão *antes* do contato. Mantivemos a variável, mas essa limitação é importante para um cenário real.
*   **Ajuste de Hiperparâmetros:** Encontrar a melhor combinação de hiperparâmetros para o `MLPClassifier` foi um processo iterativo que demandou vários testes e análises.
