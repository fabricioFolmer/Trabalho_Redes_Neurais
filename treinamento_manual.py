import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE

# Carrega o dataset
df = pd.read_csv("data\\bank-full.csv", sep=";", quotechar='"')

# Remove colunas desnecessárias
df.drop(columns=['duration', 'pdays'], inplace=True)

# Mapear a variável alvo 'y' para 0 e 1
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Separar features (X) e target (y) em dois Dataframes diferentes
features = df.drop('y', axis=1)
target = df['y']

# Identificar colunas numéricas e categóricas
numerical_cols = features.select_dtypes(include=np.number).columns.tolist()
categorical_cols = features.select_dtypes(include='object').columns.tolist()

# Pré-processamento:
# 1. Tratamento de "unknown" em colunas categóricas: Vamos tratar "unknown" como uma categoria própria.
#    O OneHotEncoder fará isso automaticamente.
# 2. Encoding de variáveis categóricas (One-hot Encoding)
# 3. Normalização de variáveis numéricas (StandardScaler)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Divisão dos dados em conjuntos de treinamento e teste
# Divide o conjunto de dados em quatro partes: 
#   X_train (características para treinamento), 
#   X_test (características para teste),
#   y_train (rótulos para treinamento),
#   y_test (rótulos para teste).
# A divisão é feita de forma que 20% dos dados (test_size=0.2) sejam usados para teste e os 80% restantes para treinamento. 
# O random_state=42 garante que a divisão seja a mesma a cada execução, para reprodutibilidade.
# O parâmetro stratify=target é crucial para problemas com classes desbalanceadas, pois assegura que a proporção das classes na variável target seja mantida tanto nos conjuntos de treinamento quanto nos de teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target) # Usar stratify para manter a proporção de classes


# Balanceamento de classes usando SMOTE (apenas no conjunto de treino para evitar vazamento de dados para o cnojunto de teste)
# Na prático, o SMOTE (Synthetic Minority Over-sampling Technique) é usado para lidar com o desbalanceamento de classes, criando amostras sintéticas da classe minoritária.
# Isso é feito para melhorar a performance do modelo em problemas de classificação desbalanceada, onde uma classe é significativamente menos representada que a outra.
# Por exemplo, a coluna 'y' no dataset pode ter uma distribuição de 90% de 'no' e 10% de 'yes', o que pode levar o modelo a prever sempre a classe majoritária.
smote = SMOTE(random_state=42)
X_train_processed, y_train_resampled = preprocessor.fit_transform(X_train), y_train
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_resampled)

# Aplicar o mesmo pré-processamento nos dados de teste (apenas transformação)
X_test_processed = preprocessor.transform(X_test)


# Construção do Modelo de Rede Neural Artificial (MLPClassifier)
# Parâmetros sugeridos para teste:
# - hidden_layer_sizes: (100,) (50, 50), (100, 50, 25)
# - activation: 'relu', 'tanh', 'logistic'
# - solver: 'adam', 'sgd'
# - learning_rate_init: 0.001, 0.01, 0.1
# - max_iter: número de iterações (epochs)
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=True
)

# Realiza o treinamento do modelo
print("\nIniciando o treinamento do modelo...")
mlp.fit(X_train_resampled, y_train_resampled)
print("Treinamento concluído.")

# Faz as previsões no conjunto de teste
y_pred = mlp.predict(X_test_processed)

# Apura as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Exibe os resultados do modelo
print("\n--- Resultados do Modelo ---")
print(f"Erro atingido no aprendizado (final loss): {mlp.loss_}")
print(f"Número de iterações atingido no aprendizado: {mlp.n_iter_}")
print(f"Taxa de aprendizado: {mlp.learning_rate_init}")
print(f"Número de camadas intermediárias: {mlp.n_layers_ - 2}") # Total layers - input - output
print(f"Neurônios nas camadas intermediárias: {mlp.hidden_layer_sizes}")
print(f"\nAcurácia: {accuracy:.4f}")
print("\nMatriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(class_report)

# Visualização da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Aderiu (0)', 'Aderiu (1)'],
            yticklabels=['Não Aderiu (0)', 'Aderiu (1)'])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()


# Exemplo de utilização do modelo com dados de amostra
def realizar_predicao(model, preprocessor_pipeline, sample_data):
    # Criar um DataFrame a partir dos dados de amostra
    sample_df = pd.DataFrame([sample_data])
    # Aplicar o pré-processamento
    processed_sample = preprocessor_pipeline.transform(sample_df)
    # Fazer a previsão
    prediction = model.predict(processed_sample)
    # Fazer a probabilidade de previsão
    prediction_proba = model.predict_proba(processed_sample)
    return "Yes" if prediction[0] == 1 else "No", prediction_proba[0]

# Exemplo de padrão de entrada definido pelo usuário
user_input_pattern = {
    'age': 22,
    'job': 'student',
    'marital': 'single',
    'education': 'high.school',
    'default': 'no',
    'balance': 5000,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 20,
    'month': 'oct',
    'duration': 600,
    'campaign': 1,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown'
}

predicted_output, probas = realizar_predicao(mlp, preprocessor, user_input_pattern)

print(f"\n--- Simulação com Padrão de Entrada Definido pelo Usuário ---")
print(f"Valores dos padrões de entrada definidos pelo usuário: {user_input_pattern}")
print(f"Valor do padrão de saída reconhecido: {predicted_output}")
print(f"Probabilidades de saída (Não Aderiu / Aderiu): {probas}")

# Testes realizados com diferentes parâmetros (exemplo)
# Para realizar testes com diferentes parâmetros, você pode criar um loop ou usar GridSearchCV
# para explorar combinações de hiperparâmetros e encontrar a melhor configuração.
# Exemplo de alteração de parâmetros:
# mlp_test = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', solver='sgd',
#                         learning_rate_init=0.01, max_iter=100, random_state=42, verbose=False)
# mlp_test.fit(X_train_resampled, y_train_resampled)
# y_pred_test = mlp_test.predict(preprocessor.transform(X_test))
# print(f"\nAcurácia com hidden_layer_sizes=(50,), activation='tanh', solver='sgd', learning_rate_init=0.01: {accuracy_score(y_test, y_pred_test):.4f}")