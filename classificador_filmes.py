import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Dados de Exemplo (um dataset real seria muito maior)
# Criamos um DataFrame com sinopses e seus respectivos gêneros.
dados = {
    'sinopse': [
        'Um grupo de amigos embarca em uma aventura hilária para recuperar um anel roubado.',
        'Uma história emocionante de amor e perda durante a guerra civil.',
        'Dois detetives cômicos investigam o roubo de um carro de luxo.',
        'A jornada de um herói para salvar seu reino de uma ameaça sombria.',
        'Uma família passa por dificuldades financeiras e encontra a esperança no outro.',
        'Conflitos familiares e segredos obscuros são revelados em um drama intenso.',
        'A busca por uma criatura lendária leva a situações muito engraçadas.'
    ],
    'genero': [
        'Comédia',
        'Drama',
        'Comédia',
        'Drama',
        'Drama',
        'Drama',
        'Comédia'
    ]
}
df = pd.DataFrame(dados)

# 2. Pré-processamento e Vetorização do Texto
# A sinopse é o "recurso" que usaremos para treinar o modelo.
# TfidfVectorizer transforma o texto em números (vetores).
vetorizador = TfidfVectorizer(lowercase=True, stop_words=['a', 'o', 'e', 'de', 'um', 'uma'])
X = vetorizador.fit_transform(df['sinopse'])
y = df['genero']

# 3. Divisão dos Dados em Treino e Teste
# Separamos 80% dos dados para treino e 20% para teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinamento do Modelo de Classificação
# Usamos a Regressão Logística, um algoritmo simples e eficaz.
modelo = LogisticRegression()
modelo.fit(X_treino, y_treino)

# 5. Avaliação do Modelo
# Fazemos previsões nos dados de teste e comparamos com os valores reais.
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# 6. Testando uma nova sinopse
nova_sinopse = ['Uma comédia romântica sobre um encontro desastroso que acaba em amor.']
nova_sinopse_vetorizada = vetorizador.transform(nova_sinopse)
previsao_nova = modelo.predict(nova_sinopse_vetorizada)

print(f"A sinopse '{nova_sinopse[0]}' foi classificada como: {previsao_nova[0]}")