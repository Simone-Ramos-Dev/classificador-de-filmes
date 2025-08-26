# Classificador de Gênero de Filme com Python

Este projeto é um classificador simples que usa aprendizado de máquina para prever o gênero de um filme (Comédia ou Drama) com base em sua sinopse. O modelo é treinado usando a linguagem Python e a biblioteca Scikit-learn.

## 🚀 Como Executar o Projeto

Siga os passos abaixo para rodar o classificador na sua máquina.

### Pré-requisitos

Certifique-se de ter o **Python 3** instalado. Você pode verificar a versão com o seguinte comando no seu terminal:

```bash
python --version
````

### Instalação das Dependências

Instale as bibliotecas necessárias usando o `pip`. Abra o terminal na pasta do projeto e execute:

```bash
pip install pandas scikit-learn
```

### Executando o Script

Após a instalação das dependências, você pode rodar o classificador.

```bash
python classificador_filmes.py
```

O script irá treinar o modelo, exibir a acurácia (o quão preciso ele foi) e fazer uma previsão para uma nova sinopse de teste.

## ⚙️ Tecnologias Utilizadas

  * **Python:** A linguagem de programação principal.
  * **Pandas:** Usada para a manipulação e organização dos dados.
  * **Scikit-learn:** Biblioteca de aprendizado de máquina que fornece as ferramentas para a vetorização do texto e o treinamento do modelo.

## 📚 Como Funciona o Código

1.  **Dados de Exemplo:** O script começa com um pequeno conjunto de dados de exemplo, com sinopses de filmes e seus respectivos gêneros.
2.  **Vetorização:** As sinopses são transformadas em números por meio do `TfidfVectorizer`, uma técnica que converte texto em vetores numéricos para que o modelo possa entendê-los.
3.  **Treinamento do Modelo:** Os dados vetorizados são usados para treinar um modelo de **Regressão Logística**.
4.  **Avaliação:** O modelo é testado em um conjunto de dados que ele nunca viu para calcular sua acurácia.
5.  **Previsão:** Por fim, o modelo é usado para classificar uma nova sinopse.

## 🤝 Contribuições

Contribuições são bem-vindas\! Se você tiver sugestões de melhoria ou quiser adicionar novas funcionalidades, sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

