# Importando as bibliotecas necessárias
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.cloud import language_v1
from flask import Flask, request, jsonify
import google.generativeai as palm
from sklearn.linear_model import LogisticRegression  # Exemplo de modelo
import pickle

# Configurando a API Key da Gemini
palm.configure(api_key='')

# Carregando o modelo de Machine Learning treinado
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Função para coletar comentários online
def coletar_comentarios(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    comentarios = soup.find_all('div', class_='comentario')
    return [comentario.text for comentario in comentarios]

# Função para pré-processar os dados coletados
def pre_processar(texto):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))
    word_tokens = word_tokenize(texto)
    texto_filtrado = [w for w in word_tokens if not w in stop_words]
    return texto_filtrado

# Função para analisar o sentimento de um texto
def analisar_sentimento(texto):
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "pt-BR"
    document = {"content": texto, "type_": type_, "language": language}
    response = client.analyze_sentiment(request={'document': document})
    return response.document_sentiment.score, response.document_sentiment.magnitude

# Função para gerar embeddings
def gerar_embeddings(text):
    response = palm.generate_embeddings(model='models/embedding-gecko', text=text)
    return response.embeddings

# Configurando a aplicação Flask
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Texto não fornecido'}), 400
    embedding = gerar_embeddings(text)
    sentiment = model.predict([embedding])[0]
    confidence = max(model.predict_proba([embedding])[0])
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
