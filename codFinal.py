import re
import pandas as pd

from langdetect import detect
from deep_translator import GoogleTranslator

def traduzir_textos(texto):
    if not isinstance(texto, str) or texto.strip() == "" or texto.lower() == "nan":
        return None

    idioma = detect(texto)
    traduzido = GoogleTranslator(source=idioma, target='en').translate(texto)
    
    return traduzido

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return None
    
    texto = texto.lower()
    
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"[^a-zà-ÿ ]", "", texto)
    
    return texto.strip()

csv = 'tokyo_2020_tweets.csv'
df = pd.read_csv(csv)

df = df.sample(n=20, random_state=42)

textos = df['text']
textos_limpos = []

for t in textos:
    texto_traduzido = traduzir_textos(t)
    
    if texto_traduzido != None:
        textos_limpos.append(limpar_texto(texto_traduzido))
        
print(textos_limpos)

import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

nlp = spacy.load("en_core_web_sm")

tfidf = TfidfVectorizer()

transformado = tfidf.fit_transform(textos_limpos)

df = pd.DataFrame(transformado[ 0 ].T.todense(), index=tfidf.get_feature_names_out(), columns=[ "TF-IDF" ])
df = df.sort_values('TF-IDF' , ascending= False)

analises = {}

textos_ambiguos = []

for t in textos_limpos:
    blob = TextBlob(t)
    
    polarity = 0
    subject = 0
    
    for sentences in blob.sentences:
        polarity = sentences.sentiment.polarity
        subject = sentences.sentiment.subjectivity
    
    s = sia.polarity_scores(t)
    score = s['compound']
    
    score_textblob = ''
    score_sia = ''
    
    print(subject)
    
    if polarity > 0.2 and polarity < 0.2:
        textos_ambiguos.append(t)
        
        continue
    
    if polarity > 0.5:
        score_textblob = 'Positivo'
    elif polarity < -0.5:
        score_textblob = 'Negativo'
    else:
        score_textblob = 'Neutro'
    
    if score > 0.5:
        score_sia = 'Positivo'
    elif score < -0.5:
        score_sia = 'Negativo'
    else:
        score_sia = 'Neutro'
    
    analises[t] = [score_textblob, score_sia]
    
print(analises)

from transformers import pipeline

analisador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analises_llm = {}

for t in textos_ambiguos:
    analises_llm[t] = analisador(t)
    
print(analises_llm)