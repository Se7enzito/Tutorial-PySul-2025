import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

texto = "Eu adoro aprender Python, é muito divertido!"
resultado = sia.polarity_scores(texto)
print(resultado)