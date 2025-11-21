from langdetect import detect
from deep_translator import GoogleTranslator

texto = "A inteligência artificial está mudando o mundo."
idioma = detect(texto)
print(idioma)  # Saída: 'pt'

texto = "Artificial intelligence is changing the world."
traduzido = GoogleTranslator(source='en', target='pt').translate(texto)
print(traduzido)