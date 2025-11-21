from langdetect import detect
from deep_translator import GoogleTranslator

texto = "A inteligência artificial está mudando o mundo."
idioma = detect(texto)
print(idioma)  # Saída: 'pt'

traduzido = GoogleTranslator(source=idioma, target='en').translate(texto)
print(traduzido)