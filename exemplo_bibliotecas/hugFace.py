from transformers import pipeline

analisador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

print(analisador("Esse produto é incrível!"))