from transformers import pipeline

analisador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def interpretar_sentimento(resultado):
    label = resultado[0]['label']
    score = resultado[0]['score']

    estrelas = int(label.split()[0])
    if estrelas <= 2:
        sentimento = "Negativo 😠"
    elif estrelas == 3:
        sentimento = "Neutro 😐"
    else:
        sentimento = "Positivo 😄"

    return sentimento, score

textos = [
    "Esse produto é incrível!",
    "Não gostei do atendimento.",
    "Foi uma experiência normal.",
    "I hate this local."
]

resultados = analisador(textos)

print(resultados)

for texto, resultado in zip(textos, resultados):
    sentimento, confianca = interpretar_sentimento([resultado])
    print(f"{texto} → {sentimento} ({confianca:.2f})")
