import spacy

nlp = spacy.load("pt_core_news_sm")
doc = nlp("O ChatGPT foi desenvolvido pela OpenAI.")

for ent in doc.ents:
    print(ent.text, ent.label_)