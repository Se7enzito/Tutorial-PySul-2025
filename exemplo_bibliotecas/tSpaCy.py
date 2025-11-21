import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("O ChatGPT foi desenvolvido pela OpenAI.")

for ent in doc.ents:
    print(ent.text, ent.label_)