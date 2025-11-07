from sklearn.feature_extraction.text import CountVectorizer

sents = ['coronavírus é uma doença altamente infecciosa' ,
    'coronavírus afeta mais os idosos' ,
    'idosos correm alto risco devido a esta doença']
    
cv = CountVectorizer()

X = cv.fit_transform(sents)
X = X.toarray()

print(X)
print(sorted(cv.vocabulary_.keys()))

"""
- Você pode ver que cada linha é a representação vetorial associada às respectivas frases em 'sents'.
- O comprimento de cada vetor é igual ao comprimento do vocabulário.
- Cada elemento da lista representa a frequência com que a palavra associada está presente no vocabulário classificado.

No exemplo acima, consideramos apenas palavras individuais como características visíveis nas chaves do vocabulário, ou seja, trata-se de uma representação unigrama. Isso pode ser ajustado para considerar características n-gramas.

Suponhamos que desejemos considerar uma representação em bigrama da nossa entrada. Isso pode ser feito simplesmente alterando o argumento padrão ao instanciar o objeto CountVectorizer:
"""

cv = CountVectorizer(ngram_range=(2, 2))

X = cv.fit_transform(sents)
X = X.toarray()

print(X)
print(sorted(cv.vocabulary_.keys()))

"""
Assim, podemos manipular as características da maneira que quisermos. Na verdade, também podemos combinar unigramas, bigramas, trigramas e muito mais para formar o espaço de características.

Embora tenhamos usado o sklearn para construir um modelo Bag of Words aqui, ele pode ser implementado de diversas maneiras, com bibliotecas como Keras, Gensim e outras. Você também pode escrever sua própria implementação de Bag of Words com bastante facilidade.

Esta é uma técnica de codificação de texto simples, porém eficaz, que pode realizar a tarefa diversas vezes.
"""