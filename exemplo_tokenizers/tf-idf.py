import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

sents = [ 'coronavírus é uma doença altamente infecciosa' ,
    'coronavírus afeta mais os idosos' ,
    'idosos correm alto risco devido a esta doença' ]

tfidf = TfidfVectorizer()

transformado = tfidf.fit_transform(sents)

df = pd.DataFrame(transformado[ 0 ].T.todense(),
    	index=tfidf.get_feature_names_out(), columns=[ "TF-IDF" ])
df = df.sort_values( 'TF-IDF' , ascending= False )

print(df)