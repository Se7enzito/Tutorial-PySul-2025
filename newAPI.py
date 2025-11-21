"""
IMPORTANTE:

ESTE CÓDIGO NÃO FOI FINALIZADO, ELE ESTÁ AQUI NO GITHUB PARA QUEM QUISER BRINCAR MELHORANDO A DataAPI E ANALISE DE SENTIMENTOS
"""

import os
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from textblob import TextBlob
from langdetect import detect
from wordcloud import WordCloud, STOPWORDS
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from main import LogAPI, TextAPI

from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DataAPI():
    def __init__(self):
        self.PASTA_DATASETS = os.path.join(BASE_DIR, 'data', 'datasets')
        self.PASTA_BASE = os.path.join(BASE_DIR, 'data', 'datasets', 'treinado')
        self.PASTA_BASE_N = os.path.join(BASE_DIR, 'data', 'datasets', 'normal')
        self.PASTA_TRATADA = os.path.join(BASE_DIR, 'data', 'datasets', 'tratado')
        self.PASTA_BRUTO = os.path.join(BASE_DIR, 'data', 'datasets', 'bruto')
                
        self.DATASETS = {}
        self.COLUNAS = {}
        
        self.CONFIG = {
            'coluna_texto': None,
            'coluna_emocao': None
        }
        
        self.emocoes_map_ml = {
            1: 'Neutro',
            2: 'Felicidade',
            3: 'Raiva',
            4: 'Surpresa',
            5: 'Tristeza',
            6: 'Disgosto',
            7: 'Medo'
        }
        
        self.emocoes_map_n = {
            0: 'Neutro',
            1: 'Felicidade',
            2: 'Raiva',
            3: 'Medo',
            4: 'Surpresa',
            5: 'Tristeza',
            6: 'Nojo',
            7: 'Confiança'
        }
        
    def get_datasets(self, pasta: str) -> dict:
        try:
            datasets = {}
            contador = 0

            for raiz, subpastas, arquivos in os.walk(pasta):
                for arquivo in arquivos:
                    if arquivo.lower().endswith(('.csv', '.xlsx', '.json')):
                        contador += 1
                        caminho_relativo = os.path.relpath(os.path.join(raiz, arquivo), pasta)
                        datasets[contador] = caminho_relativo

            self.DATASETS = datasets

            if not datasets:
                print("Nenhum dataset encontrado nas pastas especificadas.")

            return datasets

        except Exception as e:
            print(f"Erro ao listar os datasets: {e}")
            return {}

    def get_dataset_path(self, pasta: str, dataset_id: int) -> str:
        if dataset_id in self.DATASETS:
            return os.path.join(pasta, self.DATASETS[dataset_id])
        else:
            print("ID do dataset inválido.")
            return ""
    
    def carregar_dataset(self, caminho: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(caminho)
            
            return df
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")
            
            return pd.DataFrame()
        
    def salvar_dataset(self, df: pd.DataFrame, arquivo: str) -> None:
        caminho = self.PASTA_TRATADA + "/" + arquivo
        
        try:
            df.to_csv(caminho, index=False)
            
            print(f"Dataset salvo em: {caminho}")
        except Exception as e:
            print(f"Erro ao salvar o dataset: {e}")

    def reduzir_dataset(self, df: pd.DataFrame, random: bool, limite: int) -> pd.DataFrame:
        if limite > 0 and limite < len(df) + 1:
            if random:
                return df.sample(n=limite, random_state=42)
            else:
                return df.head(limite)
        else:
            print("Limite inválido ou maior que o tamanho do dataset.")
            return df
        
    def get_colunas_df(self, df: pd.DataFrame) -> dict:
        colunas = {}
        contador = 0
        
        for coluna in df.columns.tolist():
            contador += 1
            
            colunas.update({contador: coluna})
            
        self.COLUNAS = colunas
        return colunas
        
    def get_dados_coluna(self, df: pd.DataFrame, id: int) -> list:
        if id in self.COLUNAS:
            coluna = self.COLUNAS[id]
            
            return df[coluna].tolist()
        else:
            print("ID da coluna inválido.")
            
            return []
        
    def get_nome_coluna_id(self, id: int) -> str:
        if id in self.COLUNAS:
            coluna = self.COLUNAS[id]
            
            return coluna
        else:
            print("ID da coluna inválido.")
            
            return []

    def definir_coluna(self, tipo: int, id: int) -> None:
        if id in self.COLUNAS:
            coluna = self.COLUNAS[id]
            
            if tipo == 1:
                self.CONFIG['coluna_texto'] = coluna
            elif tipo == 2:
                self.CONFIG['coluna_emocao'] = coluna
            else:
                print("Tipo inválido. Use 1 para texto e 2 para emoção.")
        else:
            print("ID da coluna inválido.")

    # Datasets Treinados LLM's
    def tratar_coluna_emocoes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.CONFIG['coluna_emocao'] is None:
            print("Coluna de emoção não definida.")
            return df

        coluna_emocao = self.CONFIG['coluna_emocao']

        tqdm.pandas(desc="Limpando emoções")
        df['emocoes_formatada'] = (
            df[coluna_emocao]
            .progress_apply(lambda x: re.sub(rf"^{coluna_emocao}: ?", "", str(x)))
            .apply(lambda x: re.sub(r'"', "", x))
            .apply(lambda x: int(x) if x.isdigit() else -1)
        )

        return df

    def tratar_coluna_texto(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.CONFIG['coluna_texto'] is None:
            print("Coluna de texto não definida.")
            return df

        coluna_texto = self.CONFIG['coluna_texto']

        tqdm.pandas(desc="Limpando textos")
        
        df['texto_formatado'] = (
            df[coluna_texto]
            .progress_apply(lambda x: re.sub(rf"^{coluna_texto}: ?", "", str(x)))
            .apply(lambda x: re.sub(r'"', "", x))
        )
        
        tqdm.pandas(desc="Salvando Texto")
        
        df['texto_limpo'] = df['texto_formatado'].progress_apply(TextAPI.limpar_texto)
        
        df_validos = df.dropna().copy()
        df_validos = df_validos[df_validos['texto_limpo'].astype(str).str.lower().ne("nan")]
        df_validos['texto_limpo'].astype(str).map(str.strip).tolist()

        return df_validos
    
    def mapear_emocoes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'emocoes_formatada' not in df.columns:
            print("Coluna 'emocoes_formatada' não encontrada.")
            return df

        tqdm.pandas(desc="Mapeando emoções")
        df['emocao_nome'] = df['emocoes_formatada'].progress_apply(
            lambda x: self.emocoes_map_ml.get(x, 'Desconhecido')
        )

        return df
        
    # Datasets Geral
    def mapear_emocoes_n(self, coluna: int, df: pd.DataFrame) -> pd.DataFrame:
        coluna = self.get_nome_coluna_id(coluna)
        
        if coluna not in df.columns:
            print(f"Coluna {coluna} não encontrada.")
            return df

        tqdm.pandas(desc="Mapeando emoções")
        df['emocao_formatada'] = df[coluna]
        df['emocao_nome'] = df[coluna].progress_apply(
            lambda x: self.emocoes_map_n.get(x, 'Desconhecido')
        )

        return df

    # Comandos Geral LLM's
    def tratar_dataset_llm(self) -> bool:
        datasets = self.get_datasets(self.PASTA_BASE)
        print("Datasets disponíveis:", datasets)

        numero = int(input("Digite o número do dataset que deseja carregar: "))
        caminho = self.get_dataset_path(self.PASTA_BASE, numero)
        df = self.carregar_dataset(caminho)

        print(self.get_colunas_df(df))

        coluna_texto = int(input("Digite o número da coluna que contém os textos: "))
        self.definir_coluna(1, coluna_texto)

        coluna_emocoes = int(input("Digite o número da coluna que contém as emoções: "))
        self.definir_coluna(2, coluna_emocoes)

        df = self.tratar_coluna_texto(df)
        df = self.tratar_coluna_emocoes(df)
        df = self.mapear_emocoes(df)

        arquivo = input('Digite o nome para salvar o arquivo: ')
        self.salvar_dataset(df, arquivo + ".csv")
        
        return True
    
    # Comandos Geral Normal
    def tratar_dataset_normal(self) -> bool:
        datasets = self.get_datasets(self.PASTA_BASE_N)
        print("Datasets disponíveis:", datasets)

        numero = int(input("Digite o número do dataset que deseja carregar: "))
        caminho = self.get_dataset_path(self.PASTA_BASE_N, numero)
        df = self.carregar_dataset(caminho)
        
        print(self.get_colunas_df(df))
        print(df.head(2))
        coluna_texto = int(input("Digite o número da coluna que contém os textos: "))
        self.definir_coluna(1, coluna_texto)

        coluna_emocoes = int(input("Digite o número da coluna que contém as emoções: "))
        self.definir_coluna(2, coluna_emocoes)
        
        df = self.tratar_coluna_emocoes(df)
        df = self.tratar_coluna_texto(df)
        df = self.mapear_emocoes_n(coluna_emocoes, df)

        arquivo = input('Digite o nome para salvar o arquivo: ')
        self.salvar_dataset(df, arquivo + ".csv")
        
        return True
        
    # Comandos Geral Bruto
    def tratar_texto_dataset_bruto(self) -> bool:
        datasets = self.get_datasets(self.PASTA_BRUTO)
        print("Datasets disponíveis:", datasets)

        numero = int(input("Digite o número do dataset que deseja carregar: "))
        caminho = self.get_dataset_path(self.PASTA_BRUTO, numero)
        df = self.carregar_dataset(caminho)
        
        print(self.get_colunas_df(df))
        print(df.head(2))
        coluna_texto = int(input("Digite o número da coluna que contém os textos: "))
        self.definir_coluna(1, coluna_texto)

        df = self.tratar_coluna_texto(df)
        
        df.to_csv(caminho, index=False)
        
        return True
    
    # Comandos Geral
    def get_comentarios_tratados(self) -> list:
        datasets = self.get_datasets(self.PASTA_DATASETS)
        print("Datasets disponíveis:", datasets)

        numero = int(input("Digite o número do dataset que deseja carregar: "))
        caminho = self.get_dataset_path(self.PASTA_DATASETS, numero)
        df = self.carregar_dataset(caminho)
        
        print(self.get_colunas_df(df))
        coluna_texto = int(input("Digite o número da coluna que contém os textos: "))
        self.definir_coluna(1, coluna_texto)
        
        return self.get_dados_coluna(df, coluna_texto)
    
    def get_comentarios_tratados_direto(self, dado: any) -> list:
        datasets = self.get_datasets(self.PASTA_DATASETS)
        datasets_invertido = {v: k for k, v in datasets.items()}
        
        try:
            dado = int(dado)
            print("[DEBUG] Dado é um número.")
            
            datasets_ids = list(datasets.keys())
            if dado in datasets_ids:
                dataset = self.get_dataset_path(self.PASTA_DATASETS, dado)
                
                df = self.carregar_dataset(dataset)
                id = dado
            else:
                return []
            
        except ValueError:
            print("[DEBUG] Dado é um arquivo.")
            
            datasets_nomes = list(datasets.values())
            if dado in datasets_nomes:
                dataset = self.get_dataset_path(self.PASTA_DATASETS, dado)
                
                df = self.carregar_dataset(dataset)
                id = datasets_invertido[dado]
            else:
                return []
        
        return self.get_dados_coluna(df, id)
    
    def get_dataset_tratado(self) -> pd.DataFrame:
        datasets = self.get_datasets(self.PASTA_DATASETS)
        print("Datasets disponíveis:", datasets)

        numero = int(input("Digite o número do dataset que deseja carregar: "))
        caminho = self.get_dataset_path(self.PASTA_DATASETS, numero)
        df = self.carregar_dataset(caminho)
        
        print(self.get_colunas_df(df))
        
        return df
    
    def get_dataset_tratado_direto(self, dado: any) -> pd.DataFrame:
        datasets = self.get_datasets(self.PASTA_DATASETS)
        datasets_invertido = {v: k for k, v in datasets.items()}
        
        try:
            dado = int(dado)
            print("[DEBUG] Dado é um número.")
            
            datasets_ids = list(datasets.keys())
            if dado in datasets_ids:
                dataset = self.get_dataset_path(self.PASTA_DATASETS, dado)
                
                df = self.carregar_dataset(dataset)
                id = dado
            else:
                return []
            
        except ValueError:
            print("[DEBUG] Dado é um arquivo.")
            
            datasets_nomes = list(datasets.values())
            if dado in datasets_nomes:
                dataset = self.get_dataset_path(self.PASTA_DATASETS, dado)
                
                df = self.carregar_dataset(dataset)
                id = datasets_invertido[dado]
            else:
                return []
            
        return df
    
class AnaliseSentimentos():
    def __init__(self):
        NLTK_PASTA = os.path.join(BASE_DIR, 'data', 'nltk_data')
    
        nltk.data.path.append(NLTK_PASTA)
        nltk.download('punkt', download_dir=NLTK_PASTA)
        nltk.download('vader_lexicon', download_dir=NLTK_PASTA)
        
        self.analyzer = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer()
        # self.modelo = pipeline("text-classification", model="bigscience/bloom")
        self.modelo = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.dataAPI = DataAPI()
        
    def traduzir_para_ingles(self, texto: str) -> str:
        try:
            idioma_detectado = detect(texto)

            if idioma_detectado == "en":
                return texto 
            else:
                return GoogleTranslator(source='auto', target='en').translate(texto)
        except Exception:
            return texto
    
    def interpretar_sentimento(self, resultado):
        try:
            label = resultado[0]['label']
            score = resultado[0]['score']

            estrelas = int(label.split()[0])

            if estrelas <= 2:
                sentimento = "Negativo"
            elif estrelas == 3:
                sentimento = "Neutro"
            else:
                sentimento = "Positivo"

            return sentimento, score
        
        except Exception as e:
            print(f"[ERRO] Falha ao interpretar sentimento: {e}")
            return "Desconhecido", 0.0

    def analise_df_bruto(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'texto_limpo' not in df.columns:
            return df

        textos = df['texto_limpo']

        try:
            resultados = self.modelo(textos)
        except Exception as e:
            print(f"[ERRO] Falha ao executar modelo: {e}")
            return df

        sentimentos, confiancas = [], []
        for texto, resultado in zip(textos, resultados):
            sentimento, confianca = self.interpretar_sentimento([resultado])
            sentimentos.append(sentimento)
            confiancas.append(confianca)
            print(f"{texto} → {sentimento} ({confianca:.2f})")

        df = df.assign(sentimento=sentimentos, confianca=confiancas)

        return df

    def analisar_sentimento(self, texto: str):
        if texto == 'N/A':
            return 'Neutro'
        
        texto_traduzido = self.traduzir_para_ingles(texto)
        vs = self.analyzer.polarity_scores(texto_traduzido)
        score = vs['compound']
        if score >= 0.05:
            return 'Positivo'
        elif score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutro'

    def analisar_sentimento_ml(self, texto: str):
        if texto == 'N/A':
            return 'Neutro'
        
        texto_traduzido = self.traduzir_para_ingles(texto)
        
        resultado = self.modelo(texto_traduzido)
        
        for texto, resultado in zip(texto, resultado):
            sentimento, confianca = self.interpretar_sentimento([resultado])
            
        return f'{sentimento} | {confianca}'

    def find_sentiment_polarity_textblob(self, texto):
        texto_traduzido = self.traduzir_para_ingles(texto)
        blob = TextBlob(texto_traduzido)
        polarity = 0
        
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            
        return polarity

    def find_sentiment_subjectivity_textblob(self, texto):
        texto_traduzido = self.traduzir_para_ingles(texto)
        blob = TextBlob(texto_traduzido)
        subjectivity = 0
        
        for sentence in blob.sentences:
            subjectivity += sentence.sentiment.subjectivity
            
        return subjectivity

class App():
    def __init__(self):
        self.logAPI = LogAPI()
        self.analiseSentimentos = AnaliseSentimentos()
        self.dataAPI = DataAPI()
        
        self.LIMIT = 100
        self.RANDOM_DATA = True
        
    # === Código Principal Aplicativo ===
    def run(self) -> None:
        condicao = True
        
        while condicao:
            n = int(input("Digite uma opção (1 - Tratar Dataset; 2 - Reconhecer Emoções; 3 - Sair): "))
            
            if (n == 1):
                self.tratar_dataset()
            elif (n == 2):
                self.rodar_dataset()
            else:
                condicao = False
            
    def tratar_dataset(self) -> None:
        pass
    
    def rodar_dataset(self) -> None:
        tqdm.pandas()
        
        logger = self.logAPI.setup_logging()

        df = self.dataAPI.get_dataset_tratado()
        
        if self.LIMIT != -1:
            if self.RANDOM_DATA:
                df = df.head(self.LIMIT)
            else:
                df = df.sample(n=self.LIMIT, random_state=42)
        
        df['sentimento'] = df['texto_formatado'].progress_apply(self.analiseSentimentos.analisar_sentimento)
        
        self.show_wordcloud(df['texto_limpo'], title = 'Prevalent words in text')
        
        self.plot_sentiment(df, 'sentimento', 'Text')
        
        df['text_sentiment_polarity'] = df['texto_formatado'].progress_apply(lambda x: self.analiseSentimentos.find_sentiment_polarity_textblob(x))
        df['text_sentiment_subjectivity'] = df['texto_formatado'].progress_apply(lambda x: self.analiseSentimentos.find_sentiment_subjectivity_textblob(x))
        
        self.plot_sentiment_textblob(df, 'text', 'Text')
        
    # === Código Limpeza do Texto ===
    def limpar_texto(self, texto: str) -> str:
        if not isinstance(texto, str):
            return "N/A"
        
        texto = texto.lower()
        
        texto = re.sub(r"http\\S+", "", texto)
        texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
        texto = re.sub(r"[^a-zà-ÿ# ]", "", texto)
        
        return texto

    # === Código Conferência Dados Faltantes ===
    def missing_data(self, data):
        total = data.isnull().sum()
        percent = (data.isnull().sum()/data.isnull().count()*100)
        tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        types = []
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            types.append(dtype)
            
        tt['Types'] = types
        
        return(np.transpose(tt))

    # === Código Nuvem de Palavras ===
    def show_wordcloud(self, data, title=""):
        text = " ".join(t for t in data.dropna().astype(str))

        if not text.strip():
            print(f"[AVISO] Nenhum texto disponível para gerar wordcloud: {title}")
            return

        stopwords = set(STOPWORDS)
        stopwords.update([
            "t", "co", "https", "amp", "U", 
            "Olympics", "Tokyo2020", "TokyoOlympics", 
            "Olympic", "Olympics Tokyo2020", "Tokyo2020 Olympics"
        ])

        wordcloud = WordCloud(
            stopwords=stopwords, 
            scale=4, 
            max_font_size=50, 
            max_words=500,
            background_color="black"
        ).generate(text)

        fig = plt.figure(1, figsize=(16, 16))
        plt.axis('off')
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.show()

    # === Código Gráfico de Sentimetnos ===
    def plot_sentiment(self, df, feature, title):
        counts = df[feature].value_counts()
        percent = counts/sum(counts)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        counts.plot(kind='bar', ax=ax1, color='green')
        percent.plot(kind='bar', ax=ax2, color='blue')
        ax1.set_ylabel(f'Counts : {title} sentiments', size=12)
        ax2.set_ylabel(f'Percentage : {title} sentiments', size=12)
        plt.suptitle(f"Sentiment analysis: {title}")
        plt.tight_layout()
        plt.show()

    # === Código Gráfico de Frequência ===
    def plot_sentiment_textblob(self, df, feature, title):
        polarity = df[feature+'_sentiment_polarity']
        subjectivity = df[feature+'_sentiment_subjectivity']

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        polarity.plot(kind='kde', ax=ax1, color='magenta')
        subjectivity.plot(kind='kde', ax=ax2, color='green')
        ax1.set_ylabel(f'Sentiment polarity : {title}', size=12)
        ax2.set_ylabel(f'Sentiment subjectivity: {title}', size=12)
        plt.suptitle(f"Sentiment analysis (polarity & subjectivity): {title}")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = App()
    app.run()
    
    # TODO: Conferir mais tarde
    # modelo = pipeline("text-classification", model="bigscience/bloom")
    # print(modelo("Achei o atendimento horrível e desrespeitoso."))
    # print(modelo("I hate everyone."))