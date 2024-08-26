#------------------------------------------ Bibliotecas ---------------------------------------------------------------#
####################################################################################################################
#-------------------------------------Visulaização e Dados ---------------------------------------------------------#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

#---------------------------------------Processamento de texto------------------------------------------------------------#
import regex
import nltk

nltk.data.path.append(r'C:\Users\roose\AppData\Roaming\nltk_data')
# # Baixar os pacotes necessários (se necessário)
nltk.download('punkt', download_dir=r'C:\Users\roose\AppData\Roaming\nltk_data')
nltk.download('stopwords', download_dir=r'C:\Users\roose\AppData\Roaming\nltk_data')

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from string import punctuation
from nltk.stem import WordNetLemmatizer



def tratar_conteudo_stop_words(data:pd.DataFrame):

    stop = stopwords.words('english')

    for punct in punctuation:
        stop.append(punct)
    
    data["Conteudo_filtrado"] = data.Conteudo.apply(lambda x : filter_text(x, stop)) 
    return data[['Conteudo_filtrado', 'Categoria','Id','Conteudo']]

def plotar_barras(data:pd.DataFrame):
    # Gera o gráfico de barras
        fig, ax = plt.subplots()
        data['Categoria'].value_counts().plot.bar(ax=ax)

        # Exibe o gráfico no Streamlit
        st.pyplot(fig)
def filter_text(text, stop_words):
    word_tokens = WordPunctTokenizer().tokenize(text.lower())
    wordnet_lemmatizer = WordNetLemmatizer()

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and len(w) > 3]
    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 
    return " ".join(filtered_text)

def plot_wordcloud_and_top10(all_text: str, title: str) -> None:
    # Criação da DataFrame para as palavras e cálculo das 10 palavras mais frequentes
    count = pd.DataFrame(all_text.split(), columns=['words'])
    top_10 = count['words'].value_counts().nlargest(10).reset_index()
    top_10.columns = ['words', 'count']

    # Configuração da figura com 2 subplots lado a lado
    plt.figure(figsize=(20, 10))
    plt.suptitle(title, fontsize=22, y=1.02)

    # Plot do gráfico de barras das 10 palavras mais frequentes
    plt.subplot(1, 2, 2)
    sns.barplot(x=top_10['words'], y=top_10['count'], palette=sns.color_palette("mako"))
    plt.title('Ranking 10 Palavras ', fontsize=16)
    plt.ylabel('Qtd', fontsize=14)
    plt.xticks(rotation=45)

    # Criação e plot da nuvem de palavras
    plt.subplot(1, 2, 1)
    cloud = WordCloud(width=800, height=800, max_font_size=200, max_words=300, background_color="white").generate(all_text)
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off")
    plt.title('Nuvem de palavras ', fontsize=16)
   
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

def retirar_duplicatas(df:pd.DataFrame):
    
    unique = list(df.Conteudo.unique())
    dic = dict(df)

    uni = {}
    i = 0
    for k in range(len(list(dic['Conteudo']))):
        if dic['Conteudo'][k] in unique:
            uni[i] = [dic['Conteudo'][k], dic['Categoria'][k],dic['ID'][k]]
            unique.remove(dic['Conteudo'][k])
            i += 1


    data = pd.DataFrame(uni).T
    data.columns = ['Conteudo', 'Categoria','Id']
    return data


def importar_dados():
    names = []

    caminho_atual = os.getcwd()

    base = os.path.join(caminho_atual, 'Data')

    folders = os.listdir(base)
    data = []

    # Ler os arquivos e juntar em um arquivo só
    for folder in folders:
        files = os.listdir(os.path.join(base, folder))
        for file in files:
            try:
                if file == 'combine.csv':
                    pass
                with open(os.path.join(base, folder, file), encoding='utf-8') as f:
                    contents = " ".join(f.readlines())
                    data.append([file.split(".")[0], folder, contents])
            except Exception as e:
                pass

    df = pd.DataFrame(data, columns=['ID', 'Categoria', 'Conteudo'])
    
    return df

def data_set(visualizar:bool=False):
    ####### Importando dados #######
    
    df=importar_dados()

    
    # Retirar emails duplicados:
    data = retirar_duplicatas(df)
    print(data['Categoria'].unique())

    # tratar o conteudo e stopwords
    data = tratar_conteudo_stop_words(data)

    if visualizar== True:

        #Visualizando resultados
        all_text = " ".join(data[data.Categoria == "Crime"].filtered_text)
        plot_wordcloud_and_top10(all_text, 'Crime')

    return data

if __name__=='__main__':
   df = data_set()
   df.to_csv('Base_tratada\\database.csv',index=False)