#------------------------------------------ Bibliotecas ---------------------------------------------------------------#
####################################################################################################################
#-------------------------------------Visulaização e Dados ---------------------------------------------------------#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#---------------------------------------Processamento de texto------------------------------------------------------------#
import regex
import nltk

nltk.data.path.append(r'C:\Users\roose\AppData\Roaming\nltk_data')
# # Baixar os pacotes necessários (se necessário)
nltk.download('punkt', download_dir=r'C:\Users\roose\AppData\Roaming\nltk_data')
nltk.download('stopwords', download_dir=r'C:\Users\roose\AppData\Roaming\nltk_data')

from wordcloud import WordCloud
from nltk.corpus import stopwords, words
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer



def tratar_conteudo_stop_words(data:pd.DataFrame):

    stop = stopwords.words('english')

    for punct in punctuation:
        stop.append(punct)
    
    data["filtered_text"] = data.Conteudo.apply(lambda x : filter_text(x, stop)) 
    return data


def filter_text(text, stop_words):
    word_tokens = WordPunctTokenizer().tokenize(text.lower())
    wordnet_lemmatizer = WordNetLemmatizer()

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and len(w) > 3]
    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 
    return " ".join(filtered_text)

def plot_wordcloud_and_top10(all_text, title):
    # Criação da DataFrame para as palavras e cálculo das 10 palavras mais frequentes
    count = pd.DataFrame(all_text.split(), columns=['words'])
    top_10 = count['words'].value_counts().nlargest(10).reset_index()
    top_10.columns = ['words', 'count']

    # Configuração da figura com 2 subplots
    plt.figure(figsize=(20, 20))

    # Plot do gráfico de barras das 10 palavras mais frequentes
    plt.subplot(2, 1, 1)
    sns.barplot(x=top_10['words'], y=top_10['count'], palette=sns.color_palette("mako"))
    plt.title('Top 10 Palavras Mais Frequentes', fontsize=24)
    plt.xlabel('Palavras', fontsize=18)
    plt.ylabel('Contagem', fontsize=18)
    plt.xticks(rotation=45)

    # Criação e plot da nuvem de palavras
    plt.subplot(2, 1, 2)
    cloud = WordCloud(width=1920, height=1080, max_font_size=200, max_words=300, background_color="white").generate(words)
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off")
    plt.title(title, fontsize=24)

    plt.tight_layout()
    plt.show()

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
    print(data.shape)
    data.columns = ['Conteudo', 'Categoria','Id']
    return data

####### Importando dados #######
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
df.to_csv('combine.csv',index=False)
df['Categoria'].value_counts().plot.bar()


# Retirar emails duplicados:
data = retirar_duplicatas(df)

# tratar o conteudo e stopwords
data = tratar_conteudo_stop_words(data)

#Visualizando resultados
all_text = " ".join(data[data.Categoria == "Crime"].filtered_text) 

plot_wordcloud_and_top10(all_text, 'Crime')