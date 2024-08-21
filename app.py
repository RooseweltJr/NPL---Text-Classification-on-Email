import streamlit as st
import Dataset
import matplotlib.pyplot as plt

def main():
    """Aplicação Visual"""
    st.title("Projeto de Classificação de Texto com NPL e ML")
    st.subheader("Tópicos em Inteligencia Artificial (ELE-606)")


    #Menu
    menu = ["Database","Modelo baseline","LMM"]
    opcoes  = st.sidebar.selectbox("Selecione uma das opções",menu)

    #importar dataframe:
    data = Dataset.importar_dados()

    categorias = ['Crime', 'Entertainment', 'Politics', 'Science']
    
    #Fullwidth
    
    # Setando as abas
    match opcoes:
        case "Database":
            st.subheader("Tratamento")

            st.markdown(f"""Os arquivos para nosso estudo estão agrupados em 4 grupos: *Crime*, *Entertainment*, *Politics*, *Science*. 
                        Separados em pastas com seus repectivos nomes. Primeiro passo do tratamento é juntar todos eles em um único `dataframe`,
                        com a **Categoria** e **Conteudo** discriminados:""")
            st.dataframe(data.head(10))

            st.markdown(f"""Com os dados combinados, vamos ver a distribuição dos emails em cada Categoria: """)
            Dataset.plotar_barras(data)
            
            st.markdown(f"""Porém, essa quantidade pode estar contaminada com emails repetidos, logo vé necessário eliminar as duplicatas com nossa função
                        `retirar_duplicatas()`. Agora, essa nossa distribuição:""")
            data = Dataset.retirar_duplicatas(data)
            Dataset.plotar_barras(data)
            
            st.markdown(f"Para finalizar nosso tratamento, é preciso retirar as partes desnecessárias para nosso modelo, em suma, as pontuações e as *Stopwords*'.")
          
            data = Dataset.tratar_conteudo_stop_words(data)
            st.dataframe(data.head(10))

            st.subheader("Visualização")
            st.markdown(f"Com tratamento finalizado, podemos utilizar algumas ferramentas para visualizar melhor o contéudo dos emails de cada cateoria. ")

            cat = st.selectbox(f'Selecione a categoria:', categorias)
                  
            all_text = " ".join(data[data.Categoria == cat].Conteudo_filtrado) 

            Dataset.plot_wordcloud_and_top10(all_text, cat)
                    
        # case "Modelo":
        #     st.header("Modelo")
        #     st.write("Nessa aba vamos construir nosso modelo, treina-lo e, por fim, utiliza-lo para fazer predições com ele")
        #     modelo.modelo(data)
   

    
if __name__=='__main__':
    main()