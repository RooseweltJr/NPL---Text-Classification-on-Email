import streamlit as st
import Dataset
import modelos_baseline
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

            data.to_csv('Base_tratada\\database.csv',index=False)
            
            st.subheader("Visualização")
            st.markdown(f"Com tratamento finalizado, podemos utilizar algumas ferramentas para visualizar melhor o contéudo dos emails de cada cateoria. ")

            cat = st.selectbox(f'Selecione a categoria:', categorias)
                  
            all_text = " ".join(data[data.Categoria == cat].Conteudo_filtrado) 

            Dataset.plot_wordcloud_and_top10(all_text, cat)
                    
        case "Modelo baseline":
            st.subheader("Modelos baseline")
            st.write(""" Com os dados tratados, vamos treinar alguns modelos para 
                     testar seu desempenho com processamento de linguagem natural. Mas antes, vamos importar nossa base, bem como
                     codifica-la e ajusta-la  para o nosso treinamento""")
            
            data = modelos_baseline.importar_base_tratadas()

            df,vetor_treinamento, = modelos_baseline.codificar_categoria(data)

            st.dataframe(df.head(10))

            st.write(""" Para nosso projeto, vamos utilizar 4 *Classifiers* de aprendizado diferente:  """)
            
            
            st.markdown("""
                        - Máquina de vetores de suporte (SVC)
                        - Regressão logística (LR)
                        - K-ésimo Vizinho mais Próximo (KNN)
                        - Árvore de Decisão (DT)""")
            
            st.markdown("### Treinamento")
            
            test_size = (int(st.slider('Percentual de "Test Size" (%):', min_value=10, max_value=90, step=5)))/100
            n_split = (int(st.slider('Quantidade de "Folds para Validação Cruzada" ', min_value=2, max_value=5, step=1)))
            
            # Botão para iniciar o processamento
            if st.button("Iniciar"):
                X_train, X_test, y_train, y_test = modelos_baseline.dividir_teste_treino(vetor_treinamento,df,test_size)

                st.markdown(""" Para analisar nossos modelos, vamos utilizar uma validação cruzada para analisar a acurácia do modelo,
                            bem como AUC (Área Sob a Curva). O primeiro permite analisar o percentual de acerto, enquanto o segundo podemos visualizar
                            a relação da taxa de verdadeiros positivos (TPR ou sensibilidade) e a taxa de falsos positivos (FPR) em diferentes limiares de classificação""")
                
                st.markdown("#### Acurácia:")
                st.markdown("Isso pode levar alguns minutos")

                with st.spinner('Treinando modelo para calcular a acurácia...'):
                    modelos_baseline.treinamento_acuracia(X_train, y_train, n_split)

                st.markdown("#### AUC:")
                st.markdown("Isso pode levar alguns minutos")
                with st.spinner('Treinando modelo para calcular a AUC...'):
                    modelos_baseline.treinamento_AUC(X_train, y_train, n_split)

                st.markdown("### Teste")
                st.markdown("Isso pode levar alguns minutos")
                with st.spinner('Testando o modelo...'):
                    modelos_baseline.teste(X_test, y_test, n_split)

            

if __name__=='__main__':
    main()