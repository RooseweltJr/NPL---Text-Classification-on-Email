from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score,  confusion_matrix
import pandas as pd
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_predict
import plotly.graph_objects as go
from plotly.subplots import make_subplots



df=pd.read_csv('Base_tratada\\database.csv')
df.dropna(subset=['Conteudo_filtrado'],inplace=True)


classifiers = {
    "SVC": SVC(probability=True),
    "LR": LogisticRegression(max_iter= 500),
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier()
}

def matriz_confusao(y_test,log_reg_pred,knn_pred,svc_pred,tree_pred):
    # Calcule as matrizes de confusão
    log_reg_cf = confusion_matrix(y_test, log_reg_pred)
    knn_cf = confusion_matrix(y_test, knn_pred)
    svc_cf = confusion_matrix(y_test, svc_pred)
    tree_cf = confusion_matrix(y_test, tree_pred)

    # Labels para os heatmaps
    x= ['FP','TN']
    y= ['TP','FN']
    colorscale = [[0, 'rgb(255,0,0)'], [1, 'rgb(0,0,255)']]

    # Criação dos subplots
    conf_matrix = make_subplots(rows=2, cols=2, subplot_titles=("Logistic Regression", "K-Nearest Neighbor", "Support Vector Classifier", "Decision Tree Classifier"))

    # Adicionando os heatmaps
    conf_matrix.add_trace(go.Heatmap(z=log_reg_cf, x=x, y=y, colorscale=colorscale, text=log_reg_cf, texttemplate="%{text}", textfont={"size":20}), row=1, col=1)
    conf_matrix.add_trace(go.Heatmap(z=knn_cf, x=x, y=y, colorscale=colorscale, text=knn_cf, texttemplate="%{text}", textfont={"size":20}, showscale=False), row=1, col=2)
    conf_matrix.add_trace(go.Heatmap(z=svc_cf, x=x, y=y, colorscale=colorscale, text=svc_cf, texttemplate="%{text}", textfont={"size":20}, showscale=False), row=2, col=1)
    conf_matrix.add_trace(go.Heatmap(z=tree_cf, x=x, y=y, colorscale=colorscale, text=tree_cf, texttemplate="%{text}", textfont={"size":20}, showscale=False), row=2, col=2)

    # Atualiza os eixos e mostra o gráfico
    conf_matrix.update_yaxes(autorange="reversed")
    conf_matrix.update_layout(title_text='Confusion Matrices for Classifiers')
    conf_matrix.show()

def codificar_categoria(df:pd.DataFrame)->pd.DataFrame:
    
    # Codificação da categoria
    le = LabelEncoder()
    df['Classification'] = le.fit_transform(df['Categoria'])

    # Determina o mínimo número de ocorrências de uma categoria
    minimum_count = df['Categoria'].value_counts().min()

    # Shuffle dos dados
    df = df.sample(frac=1)

    # Seleção das amostras balanceadas por categoria
    politics_df = df.loc[df['Categoria'] == 'Politics'][:minimum_count]
    crime_df = df.loc[df['Categoria'] == 'Crime'][:minimum_count]
    entertainment_df = df.loc[df['Categoria'] == 'Entertainment'][:minimum_count]
    science_df = df.loc[df['Categoria'] == 'Science'][:minimum_count]
    normal_dist_df = pd.concat([politics_df, crime_df, entertainment_df, science_df])

    # Vetorização
    tfidf = TfidfVectorizer(lowercase=False)
    train_vec = tfidf.fit_transform(normal_dist_df['Conteudo_filtrado'])
    
    print(train_vec.shape)

    return normal_dist_df,train_vec

def dividir_teste_treino(vetor_treinamento,df_dist_normal,teste_size):
    #separar:
    X_train, X_test, y_train, y_test = train_test_split(vetor_treinamento, df_dist_normal['Classification'], stratify=df_dist_normal['Classification'], test_size=teste_size)

    print("x_train", X_train.shape, type(X_train))
    print("X_test", X_test.shape, type(X_test))
    print("y_train", y_train.shape, type(y_train))
    print("y_test", y_test.shape, type(y_test))

    return X_train, X_test, y_train, y_test

def treinamento_acuracia( X_train, y_train,n_split:int = 5):
    #TODO criar um loading, pois demora uns 2 a 3 min
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=n_split)
        print("Classifiers: ", key, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score.")

def treinamento_AUC(X_train, y_train,n_split:int = 5):
    
    method = 'predict_proba'
    n_split = 5
    log_reg_pred = cross_val_predict(classifiers["LR"], X_train, y_train, cv = n_split, method = method)
    knn_pred = cross_val_predict(classifiers["KNN"], X_train, y_train, cv = n_split, method = method)
    svc_pred = cross_val_predict(classifiers["SVC"], X_train, y_train, cv = n_split, method = method)
    tree_pred = cross_val_predict(classifiers["DT"], X_train, y_train, cv = n_split, method = method)

    
    log_reg_auc_score = roc_auc_score(y_train, log_reg_pred, multi_class = 'ovr')
    knn_auc_score = roc_auc_score(y_train, knn_pred, multi_class = 'ovr')
    svc_auc_score = roc_auc_score(y_train, svc_pred, multi_class = 'ovr')
    tree_auc_score = roc_auc_score(y_train, tree_pred, multi_class = 'ovr')

    print('Support Vector Classifier AUC: ', svc_auc_score)
    print('Logistic Regression AUC: ', log_reg_auc_score)
    print('KNears Neighbors AUC: ', knn_auc_score)
    print('Decision Tree Classifier AUC: ', tree_auc_score)

def teste(X_test, y_test,n_split:int = 5):
    method = 'predict'
    
    log_reg_pred = cross_val_predict(classifiers["LR"], X_test, y_test, cv = n_split, method = method)
    knn_pred = cross_val_predict(classifiers["KNN"], X_test, y_test, cv = n_split, method = method)
    svc_pred = cross_val_predict(classifiers["SVC"], X_test, y_test, cv = n_split, method = method)
    tree_pred = cross_val_predict(classifiers["DT"], X_test, y_test, cv = n_split, method = method)

    matriz_confusao(y_test,log_reg_pred,knn_pred,svc_pred,tree_pred)
    
