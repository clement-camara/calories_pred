from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import IPython, ipywidgets
import streamlit as st
from streamlit_multipage import MultiPage
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


#st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('data/df_lasso.csv')
df2 = pd.read_csv('data/df_dum.csv')

@st.cache
def fetch_and_clean_data(data):
    # Fetch data from URL here, and then clean it up.
    return data

fetch_and_clean_data(df)
fetch_and_clean_data(df2)

# fonction en cache pour la page avec le lien de l'app
def app_page(st, **state):
    st.title("Retrouver mon application sur ce lien :")
    st.write("https://dietappsimplonbordeaux.herokuapp.com/")

# fonction pour la page de monitoring
def monitoring(st, **state):
    st.title('Monitorer mes modèles de machine learning')
    st.write("""explorer different modèles afin de trouver lequel est le meilleur!""")
    dataset_name = st.sidebar.selectbox("Sélectionner le jeu de données", ('df', 'df encode'))
    st.write(dataset_name)
    regressior_name = st.sidebar.selectbox("Selection du modèle de régression",
                                           ("Lasso", "RandomForestRegressor", "regression linéaire"))

    def get_dataset(dataset_name):
        if dataset_name == 'df':
            data = df
        elif dataset_name == 'df encode':
            data = df2
        # Définition de la cible et des features
        if dataset_name == 'df':
            X = data.drop(['user_id', 'gender', 'calorie'], axis=1)
            y = data.calorie
            return X, y
        else:
            X = data.drop(['user_id', 'calorie'], axis=1)
            y = data.calorie
            return X, y

    X, y = get_dataset(dataset_name)
    MultiPage.save({"total": X}, namespaces=["Features"])
    st.write("Dataset", X)

    def add_parameter(clf_name):
        params = {}
        if clf_name == "Lasso":
            L = st.sidebar.slider("alpha", 0, 10)
            params["alpha"] = L
        elif clf_name == "RandomForestRegressor":
            params = None
            # max_depth = st.sidebar.slider("max_depth", 1, 11)
            # if max_depth == 11:
            #     max_depth = None
            # params["max_depth"] = max_depth
            # max_leaf_nodes = st.sidebar.slider("max_leaf_nodes", 2, 51)
            # if max_leaf_nodes == 51:
            #     max_leaf_nodes = None
            # params["max_leaf_nodes"] = max_leaf_nodes

        else:
            positive = st.sidebar.selectbox("positive", (True, False))
            params["positive"] = positive
        return params
    params = add_parameter(regressior_name)

    def get_regressor(clf_name, params):
        if clf_name == "Lasso":
            clf = linear_model.Lasso(alpha=params["alpha"])
        elif clf_name == "RandomForestRegressor":
            clf = RandomForestRegressor()
        else:
            clf = LinearRegression(positive=params["positive"] )
        return clf
    clf = get_regressor(regressior_name, params)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Score = r2_score(y_test, y_pred)

    st.write(f"regressor = {regressior_name}")
    st.write(f"R2 = {Score}")

# fonction pour la page de visualisation qui est aussi en cache
def visualisation_page(st, **state):
    MultiPage.save({"total": df}, namespaces=["df"])
    profile = ProfileReport(df,
                            title="Diet Data",
                            dataset={
                                "description": "This profiling report was generated for Simplon certification ",
                                "copyright_holder": "Camara Clement",
                                "copyright_year": "2022"
                            })
    st_profile_report(profile)


app = MultiPage()
app.navbar_style = "SelectBox"
app.st = st
app.navbar_name = "Menu"
app.add_app("Diet app Page", app_page)
app.add_app("Monitoring Page", monitoring)
app.add_app("Visualisation Page", visualisation_page)
app.run()

# st.set_option('deprecation.showPyplotGlobalUse', False)
