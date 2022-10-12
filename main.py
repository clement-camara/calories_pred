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

import hydralit_components as hc


df = pd.read_csv('df_lasso.csv')
df2 = pd.read_csv('df_dum.csv')

def app_page(st, **state):
    st.write("https://dietappsimplonbordeaux.herokuapp.com/")


def monitoring(st, **state):
    st.title('Monitorer des modèles de machine learning')
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
            L = st.sidebar.slider("alpha", 1, 10)
            params["alpha"] = L
        elif clf_name == "RandomForestRegressor":
            RF = st.sidebar.slider("n_estimators", 1, 50)
            params["n_estimators"] = RF
            criterion = st.sidebar.selectbox("criterion", ('squared_error', 'absolute_error', 'poisson'))
            params["criterion"] = criterion
            max_depth = st.sidebar.slider("max_depth", 0, 10)
            if max_depth == 0:
                max_depth = None
            params["max_depth"] = max_depth
            max_leaf_nodes = st.sidebar.slider("max_leaf_nodes", 2, 50)
            if max_leaf_nodes == 2:
                max_leaf_nodes = None
            params["max_leaf_nodes"] = max_leaf_nodes


        else:
            positive = st.sidebar.selectbox("positive", (True, False))
            params["positive"] = positive
            copy_X = st.sidebar.selectbox("copy_X", (True, False))
            params["copy_X"] = copy_X

        return params

    params = add_parameter(regressior_name)

    def get_regressor(clf_name, params):
        if clf_name == "Lasso":
            clf = linear_model.Lasso(alpha=params["alpha"])
        elif clf_name == "RandomForestRegressor":
            clf = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                criterion=params["criterion"],
                max_depth=params["max_depth"],
                max_leaf_nodes=params["max_leaf_nodes"])
        else:
            clf = LinearRegression(positive=params["positive"],copy_X=params["copy_X"])
        return clf

    clf = get_regressor(regressior_name, params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Score = r2_score(y_test, y_pred)

    st.write(f"regressor = {regressior_name}")
    st.write(f"R2 = {Score}")


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
