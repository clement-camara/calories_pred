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
import pickle
import mlflow

mlflow.set_tracking_uri('/Users/marinelafargue/Desktop/projet calorie/MLFlow/mlruns')  # set up connection
mlflow.set_experiment('test-experiment')  # set the experiment
mlflow.sklearn.autolog()

# st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('data/df_lasso.csv')
df2 = pd.read_csv('data/df_dum.csv')
df3 = pd.read_csv('data/best_df.csv')
df4 = pd.read_csv('data/best_df_with_age.csv')


@st.cache
def fetch_and_clean_data(data):
    # Fetch data from URL here, and then clean it up.
    return data

fetch_and_clean_data(df)
fetch_and_clean_data(df2)
fetch_and_clean_data(df3)
fetch_and_clean_data(df4)

# fonction en cache pour la page avec le lien de l'app
def app_page(st, **state):
    st.title("Retrouver l'application web flask sur ce lien :")
    st.write("https://dietappsimplonbordeaux.herokuapp.com/")

    st.header('3 features')

    st.text('Combien de calories je vais dépenser ?')
    duration = st.slider("Durée d'exercice en minutes", 1, 1, 30)
    age = st.slider("Age", 20, 20, 79)
    # Taille = st.slider("Taille", 120, 120, 222)
    # Poids = st.slider("Poids", 30, 30, 130)
    # Taille_metre = Taille / 100
    # #IMC_PREP = (Poids / Taille_metre ** 2)
    IMC = st.slider('IMC', 20, 20, 30)

    # load saved model
    def predict(data):
        with open('MLFlow/mlruns/0/b8e7f86bb175450093c241af67755bb0/artifacts/model/model.pkl', 'rb') as f:
            model_reg = pickle.load(f)
            return model_reg.predict(data)

    # st.text('Taille', Taille)
    # st.text('Poids', Poids)
    # st.text('Taille_metre', Taille_metre)

    if st.button("Prédire les calories brulés"):
        result = predict([[duration, age, IMC]])
        st.text(result[0])
    # if IMC > 25:
    #     st.text('Attention vous dépasser le poids idéale selon votre IMC')


        # with col2:
        #     st.text('test2')
        #     petal_l = st.slider('ca', 1.0, 7.0, 0.5)
        #     petal_l = st.slider('blllll', 1.0, 7.0, 0.5)



# fonction pour la page de monitoring
def monitoring(st, **state):
    st.title('Monitorer mes modèles de machine learning')
    st.write("""Explorer different modèles afin de trouver lequel est le meilleur!""")
    dataset_name = st.sidebar.selectbox("Sélectionner le jeu de données",
                                        ('Dataframe basique non encodé', 'df encode', 'best df', 'best df with age'))
    st.write(dataset_name)
    regressior_name = st.sidebar.selectbox("Selection du modèle de régression",
                                           ("Lasso", "RandomForestRegressor", "regression linéaire"))

    def get_dataset(dataset_name):
        if dataset_name == 'Dataframe basique non encodé':
            data = df
        elif dataset_name == 'df encode':
            data = df2
        elif dataset_name == 'best df':
            data = df3
        elif dataset_name == 'best df with age':
            data = df4
        # Définition de la cible et des features
        if dataset_name == 'Dataframe basique non encodé':
            X = data.drop(['user_id', 'gender', 'calorie'], axis=1)
            y = data.calorie
            return X, y
        elif dataset_name == 'best df':
            X = data.drop(['height', 'weight', 'Height_meters', 'calorie'], axis=1)
            y = data.calorie
            return X, y
        elif dataset_name == 'best df with age':
            X = data.drop(['Unnamed: 0','height', 'weight', 'Height_meters', 'calorie'], axis=1)
            y = data.calorie
            return X, y
        else:
            X = data.drop(['user_id', 'calorie'], axis=1)
            y = data.calorie
            return X, y

    X, y = get_dataset(dataset_name)
    MultiPage.save({"total": X}, namespaces=["Features"])
    st.write("Features", X)

    def add_parameter(clf_name):
        params = {}
        if clf_name == "Lasso":
            L = st.sidebar.slider("alpha", 0.01, 10.00)
            params["alpha"] = L
        elif clf_name == "RandomForestRegressor":
            # params = None
            n_estimators = st.sidebar.slider("n_estimators", 10, 300)
            params["n_estimators"] = n_estimators
            max_depth = st.sidebar.slider("max_depth", 1, 5)
            params["max_depth"] = max_depth
            max_leaf_nodes = st.sidebar.slider("max_leaf_nodes", 2, 30)
            params["max_leaf_nodes"] = max_leaf_nodes
        else:
            positive = st.sidebar.selectbox("positive", (True, False))
            params["positive"] = positive
        return params

    params = add_parameter(regressior_name)

    def get_regressor(clf_name, params):
        if clf_name == "Lasso":
            clf = linear_model.Lasso(alpha=params["alpha"])
        elif clf_name == "RandomForestRegressor":
            clf = RandomForestRegressor(n_estimators=params["n_estimators"],
                                        max_depth=params["max_depth"],
                                        max_leaf_nodes=params["max_leaf_nodes"])
            return clf
        else:
            clf = LinearRegression(positive=params["positive"])
        return clf

    with mlflow.start_run():
        clf = get_regressor(regressior_name, params)
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1234)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        Score = r2_score(y_test, y_pred)

        st.write(f"regressor = {regressior_name}")
        st.write(f"R2 = {Score}")
        if regressior_name == 'RandomForestRegressor':
            st.image('data/grid_search_RFG.png', caption='grid search Random Forest reg')
            st.image('data/learning_curve_RFG_gridsearch.png',
                     caption='LEARNING CURVE RANDOM FOREST REG: meilleur résultat avec les paramètres du grid search')
        elif regressior_name == 'Lasso':
            # st.image('data/grid_search_RFG.png', caption='grid search Random Forest reg')
            st.image('data/learnin_curve_lasso_gridsearch.png',
                     caption="LEARNING CURVE LASSO REG : Meilleur résultat avec alpha=0.01, apres un grid search")


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
