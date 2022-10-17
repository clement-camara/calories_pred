import pickle
import  yellowbrick
from sklearn.model_selection import cross_val_score
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import IPython, ipywidgets
import streamlit as st
from streamlit_multipage import MultiPage
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV


#### 1- RECUPERER LES DATAFRAMES

df = pd.read_csv('data/df_lasso.csv')
df2 = pd.read_csv('data/df_encode_complete_OK.csv')
df3 = pd.read_csv('data/df_encode_complete.csv')
df4 = pd.read_csv('data/df_encode_complete.csv')

@st.cache
def fetch_and_clean_data(data):
    return data

fetch_and_clean_data(df)
fetch_and_clean_data(df2)
fetch_and_clean_data(df3)
fetch_and_clean_data(df4)


#### 1- FONCTION POUR LA PAGE D ACCEUIL

def app_page(st, **state):
    image = Image.open('data/diet_image.jpg')
    st.image(image)

    st.header('Bienvenue sur le caclculateur de votre IMC')

    #### 1- CREATION DES SLIDER POUR LES INPUTS
    weight = st.number_input("Entrez votre poids en kg:")
    status = st.radio('Sélectionnez le format de votre taille: ',
                      ('centimètres', 'mètres', 'pieds'))

    if status == 'centimètres':
        height = st.number_input('Centimetres')
        try:
            bmi = weight / ((height / 100) ** 2)
        except:
            st.text("Entrer une valeur  pour votre taille")
    elif status == 'mètres':
        height = st.number_input('Mètres')
        try:
            bmi = weight / (height ** 2)
        except:
            st.text("Entrer une valeur  pour votre taille")
    else:
        height = st.number_input('Pieds')
        # 1 meter = 3.28
        try:
            bmi = weight / ((height / 3.28) ** 2)
        except:
            st.text("Entrer une valeur  pour votre taille")
    if st.button('Calculatrice IMC'):
        st.text("Votre Index de Masse Graisseuse est de : {}.".format(bmi))

        #### INTERPRETATION DES INDEX IMC

        if bmi < 16:
            st.error("Vous souffrez d'une insuffisance pondérale extrême, retenez votre IMC pour la suite")
        elif 16 <= bmi < 18.5:
            st.warning("Vous souffrez d'une insuffisance pondérale extrême, retenez votre IMC pour la suite")
        elif 18.5 <= bmi < 25:
            st.success("Vous êtes bonne santé, retenez votre IMC pour la suite")
        elif 25 <= bmi < 30:
            st.warning("Vous êtes en surcharge pondérale, , retenez votre IMC pour la suite")
        elif bmi >= 30:
            st.error("Vous êtes en surcharge pondérale extrême")

    st.header('Prédiction avec 3 features')
    st.text('Combien de calories je vais dépenser ?')
    duration = st.slider("Durée d'exercice en minutes", 1, 1, 30)
    age = st.slider("Age", 20, 20, 79)
    IMC = st.slider('IMC', 13.00, 13.00, 32.00)

    #### FONCTION POUR LA PREIDTION ET CREATION DU BOUTON AVEC UNE ANIMATION

    def predict(data):
        with open('MLFlow/mlruns/0/b8e7f86bb175450093c241af67755bb0/artifacts/model/model.pkl', 'rb') as f:
            model_reg = pickle.load(f)
            return model_reg.predict(data)

    if st.button("Prédire les calories brulées"):
        result = predict([ [ duration, age, IMC ] ])
        st.text(result[ 0 ])
        st.balloons()


    #### AJOUT DU LIEN DE L APPLICATION FLASK EN LIGNE SUR HEOKU

    st.title("Retrouver également mon application flask sur ce lien :")
    st.write("https://dietappsimplonbordeaux.herokuapp.com/")


### FONCTION POUR LA PAGE DE MONOTORING

def monitoring(st, **state):
    st.title('Monitorer mes modèles de machine learning')
    st.write("""explorer different modèles afin de trouver lequel est le meilleur!""")


    #### A- CREATION DE LA SELECTION DU DATAFRAME

    dataset_name = st.sidebar.selectbox("Sélectionner le jeu de données",
                                        ('Dataframe basique non encodé sans le genre',
                                         'Dataframe encodé',
                                         'Dataframe encodé avec feature engenering V1',
                                         'Dataframe encodé avec feature engenering V2'))
    st.write(dataset_name)


    #### B CREATION DE LA SELECTION DU MODELE DE ML

    regressior_name = st.sidebar.selectbox("Selection du modèle",
                                           ("Lasso",
                                            "RandomForestRegressor",
                                            "regression linéaire"))


    #### FONCTION QUI RECUPERE LE DATAFRAME AVEC LES BONNE COLONNES SUR LEQUEL SERA FAIT LA PREIDCTION

    def get_dataset(dataset_name):

        if dataset_name == 'Dataframe basique non encodé sans le genre':
            data = df

        elif dataset_name == 'Dataframe encodé':
            data = df2

        elif dataset_name == 'Dataframe encodé avec feature engenering V1':
            data = df3

        elif dataset_name == 'Dataframe encodé avec feature engenering V2':
            data = df4


        #### DEFINITION DE LA  CIBLE ET DES FEATURES

        if dataset_name == 'Dataframe basique non encodé sans le genre':
            X = data.drop([ 'Unnamed: 0', 'user_id', 'gender', 'calorie' ], axis=1)
            y = data.calorie
            return X, y

        elif dataset_name == 'Dataframe encodé':
            X = data.drop(['calorie', 'Unnamed: 0', 'calculated_IMC', 'Height_meters'], axis=1)
            y = data.calorie
            return X, y

        elif dataset_name == 'Dataframe encodé avec feature engenering V1':
            X = data.drop(['calorie','Height_meters','Unnamed: 0','height','weight','duration', 'female', 'male'], axis=1)
            y = data.calorie
            return X, y

        else:
            X = data.drop(['age','calorie','Height_meters','Unnamed: 0','height','weight','duration', 'female', 'male'], axis=1)
            y = data.calorie
            return X, y


    #### DEFINITION DES FEATURES ET DE LA TARGET

    X, y = get_dataset(dataset_name)
    MultiPage.save({"total": X}, namespaces=[ "Features" ])
    st.write("Dataset", X)


    ### FONCTION DE CREATION DES PARAMETRES EN FONCTION DU MODELE CHOISIT

    def add_parameter(clf_name):
        params = {}

        if clf_name == "Lasso":

            #### ALPHA
            L = st.sidebar.slider("alpha", 0.00, 10.00)
            params["alpha"] = L


        elif clf_name == "RandomForestRegressor":

            #### n_estimators
            n_estimators = st.sidebar.slider("n_estimators", 10, 300)
            params["n_estimators"] = n_estimators

            #### max_depth
            max_depth = st.sidebar.slider("max_depth", 1, 5)
            params["max_depth"] = max_depth

            #### max_leaf_nodes
            max_leaf_nodes = st.sidebar.slider("max_leaf_nodes", 2, 30)
            params["max_leaf_nodes"] = max_leaf_nodes

        else:
            #### positive
            positive = st.sidebar.selectbox("positive", (True, False))
            params["positive"] = positive
        return params

    params = add_parameter(regressior_name)


    #### 1- FONCTION DE CREATION: MODEL + PARAMETRE

    def get_regressor(clf_name, c):

        ##### LASSO
        if clf_name == "Lasso":
            clf = linear_model.Lasso(alpha=params["alpha"])

        ##### RandomForestRegressor
        elif clf_name == "RandomForestRegressor":
            clf = RandomForestRegressor(n_estimators=params["n_estimators"],
                                        max_depth=params["max_depth"],
                                        max_leaf_nodes=params["max_leaf_nodes"])
        ##### LinearRegression
        else:
            clf = LinearRegression(positive=params["positive"])
        return clf
    # with mlflow.start_run():
    clf = get_regressor(regressior_name, params)


    #### SEPARATION DES DONNES DE TEST ET D'ENTRAINEMENT

    st.sidebar.header('Définir les HyperParamètres pour la séparation des données')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 50, 90, 80, 5)
    seed = st.sidebar.slider("seed", 1, 400)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=seed)


   # CROSS VALIDATION / GRID SEARCH

    st.sidebar.header('Définir les HyperParamètres pour la cross val des données')
    parameter_cross_validation = st.sidebar.slider('Nombre de split de validation croisée', 2, 10)
    n_repeats = st.sidebar.slider('nombre de répétitions valeur croisée', 1, 10)
    random_state_CV2 = st.sidebar.slider('Random state de la validation croisée', 1, 300)

    Choose_score_metric = st.sidebar.selectbox("Choisir la métrique de performance", ('r2', 'neg_mean_absolute_error'))

    cv = RepeatedKFold(n_splits=parameter_cross_validation, n_repeats=n_repeats, random_state=random_state_CV2)
    # evaluate model
    scores = cross_val_score(clf, X_train, y_train, scoring=Choose_score_metric, cv=cv).mean()
    scores_test = cross_val_score(clf, X_test, y_test, scoring=Choose_score_metric, cv=cv).mean()
    st.write(f"Cross val score Train Set = {scores}")
    st.write(f"Cross val score Test Set = {scores_test}")

    clf.fit(X_train, y_train)


    #### ENTRAINEMENT DU MODELE SUR LE JEU D ENTRAIENEMENT

    #param_grid.fit(X_train, y_train)


    #### PREDICTION SUR LE JEU DE TEST

    y_pred = clf.predict(X_test)

    R2 = r2_score(y_test, y_pred)
    #R2_train = clf.score(X_train, y_train)

    #MAE_train = clf.mean_absolute_error(X_train, y_train)
    MAE = mean_absolute_error(y_test, y_pred)

    #### AFFICHAGE DU SCORE ET DU  MODELE

    st.sidebar.header('MODELE')
    st.write(f"regressor = {regressior_name}")

   ## st.write(f"R2_train = {R2_train}")
    st.write(f"R2 = {R2}")

    st.write(f"MAE = {MAE}")
   # st.write(f"MAE train = {MAE_train}")


    #### AFFICHAGE DE LA LEARNING CURVE SELON LE MODELE UTILISÉ

    st.set_option('deprecation.showPyplotGlobalUse', False)

    from yellowbrick.model_selection import LearningCurve
    # Create the learning curve visualizer


    visualizer = LearningCurve(clf, scoring='r2')
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()



    from yellowbrick.regressor import ResidualsPlot

    visualizer = ResidualsPlot(clf)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()


    from yellowbrick.regressor import ResidualsPlot

    visualizer = ResidualsPlot(clf, hist=False, qqplot=True)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()
    # if regressior_name == 'RandomForestRegressor':
    #     st.image('data/grid_search_RFG.png', caption='grid search Random Forest reg')
    #     st.image('data/learning_curve_RFG_gridsearch.png',
    #              caption='LEARNING CURVE RANDOM FOREST REG: meilleur résultat avec les paramètres du grid search')
    # elif regressior_name == 'Lasso':
    #     # st.image('data/grid_search_RFG.png', caption='grid search Random Forest reg')
    #     st.image('data/learnin_curve_lasso_gridsearch.png',
    #              caption="LEARNING CURVE LASSO REG : Meilleur résultat avec alpha=0.01, apres un grid search")




### FONCTION POUR LA PAGE D'EXPLORATION DES DONNÉES

def visualisation_page(st, **state):
    MultiPage.save({"total": df}, namespaces=[ "df" ])
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
