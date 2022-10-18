# MANIPULATION DES DONNÉES
import mlflow
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_multipage import MultiPage
from streamlit_pandas_profiling import st_profile_report

# DIVISER LES DONNEES
from sklearn.model_selection import train_test_split

# METTRE A L ECHELLE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# VALIDATION CROISÉE DES DONNÉES
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

# MODÈLES DE MACHINE LEARNING
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

# METRIQUES DE PERFORMANCE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# INTERPRETATION DES MODÈLES
from sklearn.model_selection import learning_curve
import  yellowbrick

# AUTRE
import pickle
import requests.models
import IPython, ipywidgets
from PIL import Image


#MLFLOW Monitoring
# import mlflow
#
mlflow.set_tracking_uri('/Users/marinelafargue/Desktop/projet calorie/MLFlow/mlruns')
mlflow.set_experiment('test-experiment BB')
mlflow.sklearn._autolog()

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

#### 1- RECUPERER LES DATAFRAMES

df = pd.read_csv('data/df_lasso.csv')
df2 = pd.read_csv('data/df_encode_complete_OK.csv')
df3 = pd.read_csv('data/df_encode_complete.csv')
df4 = pd.read_csv('data/df_encode_complete.csv')
df5 = pd.read_csv('data/dataframe_app.csv')

#@st.cache
def fetch_and_clean_data(data):
    return data

fetch_and_clean_data(df)
fetch_and_clean_data(df2)
fetch_and_clean_data(df3)
fetch_and_clean_data(df4)
fetch_and_clean_data(df5)


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

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
            st.text("Entrer une valeur pour votre taille")
    elif status == 'mètres':
        height = st.number_input('Mètres')
        try:
            bmi = weight / (height ** 2)
        except:
            st.text("Entrer une valeur pour votre taille")
    else:
        height = st.number_input('Pieds')
        # 1 meter = 3.28
        try:
            bmi = weight / ((height / 3.28) ** 2)
        except:
            st.text("Entrer une valeur pour votre taille")
    if st.button('Calculatrice IMC'):
        st.text("Votre Index de Masse Graisseuse est de : {}.".format(bmi))


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

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
    # duration = st.slider("Durée d'exercice en minutes", 1, 1, 30)
    # age = st.slider("Age", 20, 20, 79)

    # IMC = st.slider('IMC', 13.00, 13.00, 32.00)
    # heart_rate = st.slider("Pulsation cardiaque en minutes", 67, 67, 128)
    # body_temp = st.slider("Température du corps en degrés", 37, 37, 42)

    duration = st.slider("Durée d'exercice en minutes", 1, 1, 30)
    age = st.slider("Age", 20, 20, 79)
    IMC = st.slider('IMC', 13.00, 13.00, 32.00)


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    #### FONCTION POUR LA PREIDTION ET CREATION DU BOUTON AVEC UNE ANIMATION

    def predict_(data):
        with open('/Users/marinelafargue/Desktop/projet calorie/Notebook ML/RF_pkl', 'rb') as f:
            model_reg = pickle.load(f)
            st.write(age)
            st.write(IMC)
            st.write(duration)
            return model_reg.predict(data)


    if st.button("Prédire les calories brulées"):
        result = predict_([[age,duration,IMC]])

        st.text(result[0])

    #### AJOUT DU LIEN DE L APPLICATION FLASK EN LIGNE SUR HEOKU

    st.title("Retrouver également mon application flask sur ce lien :")
    st.write("https://dietappsimplonbordeaux.herokuapp.com/")

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

### FONCTION POUR LA PAGE DE MONOTORING

def monitoring(st, **state):
    st.title('Monitorer mes modèles de machine learning')
    st.write("""explorer different modèles afin de trouver lequel est le meilleur!""")


    #### A- CREATION DE LA SELECTION DU DATAFRAME

    st.sidebar.header('**FILTRES**')
    dataset_name = st.sidebar.selectbox("Sélectionner le jeu de données",
                                        ('Dataframe avec toute les features',
                                        'Dataframe basique non encodé sans le genre',
                                         'Dataframe encodé',
                                         'Dataframe encodé avec feature engenering V1',
                                         'Dataframe encodé avec feature engenering V2'))
    st.write(dataset_name)
    st.spinner(text="Dataframe n progress ...")


    #### B CREATION DE LA SELECTION DU MODELE DE ML

    regressior_name = st.sidebar.selectbox("Selection du modèle",
                                           ("Lasso",
                                            "RandomForestRegressor",
                                            'AdaBoostRegressor',
                                            "regression linéaire"))

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

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

        elif dataset_name == 'Dataframe avec toute les features':
            data = df5



    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------


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

        elif dataset_name == 'Dataframe avec toute les features':
            X = data.drop(['calorie'], axis=1)
            y = data.calorie
            return X, y

        else:
            X = data.drop(['age','calorie','Height_meters','Unnamed: 0','height','weight','duration', 'female', 'male'], axis=1)
            y = data.calorie
            return X, y


    #### DEFINITION DES FEATURES ET DE LA TARGET

    X, y = get_dataset(dataset_name)
    MultiPage.save({"total": X}, namespaces=[ "Features" ])
    st.write("**Dataset utilsé**", X)


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

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

        elif clf_name == "AdaBoostRegressor":

            #### n_estimators
            learning_rate = st.sidebar.slider("learning_rate", 0.1, 1.0)
            params["learning_rate"] = learning_rate

            #### n_estimators
            n_estimators = st.sidebar.slider("n_estimators", 50, 500)
            params["n_estimators"] = n_estimators


        else:
            #### positive
            positive = st.sidebar.selectbox("positive", (True, False))
            params["positive"] = positive
        return params

    params = add_parameter(regressior_name)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    #### 1- FONCTION DE CREATION: MODÉLE + PARAMÈTRE

    def get_regressor(clf_name, c):

        ##### LASSO
        if clf_name == "Lasso":
            clf = linear_model.Lasso(alpha=params["alpha"])

        ##### RandomForestRegressor
        elif clf_name == "RandomForestRegressor":
            clf = RandomForestRegressor(n_estimators=params["n_estimators"],
                                        max_depth=params["max_depth"],
                                        max_leaf_nodes=params["max_leaf_nodes"])
        elif clf_name == "AdaBoostRegressor":
            clf = AdaBoostRegressor(learning_rate=params["learning_rate"],
                                    n_estimators=params[ "n_estimators" ]
                                     )

        ##### LinearRegression
        else:
            clf = LinearRegression(positive=params["positive"])
        return clf
    # with mlflow.start_run():
    clf = get_regressor(regressior_name, params)


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    #### SEPARATION DES DONNES DE TEST ET D'ENTRAINEMENT

    st.sidebar.header('Définir les HyperParamètres pour la séparation des données')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 50, 90, 80, 5)
    seed = st.sidebar.slider("seed", 1, 400)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=seed)


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

   # CROSS VALIDATION / GRID SEARCH

    st.sidebar.header('Définir les HyperParamètres pour la validation croisée des données')
    parameter_cross_validation = st.sidebar.slider('Nombre de split de validation croisée', 2, 10)
    n_repeats = st.sidebar.slider('nombre de répétitions des valeurs croisées', 1, 10)
    random_state_CV2 = st.sidebar.slider('Random state de la validation croisée', 1, 300)

    cv = RepeatedKFold(n_splits=parameter_cross_validation, n_repeats=n_repeats, random_state=random_state_CV2)
    # evaluate model
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=cv).mean()
    scores_test = cross_val_score(clf, X_test, y_test, scoring='r2', cv=cv).mean()
    st.write("**Affichage des validations croisées**")
    st.write(f"Cross val score du Train Set = {scores}")
    st.write(f"Cross val score du Test Set = {scores_test}")

    st.spinner('En attente du fit des données d entrainement...')
    st.success('Done!')

    #### ENTRAINEMENT DU MODELE SUR LE JEU D ENTRAINENEMENT
    clf.fit(X_train, y_train)

    # with open('LAST_MODEL_pkl', 'wb') as files:
    #     pickle.dump(clf, files)

    #### PREDICTION SUR LE JEU DE TEST
    y_pred = clf.predict(X_test)

    #### AFFICHAGE DU MODELE ET DES METRIQUES
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)


    st.sidebar.header('  ')

    st.write("**Affichage du modèle**")
    st.write(f"regressor = {regressior_name}")
    st.write("**Affichage des métriques de performance**")
    st.write(f"R2 = {R2}")
    st.write(f"MAE = {MAE}")


    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    #### AFFICHAGE DE LA LEARNING CURVE SELON LE MODELE UTILISÉ

    st.set_option('deprecation.showPyplotGlobalUse', False)

    from yellowbrick.model_selection import LearningCurve
    # Create the learning curve visualizer
    visualizer = LearningCurve(clf, scoring='r2')
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    #### AFFICHAGE DES RESIDUS


    st.markdown("**L'erreur de prédiction doit suivre une distribution normale avec une moyenne de 0**")

    from yellowbrick.regressor import ResidualsPlot

    visualizer = ResidualsPlot(clf)
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()
    visualizer = ResidualsPlot(clf, hist=False, qqplot=True)

    #### AFFICHAGE DES RESIDUS PAR QQPLOT

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------


### FONCTION POUR LA PAGE D'EXPLORATION DES DONNÉES

def visualisation_page(st, **state):
    MultiPage.save({"total": df}, namespaces=[ "df" ])
    profile = ProfileReport(df,
                            title="Diet Data",
                            dataset={
                                "description": "Ce rapport de profilage a été généré pour la certification Simplon.",
                                "copyright_holder": "Camara Clement",
                                "copyright_year": "2022"
                            })
    st_profile_report(profile)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

##### PAGE QUI CONSTITUE LE READ ME, L'ARCHITECTURE DU PROJET

def read_me(st, **state):
    st.header('ARCHITECTURE')

    image = Image.open('data/arrchi_projet_vf.png')
    st.image(image)

    st.header('Pré-requis')
    st.text('** un IDE **')
    st.text('** Un compte github **')
    st.text('** Anaconda comme gestionnaire de paquets **')

    st.header('Premier pas sur le projet')

    st.markdown('**1) Cloner le repo Github**')
    code = '''git clone https://github.com/collectif-CAKUVA/rocketfid '''
    st.code(code, language='python')

    st.markdown("**2) Récupérer l'environnement de travail conda avec le fichier .yml:**")
    code = '''conda env create -f environment.yml
conda activate environnement '''
    st.code(code, language='python')

    st.markdown("**3) Si d'autres dépendances sont ajoutées/supprimées , mettre à jour le .yml à l'aide de la commande**")
    code = '''conda env export --from-history > environment.yml '''
    st.code(code, language='python')

    st.markdown("**3) Si vous n'utilisez pas conda récupérez les packages avec le requirements.txt**")

    st.markdown("**4) Démarrer streamlit en local**")
    code = '''streamlit run main.py'''
    st.code(code, language='python')

    st.markdown("**5) Lancer l'application Flask**")
    code = '''Lancer app.py'''
    st.code(code, language='python')

    st.header('SECURITÉ')
    st.markdown("**App Streamli **")
    "Authentification unique : tous les accès et connexions à Streamlit sont effectués via un fournisseur SSO : GitHub et GSuite. Les mots de passe des clients ne sont pas stockés."
    "Stockage des identifiants : les jetons d'authentification sont chiffrés"

    st.header('FONCTIONNALITÉS :')

    st.header('*FONCTION DB*')

    st.markdown('**Notebook insert into DB pour insérer les données dans la base de données Postresql en local**')
    code = '''Connexion
création de la table
Insertion '''
    st.code(code, language='python')

    st.markdown("**Connection à la DB sécurisé via des variables d'environnement et un fichier.env**")
    code = '''load_dotenv()  # Nécessaire pour charger les variables d'environnement précédemment définies
# Créer une connexion à postgres
connection = psycopg2.connect(host=os.environ.get('PG_HOST'),
                        user=os.environ.get('PG_USER'),
                        password=os.environ.get('PG_PASSWORD'),
                        dbname=os.environ.get('PG_DATABASE'))
connection.autocommit = True  # Assurez-vous que les données sont ajoutées à la base de données immédiatement après les commandes d'écriture.
cursor = connection.cursor()
cursor.execute('SELECT %s as connected;', ('Connection à postgres Réussie!',))
print(cursor.fetchone())
 '''
    st.code(code, language='python')

    st.markdown('**Fonction fetch de la donnée depuis la DB**')
    code = '''
def postgresql_to_dataframe(conn, select_query, column_names):
    """
    Transformer une requête SELECT en un dataframe pandas
    """
    cursor = conn.cursor()
    try:
        cursor.execute(select_query)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        cursor.close()
        return 1
    
    # Naturellement, nous obtenons une liste de "tupples".
    tupples = cursor.fetchall()
    cursor.close()
    
    # Nous devons juste le transformer en dataframe pandas.
    df = pd.DataFrame(tupples, columns=column_names)
    return df'''
    st.code(code, language='python')

    st.markdown('**Fonction de création du dataframe**')
    code = '''conn = connection
column_names = ["user_id","gender", "age", "height", "weight", "duration", "heart_rate", "body_temp", "calorie"]
# Execute the "SELECT *" query
df_db = postgresql_to_dataframe(conn, 
"SELECT persons.user_id as id, gender, age, height, weight, duration, heart_rate, body_temp,calorie FROM calories INNER JOIN persons ON calories.user_id = persons.user_id"
                                , column_names)
df_db.head() '''
    st.code(code, language='python')


    st.header('*FONCTION DE L APP*')

    st.markdown('**Fonction pour récupérer les données**')
    code = '''fetch_and_clean_data(data) '''
    st.code(code, language='python')

    st.markdown('**Fonction pour ajouter les pages de mon app**')
    code = '''app_page(st, **state): '''
    st.code(code, language='python')

    st.markdown('**Fonction pour prédire**')
    code = '''predict(data):'''
    st.code(code, language='python')

    st.markdown("**Fonction qui s'occupe de la page monitoring**")
    code = '''monitoring(st, **state)'''
    st.code(code, language='python')

    st.markdown("**Fonction qui récupère les 4 Datasets utilsiés**")
    code = '''get_dataset(dataset_name)'''
    st.code(code, language='python')

    st.markdown("**Fonction qui créé et ajoute les paramètres du modèle**")
    code = '''add_parameter(clf_name)'''
    st.code(code, language='python')

    st.markdown("**Fonction qui récupère le modèle**")
    code = '''get_regressor(clf_name, c)'''
    st.code(code, language='python')

    st.markdown("**Fonction qui créé et ajoute les paramètres du modèle**")
    code = '''add_parameter(clf_name)'''
    st.code(code, language='python')

    st.markdown("**Fonction qui créé la page de visualisation avec profile report**")
    code = '''visualisation_page(st, **state)'''
    st.code(code, language='python')

    st.header('*ML FLOW*')

    st.markdown("**Le fichier pickles pour charger le modèle est généré par MLFLow en direct**")
    code =     """    def predict(data):
        with open('MLFlow/mlruns/0/b8e7f86bb175450093c241af67755bb0/artifacts/model/model.pkl', 'rb') as f:
            model_reg = pickle.load(f)
            return model_reg.predict(data)"""

    st.markdown("**Les artifacts, paramètres, métriques et modèles sont chargé en direct sur MLFlow depuis l'application de monitoring**")
    image = Image.open('data/mlflow_folder.png')
    st.image(image)




# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

app = MultiPage()
app.navbar_style = "SelectBox"
app.st = st
app.navbar_name = "Menu"

app.add_app("Diet app Page", app_page)
app.add_app("Monitoring Page", monitoring)
app.add_app("Visualisation Page", visualisation_page)
app.add_app("Documentation de l'application", read_me)
app.run()
