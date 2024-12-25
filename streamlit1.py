import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from ydata_profiling import ProfileReport

st.title('Expresso Churn Prediction')

# Chemin d'accès à ton fichier CSV
chemin_fichier = "C:\\Users\\Admin.local\\Desktop\\OSL 2024\\GoMyCode\\CHECKPOINT STREAMLIT 1\\Expresso_churn_dataset.csv"

# Charger le dataset avec gestion des erreurs
try:
    df = pd.read_csv(chemin_fichier)
    st.write("Dataset chargé avec succès !")
except FileNotFoundError:
    st.error(f"Fichier CSV non trouvé à l'emplacement : {chemin_fichier}. Veuillez vérifier le chemin.")
    st.stop()
except pd.errors.ParserError:
    st.error(f"Erreur lors de la lecture du fichier CSV. Veuillez vérifier le format du fichier.")
    st.stop()
except Exception as e:
    st.error(f"Une erreur inattendue s'est produite : {e}")
    st.stop()

    # Exploration des données avec ydata-profiling
st.subheader("Exploration des données")
if st.checkbox("Générer un rapport de profilage"):
    try:
        profile = ProfileReport(df)
        st.components.v1.html(profile.to_html(), height=800, scrolling=True)
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport de profilage : {e}")

st.write("Aperçu des données :")
st.dataframe(df.head())

st.write("Informations générales :")
st.write(df.info())

st.write("Statistiques descriptives :")
st.write(df.describe())

# Prétraitement des données
st.subheader("Prétraitement des données")

# Gestion des valeurs manquantes (imputation par la médiane pour les numériques)
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Gestion des valeurs catégorielles (Label Encoding)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Suppression des doublons
df.drop_duplicates(inplace=True)

# Gestion des outliers (exemple simple : suppression des valeurs > 3 écarts-types)
for col in df.select_dtypes(include=np.number).columns:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

# Préparation des données pour le modèle
if 'CHURN' in df.columns:
    X = df.drop('CHURN', axis=1)
    y = df['CHURN']

    # Standardisation des features numériques
    scaler = StandardScaler()
    X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X[X.select_dtypes(include=np.number).columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    st.subheader("Entraînement du modèle")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Évaluation du modèle
    st.subheader("Évaluation du modèle")
    score = model.score(X_test, y_test)
    st.write(f"Précision du modèle : {score:.2%}")

    y_pred = model.predict(X_test)
    st.write("Rapport de classification :")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

    # Interface Streamlit pour les prédictions
    st.subheader("Prédictions")

    input_data = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            min_val = float(X[col].min())
            max_val = float(X[col].max())

            # Gestion du cas où min_val == max_val
            if min_val == max_val:
                if max_val == 0:
                    max_val = 1.0 #Pour ne pas avoir de division par zero
                    min_val = 0.0
                else:
                    max_val += 1.0  # Incrémenter pour éviter l'erreur
                    min_val -= 1.0 # Décrémenter min_val pour que min_val < max_val
                st.warning(f"La colonne '{col}' n'a qu'une valeur unique ({min_val+1}). Le slider a été ajusté. [{min_val},{max_val}]") #Message d'avertissement plus clair
            input_data[col] = st.slider(col, min_val, max_val, float(X[col].mean()))
        else:
            unique_vals = X[col].unique()
            input_data[col] = st.selectbox(col, unique_vals)

    if st.button("Prédire le churn"):
        input_df = pd.DataFrame([input_data])
        input_df[input_df.select_dtypes(include=np.number).columns] = scaler.transform(input_df[input_df.select_dtypes(include=np.number).columns])
        prediction = model.predict(input_df)
        st.write(f"Prédiction : {'Churn' if prediction[0] == 1 else 'Pas de churn'}")

else:
    st.warning("La colonne 'CHURN' n'a pas été trouvée dans le dataset. Veuillez vérifier le nom de la colonne cible.")