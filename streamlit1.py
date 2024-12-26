import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from ydata_profiling import ProfileReport

st.title('Prédiction du Churn Expresso')

chemin_fichier = "Expresso_churn_dataset.csv"

try:
    df = pd.read_csv(chemin_fichier)
    st.write("Dataset chargé avec succès !")
except FileNotFoundError:
    st.error(f"Fichier non trouvé : {chemin_fichier}")
    st.stop()
except pd.errors.ParserError:
    st.error("Erreur de lecture du fichier CSV. Vérifiez le format.")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue : {e}")
    st.stop()

cols_a_supprimer = ['MRG', 'TENURE', 'TIGO', 'ZONE1', 'ZONE2']
cols_a_supprimer = [col for col in cols_a_supprimer if col in df.columns]
if cols_a_supprimer:
    df.drop(cols_a_supprimer, axis=1, inplace=True)
    st.write(f"Suppression des colonnes : {cols_a_supprimer}")

st.subheader("Exploration des données")
if st.checkbox("Générer un rapport de profilage (peut être lent)", key="profiling_checkbox"):
    try:
        profile = ProfileReport(df)
        st.components.v1.html(profile.to_html(), height=800, scrolling=True)
    except Exception as e:
        st.error(f"Erreur de génération du rapport : {e}")

st.write("Aperçu des données :")
st.dataframe(df.head())

st.write("Informations générales :")
st.write(df.info())

st.write("Statistiques descriptives :")
st.write(df.describe())

st.subheader("Prétraitement des données")

with st.expander("Détails de l'imputation des valeurs manquantes :"):
    for col in df.select_dtypes(include=np.number):
        nb_na_avant = df[col].isna().sum()
        df[col].fillna(df[col].median(), inplace=True)
        if nb_na_avant > 0:
            st.write(f"- Colonne '{col}' : {nb_na_avant} valeurs manquantes imputées par la médiane.")

for col in df.select_dtypes(include='object'):
    try:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    except Exception as e:
        st.error(f"Erreur d'encodage : {e}")
        st.stop()

df.drop_duplicates(inplace=True)

with st.expander("Détails de la gestion des outliers (méthode IQR)"):
    for col in df.select_dtypes(include=np.number):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

if 'CHURN' in df.columns:
    X = df.drop('CHURN', axis=1)
    y = df['CHURN']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    st.subheader("Entraînement du modèle")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Évaluation du modèle")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_scores_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
    cv_scores_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
    cv_scores_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')

    st.write(f"Précision moyenne en validation croisée : {cv_scores_accuracy.mean():.2%} ± {cv_scores_accuracy.std():.2%}")
    st.write(f"F1-score moyen en validation croisée : {cv_scores_f1.mean():.2%} ± {cv_scores_f1.std():.2%}")
    st.write(f"Recall moyen en validation croisée : {cv_scores_recall.mean():.2%} ± {cv_scores_recall.std():.2%}")
    st.write(f"Precision moyenne en validation croisée : {cv_scores_precision.mean():.2%} ± {cv_scores_precision.std():.2%}")

    y_pred = model.predict(X_test)

    st.write(f"Précision sur l'ensemble de test : {accuracy_score(y_test, y_pred):.2%}")
    st.write(f"F1-score sur l'ensemble de test : {f1_score(y_test, y_pred):.2%}")
    st.write(f"Recall sur l'ensemble de test : {recall_score(y_test, y_pred):.2%}")
    st.write(f"Precision sur l'ensemble de test : {precision_score(y_test, y_pred):.2%}")

    st.write("Rapport de classification :")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues",
                xticklabels=['Pas Churn', 'Churn'], yticklabels=['Pas Churn', 'Churn'])
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Valeurs Réelles')
    ax.set_title('Matrice de Confusion')
    st.pyplot(fig)

    st.subheader("Prédictions")

    input_data = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            if min_val == max_val:
                st.warning(f"La colonne '{col}' n'a qu'une seule valeur : {min_val}. Valeur utilisée pour la prédiction.")
                input_data[col] = min_val
            else:
                input_data[col] = st.slider(col, min_val, max_val, float(X[col].mean()))
        else:
            unique_vals = X[col].unique()
            input_data[col] = st.selectbox(col, unique_vals)

    missing_cols = set(X.columns) - set(input_data.keys())
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            input_data[col] = float(X[col].mean())
            st.info(f"Ajout de la colonne numérique manquante {col} avec la valeur par défaut: {float(X[col].mean())}")

    if st.button("Prédire le churn"):
        input_df = pd.DataFrame([input_data])

        for col in X.columns:
            input_df[col] = input_df[col].astype(X[col].dtype)
        input_df = input_df[X.columns]

        input_df[input_df.select_dtypes(include=np.number).columns] = scaler.transform(input_df[input_df.select_dtypes(include=np.number).columns])

        prediction = model.predict(input_df)
        st.write(f"Prédiction : {'Churn' if prediction[0] == 1 else 'Pas de churn'}")

else:
    st.warning("La colonne 'CHURN' n'a pas été trouvée. Vérifiez le nom de la colonne cible.")
