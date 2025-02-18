import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pdfkit
from streamlit_extras.let_it_rain import rain
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# ✅ Charger les fichiers `.pkl`
try:
    model = joblib.load("models/xgboost_credit_risk.pkl")
    ohe = joblib.load("models/onehot_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Fichier manquant : {e}")
    st.stop()

# ✅ Titre de l'application
st.title("📝 Prédiction d'accord de prêt")

# ✅ Définition des colonnes encodées et normalisées
ohe_columns = ['cb_person_default_on_file', 'person_home_ownership', 'loan_intent', 'income_group', 'age_group', 'loan_amount_group']  

normal_cols = ['person_income', 'person_age', 'person_emp_length', 'loan_amnt', 
               'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income', 
               'loan_to_income_ratio', 'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio']

# ✅ Interface Utilisateur
person_age = st.slider("Âge de l'emprunteur", 20, 80, 30)
person_income = st.number_input("Revenu annuel (€)", min_value=4000, max_value=6000000, value=32000, step=1000)
person_home_ownership = st.selectbox("Type de propriété", ["RENT", "MORTGAGE", "OWN", "OTHER"])
person_emp_length = st.slider("Durée d'emploi (années)", 0, 60, 5)
loan_intent = st.selectbox("Motif du prêt", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_amnt = st.number_input("Montant du prêt (€)", min_value=500, max_value=35000, value=10000, step=500)
loan_int_rate = st.slider("Taux d'intérêt du prêt (%)", 4.0, 24.0, 10.0, step=0.1)
cb_person_default_on_file = st.selectbox("Historique de défaut de paiement ?", ["Y", "N"])
cb_person_cred_hist_length = st.slider("Durée d'historique de crédit (années)", 1, 30, 5)

# ✅ Calcul des nouvelles features
loan_percent_income = loan_amnt / person_income  # 🔥 Ajouté ici pour s'assurer qu'il existe
loan_to_income_ratio = loan_amnt / person_income  # 🔥 Vérifié à nouveau
loan_to_emp_length_ratio = person_emp_length / loan_amnt if loan_amnt != 0 else 0
int_rate_to_loan_amt_ratio = loan_int_rate / loan_amnt if loan_amnt != 0 else 0

# ✅ Création du DataFrame utilisateur
input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "loan_percent_income": loan_percent_income,  # 🔥 Vérifié qu'il est bien là
    "loan_to_income_ratio": loan_to_income_ratio,  # 🔥 Vérifié qu'il est bien là
    "loan_to_emp_length_ratio": loan_to_emp_length_ratio,
    "int_rate_to_loan_amt_ratio": int_rate_to_loan_amt_ratio
}])

#  Vérification avant encodage
#st.write("📋 Colonnes disponibles dans `input_data` avant encodage :", input_data.columns.tolist())

# Ajouter les nouvelles colonnes catégoriques
input_data['age_group'] = pd.cut(input_data['person_age'],
                                 bins=[20, 26, 36, 46, 56, 66],
                                 labels=['20-25', '26-35', '36-45', '46-55', '56-65'])

input_data['income_group'] = pd.cut(input_data['person_income'],
                                    bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                                    labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])

input_data['loan_amount_group'] = pd.cut(input_data['loan_amnt'],
                                         bins=[0, 5000, 10000, 15000, float('inf')],
                                         labels=['small', 'medium', 'large', 'very large'])

#  Vérification après ajout des groupes
#st.write("📋 Colonnes après ajout des groupes :", input_data.columns.tolist())

# Appliquer l'encodage OneHotEncoder
try:
    encoded_data = pd.DataFrame(ohe.transform(input_data[ohe_columns]), columns=ohe.get_feature_names_out())  
except Exception as e:
    st.error(f"❌ Erreur lors de l'encodage OneHotEncoder : {e}")
    st.stop()

# Vérification des colonnes après encodage
#st.write("📋 Colonnes après encodage OneHotEncoder :", encoded_data.columns.tolist())

# Vérifier que toutes les colonnes de `normal_cols` existent avant la normalisation
missing_normal_cols = [col for col in normal_cols if col not in input_data.columns]
if missing_normal_cols:
    st.error(f"⚠️ Colonnes manquantes pour la normalisation : {missing_normal_cols}")
    st.stop()

# Affichage des colonnes existantes avant la normalisation
#st.write("📋 Colonnes présentes avant la normalisation :", input_data.columns.tolist())

# Appliquer la normalisation StandardScaler
try:
    scaled_data = pd.DataFrame(scaler.transform(input_data[normal_cols]), columns=normal_cols)
except Exception as e:
    st.error(f"❌ Erreur lors de la normalisation StandardScaler : {e}")
    st.stop()

# Fusion des données encodées et normalisées
final_data = pd.concat([encoded_data, scaled_data], axis=1)

# Vérification des dimensions
if final_data.shape[1] != model.n_features_in_:
    st.error(f"⚠️ Nombre de features incorrect ! Attendu : {model.n_features_in_}, reçu : {final_data.shape[1]}")
    st.stop()

# Faire la prédiction avec le modèle XGBoost
prediction = model.predict(final_data)

st.metric(label="💰 Montant du prêt", value=f"{loan_amnt} €")
st.metric(label="📈 Taux d'intérêt", value=f"{loan_int_rate} %")
st.metric(label="💼 Revenu annuel", value=f"{person_income} €")




# 🎉 Résultat avec effet spécial si le prêt est accordé
if prediction[0] == 1:
     st.toast("Essayez d'augmenter vos revenus ou de diminuer le montant du prêt.", icon="⚠️")
else:
    st.success("✅ Prêt accordé ! 🎉💸")

    # 🌧️💵 Faire pleuvoir des billets
    rain(
        emoji="💸",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

st.title("📋 Synthèse de l'importance de chaque variable dans la décision du modèle :")

# Récupérer l'importance des variables
importances = model.feature_importances_
feature_names = final_data.columns

# Trier les variables par importance
sorted_indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]

# Regrouper les variables par catégorie pour une meilleure lisibilité
category_labels = {
    "loan_to_income_ratio": "💰 Ratios financiers",
    "loan_to_emp_length_ratio": "💰 Ratios financiers",
    "int_rate_to_loan_amt_ratio": "💰 Ratios financiers",
    "loan_percent_income": "💰 Ratios financiers",
    "cb_person_cred_hist_length": "📈 Longueur historique de crédit",
    "loan_int_rate": "💰 Taux d'intérêt",
    "loan_amnt": "💰 Montant du prêt",
    "person_emp_length": "💼 Ancienneté emploi",
    "person_age": "👤 Âge de l'emprunteur",
    "person_income": "💰 Revenu annuel",
    
    "age_group_20-25": "👤 Âge de l'emprunteur",
    "age_group_26-35": "👤 Âge de l'emprunteur",
    "age_group_36-45": "👤 Âge de l'emprunteur",
    "age_group_46-55": "👤 Âge de l'emprunteur",
    "age_group_56-65": "👤 Âge de l'emprunteur",

    "income_group_low": "💰 Revenu annuel",
    "income_group_low-middle": "💰 Revenu annuel",
    "income_group_middle": "💰 Revenu annuel",
    "income_group_high-middle": "💰 Revenu annuel",
    "income_group_high": "💰 Revenu annuel",

    "loan_intent_DEBTCONSOLIDATION": "📑 Type de prêt",
    "loan_intent_EDUCATION": "📑 Type de prêt",
    "loan_intent_HOMEIMPROVEMENT": "📑 Type de prêt",
    "loan_intent_MEDICAL": "📑 Type de prêt",
    "loan_intent_PERSONAL": "📑 Type de prêt",
    "loan_intent_VENTURE": "📑 Type de prêt",

    "person_home_ownership_MORTGAGE": "🏠 Type de propriété",
    "person_home_ownership_OTHER": "🏠 Type de propriété",
    "person_home_ownership_OWN": "🏠 Type de propriété",
    "person_home_ownership_RENT": "🏠 Type de propriété",

    "cb_person_default_on_file_Y": "🚨 Historique de défaut",
    "cb_person_default_on_file_N": "🚨 Historique de défaut",
}

# Sidebar pour choisir les variables à afficher
selected_categories = st.sidebar.multiselect(
    "📊 Sélectionnez les catégories à afficher",
    options=set(category_labels.values()),  # Liste unique des catégories
    default=["💰 Ratios financiers", "📑 Type de prêt", "🏠 Type de propriété"]  # Catégories par défaut
)

# Regrouper les variables par catégorie sélectionnée
aggregated_features = {}
for feature, importance in zip(sorted_features, sorted_importances):
    category = category_labels.get(feature, "Autres")
    if category in selected_categories:  # Vérifier si la catégorie est sélectionnée
        aggregated_features[feature] = importance

# Afficher un graphique unique avec toutes les catégories sélectionnées
if aggregated_features:
    fig, ax = plt.subplots(figsize=(10, len(aggregated_features) * 0.3))
    sns.barplot(x=list(aggregated_features.values()), y=list(aggregated_features.keys()), palette="Blues_r", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Variables")
    ax.set_title("🔍 Importance des variables sélectionnées")

    st.pyplot(fig)
else:
    st.warning("⚠️ Aucune catégorie sélectionnée. Veuillez en choisir au moins une dans la barre latérale.")
