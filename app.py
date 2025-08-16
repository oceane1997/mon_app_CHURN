import streamlit as st
import pandas as pd
import joblib

#st.title("Prédiction du Churn des Clients")
#st.write("Si tu vois ceci, Streamlit fonctionne bien !")

#import os
#st.write("Fichiers dans le dossier :", os.listdir())

model = joblib.load('log_model.pkl')  # Remplace par le nom exact de ton fichier
scaler = joblib.load('scaler.pkl')        # Si tu as utilisé un scaler

st.title("Prédiction du Churn des Clients")

st.markdown("Remplissez les informations ci-dessous pour prédire le churn.")

# === Variables numériques avec winsorisation ===

REGULARITY_CAPPED = min(st.number_input("REGULARITE (Nombre de fois que le client est actif pendant 90 jours)", min_value=0), 62)
FREQ_TOP_PACK_CAPPED = min(st.number_input("FREQUENCE D'ACTIVATION DES TOP PACK", min_value=0), 50)
ON_NET_CAPPED = min(st.number_input("FREQUENCE D'APPEL INTER EXPRESSO", min_value=0), 3304)
ORANGE_CAPPED = min(st.number_input("FREQUENCE D'APPEL VERS ORANGE", min_value=0), 707)
DATA_VOLUME_CAPPED = min(st.number_input("DATA VOLUME", min_value=0), 25957)
ARPU_SEGMENT_C = min(st.number_input("ARPU SEGMENT (MOYENNE DES REVENUS SUR 90 Jours)", min_value=0.0), 9779)
FREQUENCE_CAPPED = min(st.number_input("FREQUENCE (Nombre de fois qu'un client a fait un revenu)", min_value=0), 62)
MONTANT_CAPPED = min(st.number_input("MONTANT DE RECHARGE", min_value=0), 28776)
TIGO_CAPPED = min(st.number_input("FREQUENCE D'APPEL VERS TIGO", min_value=0), 151)
FREQUENCE_RECH_CAPPED = min(st.number_input("FREQUENCE RECHARGE (Nombre de fois qu'un client a fait une recharge)", min_value=0), 57)


# === Variable TENURE avec encodage ordinal ===

ordre_tenure = {
    'D 3-6 month': 1,
    'E 6-9 month': 2,
    'F 9-12 month': 3,
    'G 12-15 month': 4,
    'H 15-18 month': 5,
    'I 18-21 month': 6,
    'J 21-24 month': 7,
    'K > 24 month': 8
}
tenure_label = st.selectbox("Ancienneté du client (TENURE)", list(ordre_tenure.keys()))
TENURE = ordre_tenure[tenure_label]

#=====Variable region==============

region_input = st.selectbox("Sélectionnez votre région", [
    "Dakar", "Thies", "Saint-Louis", "Fatick", "Kaolack", "Kolda", "Ziguinchor",
    "Matam", "Tambacounda", "Kaffrine", "Diourbel", "Louga", "Sédhiou", "Podor", "Non renseignée"
])

# Initialiser les colonnes de one-hot encoding (4 modalités conservées dans le modèle final)
REGION_DAKAR = 1 if region_input == "Dakar" else 0
REGION_THIES = 1 if region_input == "Thies" else 0
REGION_SAINT_LOUIS = 1 if region_input == "Saint-Louis" else 0
REGION_INCONNUE = 1 if region_input == "Non renseignée" else 0



# Création du DataFrame final


user_input = {
    'REGULARITY_CAPPED': REGULARITY_CAPPED,
    'REGION_INCONNUE': REGION_INCONNUE,
    'FREQ_TOP_PACK_CAPPED': FREQ_TOP_PACK_CAPPED,
    'ON_NET_CAPPED': ON_NET_CAPPED,
    'ORANGE_CAPPED': ORANGE_CAPPED,
    'DATA_VOLUME_CAPPED': DATA_VOLUME_CAPPED,
    'ARPU_SEGMENT_C': ARPU_SEGMENT_C,
    'FREQUENCE_CAPPED': FREQUENCE_CAPPED,
    'MONTANT_CAPPED': MONTANT_CAPPED,
    'TIGO_CAPPED': TIGO_CAPPED,
    'FREQUENCE_RECH_CAPPED': FREQUENCE_RECH_CAPPED,
    'REGION_DAKAR': REGION_DAKAR,
    'TENURE': TENURE,
    'REGION_THIES': REGION_THIES,
    'REGION_SAINT-LOUIS': REGION_SAINT_LOUIS
}


X_user = pd.DataFrame([user_input])


# Application du scaler
X_user_scaled = scaler.transform(X_user)


#st.write("Données après mise à l'échelle (standardisation):")
#st.dataframe(pd.DataFrame(X_user_scaled, columns=X_user.columns))

#Prediction
prediction = model.predict(X_user_scaled)
proba = model.predict_proba(X_user_scaled)[0][1]


st.subheader("Résultat de la prédiction :")
if prediction[0] == 1:
    st.error(f"⚠️ Ce client risque de se désabonner. Probabilité : {proba:.2%}")
else:
    st.success(f"✅ Ce client ne présente pas de risque élevé de churn. Probabilité : {proba:.2%}")


st.markdown("---")
st.subheader("🔗 Liens utiles")

st.markdown("""
- 💻 [Voir le code source sur GitHub](https://github.com/oceane1997/mon_app_CHURN.git)
- 📓 [Notebook Google Colab d'entraînement du modèle](https://colab.research.google.com/drive/16FLDUKy38w0e3_HtZIr0JO_23Mz6-5Aa?usp=sharing)
""")
