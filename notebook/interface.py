# app.py

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Silver LSTM Dashboard", layout="wide")

# ========================
# TITRE PRINCIPAL
# ========================
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📈 Prédiction du prix du Silver avec LSTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Interface professionnelle pour prédire le prix du Silver avec un modèle LSTM entraîné.</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================
# CHARGEMENT DU MODELE ET SCALERS
# ========================
@st.cache_resource
def load_model_scalers():
    model = tf.keras.models.load_model("silver_lstm.keras", compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_scalers()

# ========================
# SECTION INPUT
# ========================
st.subheader("1️⃣ Entrée des données")
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    - Entrez **les 30 dernières valeurs du Silver**, séparées par des virgules.
    - Exemple : 15.4,15.2,15.8,...,16.0
    """)
    values_input = st.text_area("Valeurs du Silver :", 
    "15.4,15.2,15.8,16.0,15.7,15.5,15.6,15.8,16.1,15.9,15.7,15.6,15.5,15.4,15.6,15.7,15.8,16.0,15.9,15.8,15.7,15.6,15.5,15.4,15.6,15.7,15.8,16.0,15.9,15.8"
    )

with col2:
    st.subheader("Paramètres")
    n_days = st.slider("Nombre de jours à prédire :", min_value=1, max_value=60, value=30)

st.markdown("---")

# ========================
# BOUTON PRÉDICTION
# ========================
if st.button("🚀 Prédire le prix du Silver"):
    try:
        x = np.array([float(v.strip()) for v in values_input.split(",")])
        if len(x) != 30:
            st.error("⚠️ Le modèle attend exactement 30 valeurs !")
        else:
            # Normalisation
            x_scaled = scaler_X.transform(x.reshape(1, -1)).reshape(1, 30, 1)

            # Prédictions futures
            predictions_scaled = []
            current_data = x_scaled.copy()
            for i in range(n_days):
                pred = model.predict(current_data, verbose=0)
                predictions_scaled.append(pred[0][0])
                current_data = np.roll(current_data, -1)
                current_data[0, -1, 0] = pred[0][0]

            future_predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1,1)).flatten()
            last_date = pd.Timestamp.today()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days)

            st.success(f"✅ Prédictions pour les {n_days} prochains jours générées !")

            # ========================
            # GRAPHIQUE PROFESSIONNEL
            # ========================
            fig = go.Figure()
            # Historique (30 derniers jours)
            fig.add_trace(go.Scatter(
                y=x, 
                x=pd.date_range(end=last_date, periods=30),
                mode='lines+markers',
                name='Historique',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            # Prédictions futures
            fig.add_trace(go.Scatter(
                y=future_predictions,
                x=future_dates,
                mode='lines+markers',
                name='Prédictions LSTM',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(symbol='circle', size=6)
            ))

            # Layout
            fig.update_layout(
                title="📊 Prix du Silver : Historique vs Prédictions LSTM",
                xaxis_title="Date",
                yaxis_title="Prix Silver ($/oz)",
                template="plotly_white",
                legend=dict(x=0.02, y=0.98),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ========================
            # EXPORT CSV
            # ========================
            if st.button("📥 Exporter les prédictions"):
                df_pred = pd.DataFrame({'Date': future_dates, 'Predicted_Silver': future_predictions})
                df_pred.to_csv("predictions_silver.csv", index=False)
                st.success("Fichier predictions_silver.csv généré ✅")

    except Exception as e:
        st.error(f"❌ Erreur : format incorrect ou problème de saisie.\nDétails : {e}")

st.markdown("---")

# ========================
# RÉSUMÉ DU MODÈLE
# ========================
st.subheader("3️⃣ Architecture du modèle LSTM")
if st.checkbox("Afficher le résumé du modèle"):
    model.summary(print_fn=lambda x: st.text(x))
