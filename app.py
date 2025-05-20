import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
import time

# Configuration page
st.set_page_config(page_title="Advanced Fruit Classifier", page_icon="🍏", layout="wide")

# Charger modèle MobileNetV2 pré-entraîné
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# Titre + description
st.title("🍎 Advanced Fruit Image Classifier with MobileNetV2")
st.markdown("""
Bienvenue dans cette application qui permet de classifier des images de fruits et légumes
en utilisant un modèle MobileNetV2 pré-entraîné sur ImageNet.  
Cette application vous permet de télécharger une image, de lancer une prédiction manuelle,  
et d’afficher des résultats clairs et détaillés.

> **Note :** Le modèle fonctionne mieux avec des images claires et bien cadrées.
""")

# Upload image
uploaded_file = st.file_uploader("📤 Choisissez une image de fruit ou légume au format JPG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
    except UnidentifiedImageError:
        st.error("⚠️ L’image chargée est corrompue ou invalide. Veuillez réessayer avec une autre image.")
        st.stop()

    # Mise en page colonnes
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image téléchargée")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Paramètres de prédiction")
        top_k = st.slider("Nombre de classes à afficher", min_value=3, max_value=10, value=5, step=1)

        if st.button("▶️ Lancer la prédiction"):
            with st.spinner("Analyse de l'image en cours..."):
                # Redimensionner et prétraiter
                img_resized = img.resize((224, 224))
                x = image.img_to_array(img_resized)
                x = preprocess_input(np.expand_dims(x, axis=0))

                # Simulation de progression avec barre
                progress_bar = st.progress(0)
                for i in range(5):
                    time.sleep(0.3)
                    progress_bar.progress((i + 1) * 20)

                # Prédiction
                preds = model.predict(x)
                decoded_preds = decode_predictions(preds, top=top_k)[0]

                st.success("✅ Prédiction terminée ! Résultats :")

                # Affichage avec barres de progression et styles
                for i, (imagenet_id, label, confidence) in enumerate(decoded_preds):
                    percent = confidence * 100
                    st.markdown(f"""
                        <div style='background:#E8F0FE; border-radius:8px; padding:10px; margin-bottom:8px;'>
                        <strong>{i+1}. {label.replace('_', ' ').title()}</strong> — {percent:.2f}%
                        <div style='background:#D0E1FD; border-radius:5px; margin-top:5px;'>
                            <div style='width:{percent}%; background:#2E86C1; padding:5px 0; border-radius:5px;'></div>
                        </div>
                        </div>
                    """, unsafe_allow_html=True)

else:
    st.info("ℹ️ Veuillez uploader une image pour commencer.")

# Footer
st.markdown("---")
st.markdown("Developpé par **TonNom** | Modèle MobileNetV2 - TensorFlow & Streamlit")
