import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Classificador de Cancros de Pele", layout="centered")

# Classes (ajusta conforme as tuas labels)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Modelos disponíveis
modelos_disponiveis = {
    "Modelo S - Adam + KLD (2B1)": "modelS_2B1_com_data_aug_adam_KLD.keras",
    "Modelo S - RMS + CatCross (2B2)": "modelS_2B2_com_data_aug_RMS_cat_cross_best_acc.keras",
    "Modelo S - SGD + MSE (2B3)": "modelS_2B3_com_data_aug_SGD_MSE_worst_acc.keras",
    "Modelo S - Optuna + Adam (2C)": "modelS_2C_optuna_com_data_aug_adam_cat_cross.keras",
    "Modelo T - Adam + CatCross (3A)": "modelT_3A_com_data_aug_adam_cat_cross_best_acc.keras",
    "Modelo T - Optuna + RMS (3B)": "modelT_3B_optuna_sem_data_aug_RMS_SGD.keras"
}

@st.cache_resource
def carregar_modelo(path):
    return tf.keras.models.load_model(path)

st.title("🔬 Classificador de Cancros de Pele")
st.markdown("Seleciona um modelo e envia uma imagem.")

# Escolha do modelo
escolha = st.selectbox("Seleciona o modelo:", list(modelos_disponiveis.keys()))
modelo = carregar_modelo(modelos_disponiveis[escolha])

# Upload de imagem
uploaded_file = st.file_uploader("Carrega uma imagem:", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    if st.button("Classificar"):
        st.spinner("A classificar...")
        input_img = preprocess_image(image)
        preds = modelo.predict(input_img)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        st.success(f"🧬 Classe predita: **{predicted_class.upper()}**")
        st.info(f"Confiança: {confidence:.2%}")
