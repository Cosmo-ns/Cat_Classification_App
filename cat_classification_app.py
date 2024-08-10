import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# モデルのロード
model = load_model('cat_classifier_modell_n_s_e3.h5')

# 画像サイズの定義
IMG_SIZE = (416, 416)

def classify_image(uploaded_file, model):
    # 画像の前処理
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # バッチ次元の追加
    img /= 255.0  # 正規化
    predictions = model.predict(img)
    return predictions

# Streamlit アプリ
st.title("Cat Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    predictions = classify_image(uploaded_file, model)
    
    # st.write(f"Predictions shape: {predictions.shape}")
    # st.write(f"Predictions content: {predictions}")
    
    # `predictions`の形状が1出力である場合（つまり、1つの確率である場合）
    if predictions.shape[1] == 1:  
        cat2_prob = predictions[0][0]
        cat1_prob = 1.0 - cat2_prob
        
        st.write(f"Cat 1: {cat1_prob * 100:.2f}%")
        st.write(f"Cat 2: {cat2_prob * 100:.2f}%")
    else:
        st.write("Unexpected prediction shape:", predictions.shape)
