import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model dan scaler
model_dt = pickle.load(open("model_dt.pkl", "rb"))
scaler_dt = pickle.load(open("scaler_dt.pkl", "rb"))

st.title("Aplikasi Klasifikasi Time Series Deteksi Penuaan Tulang")
st.write("Masukkan data time series sebanyak 80 kolom, dipisahkan spasi, atau upload file CSV")

#tab
tab_manual, tab_file = st.tabs(["Input Manual", "Upload File CSV"])

# fungsi scaler dan Model
def predict_from_array(data_array):
    data_scaled = scaler_dt.transform(data_array)
    prediction = model_dt.predict(data_scaled)
    return prediction


# Tab Input Manual
with tab_manual:
    st.subheader("Input Manual")
    input_text = st.text_area("Masukkan 80 data time series (dipisahkan spasi):")
    
    if st.button("Prediksi (Manual)"):
        try:
            data = [float(x) for x in input_text.split()]
            if len(data) != 80:
                st.error("Input manual harus tepat 80 data")
            else:
                data_array = np.array(data).reshape(1, -1)
                pred = predict_from_array(data_array)
                if pred[0] == 1.0:
                    st.success(f"Hasil Prediksi: {pred[0]} (Benar)")
                else:
                    st.success(f"Hasil Prediksi: {pred[0]} (Salah)")
        except ValueError:
            st.error("Pastikan input hanya angka dan dipisahkan spasi")

# Tab Upload File CSV
with tab_file:
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Upload file CSV (1 baris = 1 sampel, 80 kolom)", type=["csv"])
    
    if st.button("Prediksi (File)"):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                if df.shape[1] != 80:
                    st.error("File harus memiliki tepat 80 kolom")
                else:
                    pred = predict_from_array(df.values)
                    for i, pred in enumerate(pred):
                        if pred == 1.0:
                            st.success(f"Hasil Prediksi Baris {i+1}: {pred} (Benar)")
                        else:
                            st.success(f"Hasil Prediksi {i+1}: {pred} (Salah)")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan upload file CSV terlebih dahulu")
