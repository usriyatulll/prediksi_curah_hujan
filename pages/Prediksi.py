import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# === Load Model & Scaler ===
base = os.path.dirname(os.path.abspath(__file__))
model_paths = {
    "Random Forest": os.path.join(base, '..', 'model', 'rf_model.pkl'),
    "KNN": os.path.join(base, '..', 'model', 'knn_model.pkl'),
    "Naive Bayes": os.path.join(base, '..', 'model', 'nb_model.pkl'),
    "Stacking Classifier": os.path.join(base, '..', 'model', 'stacking_model.pkl'),
}
scaler_path = os.path.join(base, '..', 'model', 'scaler.pkl')
le_path = os.path.join(base, '..', 'model', 'label_encoder.pkl')

# Load model dan scaler
models = {name: joblib.load(path) for name, path in model_paths.items()}
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

# === UI ===
st.title("🔮 Prediksi Curah Hujan Beberapa Hari ke Depan")
st.header("🧾 Input Data Hari Ini")

# Pilih model
model_labels = {
    "Stacking (NB+KNN+RF) (Direkomendasikan)": "Stacking Classifier",
    "Random Forest": "Random Forest",
    "KNN": "KNN",
    "Naive Bayes": "Naive Bayes"
   
}
model_selected_label = st.selectbox("🧠 Pilih Model Prediksi", list(model_labels.keys()))
model_selected = model_labels[model_selected_label]

# Input Data
col1, col2 = st.columns(2)
with col1:
    tn = st.number_input("Suhu Minimum (°C)", min_value=0.0, value=23.2)
    tx = st.number_input("Suhu Maksimum (°C)", min_value=0.0, value=32.1)
    tavg = st.number_input("Rata-rata Suhu (°C)", min_value=0.0, value=28.0)

with col2:
    rh = st.number_input("Kelembaban Rata-rata (%)", min_value=0.0, value=79.0)
    ss = st.number_input("Lama Penyinaran Matahari (jam)", min_value=0.0, value=1.0)
    rr = st.number_input("Curah Hujan Hari Ini (mm)", min_value=0.0, value=16.5)

n_days = st.slider("🔁 Prediksi Berapa Hari ke Depan?", min_value=1, max_value=7, value=3)

emoji_dict = {
    'Tidak Hujan': '☀️',
    'Hujan Ringan': '🌦️',
    'Hujan Sedang': '🌧️',
    'Hujan Lebat': '⛈️'
}
tips_dict = {
    'Tidak Hujan': 'Cuaca cerah, cocok untuk aktivitas luar ruangan.',
    'Hujan Ringan': 'Bawa payung kecil, kemungkinan hujan ringan.',
    'Hujan Sedang': 'Gunakan jas hujan atau payung, hujan sedang mungkin terjadi.',
    'Hujan Lebat': 'Waspada genangan, hindari berkendara terlalu cepat.'
}

# === Tombol Prediksi ===
if st.button("🔍 Prediksi"):
    try:
        hasil_prediksi = []
        tanggal_awal = datetime.today()

        # Inisialisasi histori
        hist = {'TN': [tn], 'TX': [tx], 'TAVG': [tavg],
                'RH_AVG': [rh], 'SS': [ss], 'RR': [rr]}

        for i in range(n_days):
            fitur = {
                'TN': hist['TN'][-1], 'TX': hist['TX'][-1], 'TAVG': hist['TAVG'][-1],
                'RH_AVG': hist['RH_AVG'][-1], 'SS': hist['SS'][-1],
                'RR_lag1': hist['RR'][-1], 'TAVG_lag1': hist['TAVG'][-1],
                'RH_AVG_lag1': hist['RH_AVG'][-1], 'SS_lag1': hist['SS'][-1],
                'TX_lag1': hist['TX'][-1], 'TN_lag1': hist['TN'][-1],
                'RR_rolling_mean_3d_lag1': np.mean(hist['RR'][-3:]),
            }

            df_fitur = pd.DataFrame([fitur])
            scaled = scaler.transform(df_fitur)

            model = models[model_selected]
            pred_enc = model.predict(scaled)[0]
            pred_label = le.inverse_transform([pred_enc])[0]
            tanggal = tanggal_awal + timedelta(days=i + 1)
            hasil_prediksi.append((tanggal.strftime('%d-%m-%Y'), pred_label))

            # Tambahkan variasi ringan untuk simulasi hari selanjutnya
            for key in hist:
                hist[key].append(hist[key][-1] + np.random.normal(0, 0.2))

        st.session_state['hasil_prediksi'] = hasil_prediksi
        st.session_state['hist'] = hist

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")

# === Tampilkan Hasil Prediksi ===
if 'hasil_prediksi' in st.session_state:
    hasil_prediksi = st.session_state['hasil_prediksi']
    hist = st.session_state['hist']

    st.markdown(f"### 📆 Ringkasan Prediksi Cuaca ({model_selected})")
    for i, (tgl, pred) in enumerate(hasil_prediksi):
        icon = emoji_dict.get(pred, '☁️')
        tips = tips_dict.get(pred, '')
        suhu = hist['TAVG'][i + 1]
        st.markdown(f"""
        <div style='display: flex; align-items: center; justify-content: space-between; 
                    padding: 12px; margin: 6px 0; border-radius: 10px; 
                    background: #f5f5f5; border: 1px solid #ddd;'>
            <div style='font-weight: bold; width: 100px;'>{tgl}</div>
            <div style='font-size: 32px; width: 50px;'>{icon}</div>
            <div style='width: 130px;'>{pred}</div>
            <div style='width: 140px;'>🌡️ {suhu:.1f}°C</div>
            <div style='flex: 1; color: #666; font-size: 14px;'>{tips}</div>
        </div>
        """, unsafe_allow_html=True)

    # Peta Lokasi
    st.subheader("🗺️ Lokasi Prediksi Cuaca")
    st.markdown('<p style="margin-top: -10px; font-size: 0.9em;">📍 Kabupaten Cilacap, Jawa Tengah</p>', unsafe_allow_html=True)
    m = folium.Map(location=[-7.719, 109.015], zoom_start=10)
    folium.Marker(
        [-7.719, 109.015],
        popup="Kabupaten Cilacap",
        tooltip="Prediksi Cuaca",
        icon=folium.Icon(color="blue", icon="cloud")
    ).add_to(m)
    st_folium(m, width=700, height=400)
