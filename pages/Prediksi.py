import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# === Load model dan scaler ===
base = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base, '..', 'model', 'random_forest_model.pkl')
scaler_path = os.path.join(base, '..', 'model', 'scaler.pkl')
le_path = os.path.join(base, '..', 'model', 'label_encoder.pkl')

rf = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

# === UI ===
st.title("🔮 Prediksi Curah Hujan Beberapa Hari ke Depan")
st.header("🧾 Input Data Hari Ini")

col1, col2 = st.columns(2)
with col1:
    tn = st.number_input("TN (°C)", 0.0, 40.0, 24.0)
    tx = st.number_input("TX (°C)", 0.0, 45.0, 32.0)
    tavg = st.number_input("TAVG (°C)", 0.0, 42.0, 28.0)
with col2:
    rh = st.number_input("RH_AVG (%)", 0.0, 100.0, 80.0)
    ss = st.number_input("SS (jam)", 0.0, 15.0, 5.0)
    rr = st.number_input("RR (Curah Hujan Hari Ini, mm)", 0.0, 200.0, 10.0)

n_days = st.slider("🔁 Prediksi Berapa Hari ke Depan?", 1, 7, 3)

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
if st.button("Prediksi"):
    hasil_prediksi = []
    tanggal_awal = datetime.today()
    hist = {'TN': [tn], 'TX': [tx], 'TAVG': [tavg], 'RH_AVG': [rh], 'SS': [ss], 'RR': [rr]}

    for i in range(n_days):
        fitur = {
            'TN': hist['TN'][-1], 'TX': hist['TX'][-1], 'TAVG': hist['TAVG'][-1],
            'RH_AVG': hist['RH_AVG'][-1], 'SS': hist['SS'][-1],
            'RR_lag1': hist['RR'][-1], 'TAVG_lag1': hist['TAVG'][-1],
            'RH_AVG_lag1': hist['RH_AVG'][-1], 'SS_lag1': hist['SS'][-1],
            'TX_lag1': hist['TX'][-1], 'TN_lag1': hist['TN'][-1],
            'RR_rolling_mean_3d_lag1': np.mean(hist['RR'][-3:])
        }

        df_fitur = pd.DataFrame([fitur])
        scaled = scaler.transform(df_fitur)
        pred_enc = rf.predict(scaled)[0]
        pred_lab = le.inverse_transform([pred_enc])[0]
        tanggal = tanggal_awal + timedelta(days=i + 1)
        hasil_prediksi.append((tanggal.strftime('%d-%m-%Y'), pred_lab))

        for key in hist:
            hist[key].append(hist[key][-1])  # Simpan prediksi sebagai input berikutnya

    st.session_state['hasil_prediksi'] = hasil_prediksi
    st.session_state['hist'] = hist

# === Tampilkan Prediksi Jika Ada ===
if 'hasil_prediksi' in st.session_state:
    hasil_prediksi = st.session_state['hasil_prediksi']
    hist = st.session_state['hist']

    st.markdown("### 📆 Ringkasan Prediksi Cuaca")
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

# Subheader dan peta hanya muncul jika prediksi tersedia
    st.subheader("🗺️ Lokasi Prediksi Cuaca")

    with st.container():
        m = folium.Map(location=[-7.719, 109.015], zoom_start=10)
        folium.Marker(
            [-7.719, 109.015],
            popup="Kabupaten Cilacap",
            tooltip="Prediksi Cuaca",
            icon=folium.Icon(color="blue", icon="cloud")
        ).add_to(m)

        st_folium(m, width=700, height=400)

        st.markdown(
            '<p style="margin-top: -10px; font-size: 0.9em;">📍 Kabupaten Cilacap, Jawa Tengah</p>',
            unsafe_allow_html=True
        )
