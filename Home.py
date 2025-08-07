import streamlit as st
import folium
from streamlit_folium import st_folium

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Curah Hujan",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Halaman Home (tanpa sidebar navigation lagi)
st.title("🌧️ Sistem Prediksi Curah Hujan")
st.markdown("---")

# Layout 2 kolom
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Selamat Datang di Aplikasi Prediksi Curah Hujan

    Aplikasi ini menggunakan **Machine Learning** untuk memprediksi kategori curah hujan berdasarkan data cuaca historis.

    ### 🎯 Fitur Utama:
    - **Analisis Data Cuaca**: Eksplorasi data cuaca historis 2020–2024  
    - **Evaluasi Model**: Perbandingan performa 3 algoritma ML  
    - **Prediksi**: Simulasi prediksi curah hujan untuk hari-hari mendatang  

    ### 🤖 Algoritma yang Digunakan:
    1. **Naive Bayes** – Algoritma probabilistik sederhana dan efektif  
    2. **K-Nearest Neighbors (KNN)** – Klasifikasi berdasarkan tetangga terdekat  
    3. **Random Forest** – Ensemble learning dengan multiple decision trees  
    4. **Stacking Classifier** – Kombinasi dari model-model di atas untuk performa terbaik  

    ### 📊 Kategori Prediksi:
    - **No Rain**: 0 mm  
    - **Hujan Ringan**: 0–20 mm  
    - **Hujan Sedang**: 20–50 mm  
    - **Hujan Lebat**: 50–100 mm  
    - **Hujan Sangat Lebat**: >100 mm  
    """)

    # Peta Lokasi
    st.subheader("🗺️ Lokasi Prediksi Cuaca")
    st.markdown(
        '<p style="margin-top: -10px; font-size: 0.9em;">📍 Kabupaten Cilacap, Jawa Tengah</p>',
        unsafe_allow_html=True
    )

    m = folium.Map(location=[-7.719, 109.015], zoom_start=10)
    folium.Marker(
        [-7.719, 109.015],
        popup="Kabupaten Cilacap",
        tooltip="Prediksi Cuaca",
        icon=folium.Icon(color="blue", icon="cloud")
    ).add_to(m)
    st_folium(m, width=700, height=400)

with col2:
    st.markdown("### 🌡️ Parameter Curah Hujan")
    st.info("""
    **TN**: Suhu Minimum (°C)  
    **TX**: Suhu Maksimum (°C)  
    **TAVG**: Suhu Rata-rata (°C)  
    **RH_AVG**: Kelembaban Rata-rata (%)  
    **RR**: Curah Hujan (mm)  
    **SS**: Lama Penyinaran Matahari (jam)
    """)
    # Petunjuk Navigasi
st.markdown("""
<div style='padding: 1rem; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 10px'>
<p><b>Cara Menggunakan Aplikasi:</b></p>
<ol>
  <li><b>📊 Halaman Data</b> – untuk eksplorasi dataset</li>
  <li><b>📈 Evaluasi Model</b> – untuk melihat perbandingan performa</li>
  <li><b>🔮 Prediksi</b> – untuk memprediksi curah hujan ke depan</li>
</ol>
</div>
""", unsafe_allow_html=True)



st.markdown("---")
st.caption("👩‍💻 Dibuat oleh Usriyatul Khamimah – khamimahusriyatul@gmail.com")
