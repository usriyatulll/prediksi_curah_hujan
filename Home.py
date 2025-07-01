import streamlit as st

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

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Selamat Datang di Aplikasi Prediksi Curah Hujan

    Aplikasi ini menggunakan **Machine Learning** untuk memprediksi kategori curah hujan berdasarkan data cuaca historis.

    ### 🎯 Fitur Utama:
    - **Analisis Data Cuaca**: Eksplorasi data cuaca historis 2020-2024
    - **Evaluasi Model**: Perbandingan performa 3 algoritma ML
    - **Prediksi Real-time**: Prediksi curah hujan untuk hari-hari mendatang

    ### 🤖 Algoritma yang Digunakan:
    1. **Naive Bayes** – Algoritma probabilistik sederhana dan efektif
    2. **K-Nearest Neighbors (KNN)** – Klasifikasi berdasarkan tetangga terdekat
    3. **Random Forest** – Ensemble learning dengan multiple decision trees

    ### 📊 Kategori Prediksi:
    - **No Rain**: 0 mm
    - **Hujan Ringan**: 0–20 mm
    - **Hujan Sedang**: 20–50 mm
    - **Hujan Lebat**: 50–100 mm
    - **Hujan Sangat Lebat**: >100 mm
    """)

with col2:
    st.markdown("### 🌡️ Parameter Cuaca")
    st.info("""
    **TN**: Suhu Minimum (°C)  
    **TX**: Suhu Maksimum (°C)  
    **TAVG**: Suhu Rata-rata (°C)  
    **RH_AVG**: Kelembaban Rata-rata (%)  
    **RR**: Curah Hujan (mm)  
    **SS**: Lama Penyinaran Matahari (jam)
    """)

    st.success("🚀 Mulai eksplorasi dengan memilih halaman di sidebar kiri!")
    
with st.expander("ℹ️ Informasi Tambahan"):
    st.markdown("""
**Sumber Data**: BMKG Stasiun Meteorologi Cilacap  
**Periode Data**: Januari 2020 – Mei 2024  
**Kontak Pengembang**: khamimahusriyatul@gmail.com  
""")

with st.expander("❓ Cara Menggunakan"):
    st.markdown("""
1. Gunakan menu **📊 Halaman Data** untuk eksplorasi dataset.  
2. Lihat **📈 Evaluasi Model** untuk perbandingan performa.  
3. Gunakan **🔮 Prediksi** untuk memprediksi curah hujan beberapa hari ke depan.
""")

