import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Atur opsi Pandas sekali saja di awal
pd.set_option('future.no_silent_downcasting', True)

@st.cache_data
def load_data():
    """Load dataset cuaca"""
    try:
        df = pd.read_excel('./data/Dataset_2020-2024.xlsx')
        return df
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan file 'Dataset_2020-2024.xlsx' ada di folder 'data/'")
        return None

def rr_categorization(rr):
    """Kategori curah hujan berdasarkan nilai RR"""
    try:
        rr = float(rr)
    except:
        return "Tidak Diketahui"
    if rr == 0:
        return 'No Rain'
    elif 0 < rr <= 20:
        return 'Hujan Ringan'
    elif 20 < rr <= 50:
        return 'Hujan Sedang'
    elif 50 < rr <= 100:
        return 'Hujan Lebat'
    else:
        return 'Hujan Sangat Lebat'


def data_page():
    st.title("📊 Halaman Data")
    
    st.markdown("""
    Dataset ini berisi data cuaca harian dari **Kabupaten Cilacap** selama periode **2020 hingga 2024**, dengan total sekitar **1.814 baris data** dan **6 parameter cuaca utama**, yaitu:

    - **TN**: Suhu Minimum Harian (°C)
    - **TX**: Suhu Maksimum Harian (°C)
    - **TAVG**: Suhu Rata-rata Harian (°C)
    - **RH_AVG**: Kelembapan Relatif Rata-rata (%)
    - **RR**: Curah Hujan (mm)
    - **SS**: Lama Penyinaran Matahari (jam)

  
     """)
    
    st.markdown("---")
    
    df = load_data()
    if df is None:
        return

    # Preprocessing awal
    df_clean = df.replace(['8888', '9999', '-', 8888, 9999], np.nan).infer_objects(copy=False)
    numeric_columns = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS']
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Konversi tanggal
    df_clean['TANGGAL'] = pd.to_datetime(df_clean['TANGGAL'], format='%d-%m-%Y', errors='coerce')
    df_clean['TAHUN'] = df_clean['TANGGAL'].dt.year

    # Distribusi kategori curah hujan
    df_clean['RR_KATEGORI'] = df_clean['RR'].apply(rr_categorization)
    fig_cat = px.pie(df_clean, names='RR_KATEGORI', title="Distribusi Kategori Curah Hujan")
    st.plotly_chart(fig_cat, use_container_width=True, key="pie_rr_kategori")

    # Tabs
    tab1, tab2 = st.tabs(["📋 Sample Data", "🗓️ Trend Temporal"])

    with tab1:
        # col1, col2 = st.columns(2)

        # with col1:
            # st.subheader("📊 Informasi Dataset")
            # st.write(f"**Jumlah Baris**: {len(df_clean):,}")
            # st.write(f"**Jumlah Kolom**: {len(df_clean.columns)}")
            # st.write(f"**Periode Data**: {df_clean['TANGGAL'].min().date()} - {df_clean['TANGGAL'].max().date()}")

            # # # Missing values
            # st.subheader("❌ Missing Values")
            # missing_data = df_clean.isnull().sum()
            # for col, missing in missing_data.items():
            #     if missing > 0:
            #         st.write(f"**{col}**: {missing} ({missing/len(df_clean)*100:.1f}%)")

        # with col2:
            st.subheader("📋 Sample Data")
            st.dataframe(df_clean.head(10), use_container_width=True)

    with tab2:
        st.subheader("🗓️ Trend Temporal")
        df_clean = df_clean.sort_values('TANGGAL')

        param_ts = st.selectbox("Pilih Parameter:", numeric_columns, key="ts_param")
        
        fig_ts = px.line(df_clean, x='TANGGAL', y=param_ts,
                         title=f"Trend {param_ts} dari Waktu ke Waktu")
        st.plotly_chart(fig_ts, use_container_width=True, key="trend_harian")

        df_monthly = df_clean.groupby(df_clean['TANGGAL'].dt.to_period('M'))[numeric_columns].mean()
        df_monthly.index = df_monthly.index.astype(str)

        fig_monthly = px.line(df_monthly, x=df_monthly.index, y=param_ts,
                              title=f"Rata-rata Bulanan {param_ts}")
        st.plotly_chart(fig_monthly, use_container_width=True, key="trend_bulanan")

        st.subheader("📅 Total Curah Hujan per Tahun")
        rr_per_year = df_clean.groupby('TAHUN')['RR'].sum().reset_index()
        fig_yearly_rr = px.bar(rr_per_year, x='TAHUN', y='RR', text_auto=True,
                               title="Total Curah Hujan Tahunan di Cilacap (mm)",
                               labels={"RR": "Curah Hujan (mm)", "TAHUN": "Tahun"})
        st.plotly_chart(fig_yearly_rr, use_container_width=True, key="rr_tahunan")

        # Insight Otomatis
        if not rr_per_year.empty:
            max_year = rr_per_year.loc[rr_per_year['RR'].idxmax()]
            min_year = rr_per_year.loc[rr_per_year['RR'].idxmin()]
            st.info(f"Tahun dengan curah hujan tertinggi: **{int(max_year['TAHUN'])}** ({max_year['RR']:.2f} mm)")
            st.info(f"Tahun dengan curah hujan terendah : **{int(min_year['TAHUN'])}** ({min_year['RR']:.2f} mm)")

# Jalankan halaman
data_page()
