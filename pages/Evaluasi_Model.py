import streamlit as st
import pandas as pd
import plotly.express as px
import os

def load_evaluation_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'model_evaluation_metrics.csv')
    
    if not os.path.exists(data_path):
        st.error("❌ File metrik evaluasi tidak ditemukan. Pastikan Anda sudah menyimpannya dari training.")
        return None

    df = pd.read_csv(data_path)

    # Normalisasi kolom agar kompatibel
    if 'F1-Score' in df.columns:
        df = df.rename(columns={'F1-Score': 'F1'})

    return df

def page():
    st.title("📈 Evaluasi Model Machine Learning")

    df_metrics = load_evaluation_data()
    if df_metrics is None:
        st.stop()

    # Menentukan model terbaik
    try:
        best_row = df_metrics.loc[df_metrics['F1'].idxmax()]
        best_model = best_row['Model']
    except KeyError:
        st.error("❌ Kolom 'F1' tidak ditemukan. Pastikan file CSV memiliki header: Model, Accuracy, Precision, Recall, F1(-Score)")
        st.stop()

    # === Tampilkan Perbandingan Metrik
    st.subheader("📋 Perbandingan Metrik")
    st.dataframe(df_metrics.round(3), use_container_width=True)

    fig_bar = px.bar(
        df_metrics, 
        x='Model', 
        y=['Accuracy', 'Precision', 'Recall', 'F1'], 
        barmode='group',
        title="Perbandingan Metrik Evaluasi Tiap Model"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.success(f"Model terbaik berdasarkan F1‑Score ➡️ **{best_model}** ({best_row['F1']:.3f})")

    # === Tampilkan Confusion Matrix sebagai gambar
    st.divider()
    st.subheader("🔄 Visualisasi Confusion Matrix")
    choice = st.selectbox("Pilih Model:", df_metrics['Model'].tolist())

    image_filename = f"confusion_matrix_{choice.lower().replace(' ', '_')}.png"
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', image_filename)

    if os.path.exists(image_path):
        st.image(image_path, caption=f'Confusion Matrix – {choice}', use_container_width=True)
    else:
        st.warning(f"⚠️ Gambar confusion matrix untuk model {choice} tidak ditemukan: `{image_filename}`")

    # Penjelasan model (opsional)
    explanations = {
        "Naive Bayes": """
**Naive Bayes**  
Sederhana dan cepat, namun kurang akurat untuk data yang kompleks. Banyak kesalahan di kelas yang mirip.
        """,
        "K-Nearest Neighbors": """
**K-Nearest Neighbors (KNN)**  
Sangat akurat bila data bersih dan seimbang. Mampu membedakan kelas ekstrem dengan baik.
        """,
        "Random Forest": """
**Random Forest**  
Kuat terhadap noise dan overfitting. Performa stabil dan bisa dijelaskan.
        """,
        "Stacking": """
**Stacking Ensemble**  
Menggabungkan banyak model dasar. Biasanya memberikan hasil terbaik dan generalisasi kuat.
        """
    }
    st.markdown(explanations.get(choice, ""))

    with st.expander("ℹ️ Cara Membaca Confusion Matrix"):
        st.markdown("""
- **Baris** = label **aktual** (ground truth)  
- **Kolom** = label **prediksi** model  
- Nilai besar di **diagonal** ⇒ prediksi benar  
- Nilai di luar diagonal ⇒ kesalahan klasifikasi  
Semakin pekat warna diagonal, semakin baik performa model.
        """)

if __name__ == "__main__":
    page()
