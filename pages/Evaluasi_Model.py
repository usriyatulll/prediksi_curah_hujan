import streamlit as st
import pandas as pd
import plotly.express as px
import joblib, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, '..', 'model')
    data_dir  = os.path.join(base, '..', 'data')
    try:
        models = {
            'Naive Bayes'        : joblib.load(os.path.join(model_dir, 'nb_model.pkl')),
            'K-Nearest Neighbors': joblib.load(os.path.join(model_dir, 'knn_model.pkl')),
            'Random Forest'      : joblib.load(os.path.join(model_dir, 'rf_model.pkl')),
            'Stacking'           : joblib.load(os.path.join(model_dir, 'stacking_model.pkl')),  # ⬅️ Tambahan
        }
        le      = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        X_test  = pd.read_csv(os.path.join(data_dir,  'X_test_preprocessed.csv'))
        y_test  = pd.read_csv(os.path.join(data_dir,  'y_test_preprocessed.csv'))['RR_KAT_ENC']
        return models, le, X_test, y_test
    except FileNotFoundError as e:
        st.error(f"Artefak tidak ditemukan: {e}")
        return None, None, None, None

def page():
    st.title("📈 Evaluasi Model")

    models, le, X_test, y_test = load_artifacts()
    if models is None:
        st.stop()

    # === Hitung metrik ===
    metrics, matrices = {}, {}
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)
        metrics[name] = {
            'Accuracy' : accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall'   : recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1'       : f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        matrices[name] = confusion_matrix(y_test, y_pred)

    # === Tampilkan Perbandingan Metrik
    st.subheader("📋 Perbandingan Metrik")
    st.dataframe(pd.DataFrame(metrics).T.round(3), use_container_width=True)
    best = max(metrics, key=lambda k: metrics[k]['F1'])
    df_metric = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    fig_bar = px.bar(df_metric, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1'], barmode='group',
                    title="Perbandingan Metrik Evaluasi Tiap Model")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.success(f"Model terbaik berdasarkan F1‑Score ➡️ **{best}** ({metrics[best]['F1']:.3f})")

    st.divider()
    st.subheader("🔄 Confusion Matrix")
    choice = st.selectbox("Pilih Model:", list(models.keys()))
    cm      = matrices[choice]
    cats    = le.classes_
    fig = px.imshow(cm, text_auto=True, x=cats, y=cats,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix – {choice}")
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)

    # Penjelasan model
    explanations = {
        "Naive Bayes": """
**Naive Bayes**  
Model ini memiliki banyak nilai di luar diagonal → sering salah membedakan kelas yang mirip (*Hujan Ringan* vs *Hujan Sedang*).  
Cocok sebagai baseline, tetapi kurang akurat untuk pola curah hujan kompleks.
        """,
        "K-Nearest Neighbors": """
**K-Nearest Neighbors (KNN)**  
Hampir semua prediksi tepat di diagonal → akurasi sangat tinggi.  
KNN berhasil menangkap pola lokal; sangat baik membedakan *No Rain* dan *Hujan Sangat Lebat*.
        """,
        "Random Forest": """
**Random Forest**  
Prediksi mayoritas benar; hanya sedikit kesalahan pada *Hujan Ringan* → *Hujan Sedang*.  
Model stabil, tahan outlier, dan seimbang antara akurasi & efisiensi.
        """,
        "Stacking": """
**Stacking Ensemble**  
Menggabungkan kekuatan beberapa model dasar untuk meningkatkan generalisasi.  
Performa terbaik karena mampu menangkap berbagai pola dari model-model base learner.
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
