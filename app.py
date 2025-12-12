import streamlit as st
import numpy as np
import re
import joblib


model = joblib.load("cluster_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Segmentasi UMKM Berdasarkan Perilaku Penjualan",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Segmentasi UMKM Berdasarkan Perilaku Penjualan")
st.write("Aplikasi ini digunakan untuk memetakan UMKM berdasarkan pola penjualannya menggunakan metode *clustering*.")

st.markdown("---")

st.subheader("💡 Masukkan Data Perilaku Penjualan")

col1, col2 = st.columns(2)

with col1:
    total_items = st.number_input("Jumlah Item Terjual (Total_Items)", min_value=0, max_value=1000, step=1)
def convert_rupiah_to_number(value):
    cleaned = re.sub(r"[^0-9]", "", value)
    if cleaned == "":
        return 0
    return int(cleaned)

rupiah_input = st.text_input("Total Pendapatan (Rp)", "Rp 0")

total_cost = convert_rupiah_to_number(rupiah_input)

st.write("Angka bersih:", total_cost)

with col2:
    product_count = st.number_input("Jumlah Jenis Produk (Product_Count)", min_value=1, max_value=100, step=1)
    discount = st.selectbox("Diskon Diberikan?", ["Tidak", "Ya"])

discount_flag = 1 if discount == "Ya" else 0

# Data yang dipakai K-Means (utama)
data = np.array([[total_items, total_cost, product_count]])

if st.button("🔍 Prediksi Cluster"):

    scaled_data = scaler.transform(data)

    cluster = model.predict(scaled_data)[0]

    st.markdown("---")
    st.subheader("📌 Hasil Segmentasi UMKM")

    st.success(f"UMKM ini termasuk dalam **Cluster {cluster}**")

    cluster_desc = {
        0: "Cluster 0 → Penjualan rendah, transaksi jarang.",
        1: "Cluster 1 → Penjualan sedang dan stabil.",
        2: "Cluster 2 → Penjualan tinggi dengan variasi produk tinggi.",
        3: "Cluster 3 → Penjualan sangat tinggi dan frekuensi transaksi rutin."
    }

    if cluster in cluster_desc:
        st.info(cluster_desc[cluster])

    st.markdown("---")

    st.caption("Model menggunakan metode K-Means berdasarkan perilaku penjualan (Total_Items, Total_Cost, Product_Count).")
