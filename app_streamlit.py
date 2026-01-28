import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Prediksi Nilai TKA"
)

model=joblib.load("model.joblib")

st.title("Prediksi Nilai TKA")
st.markdown("Machine Learning untuk memprediksi nilai TKA")	

jam_belajar_per_hari = st.slider("Jam belajar", 0, 24, 1)
persen_kehadiran = st.slider("persen kehadiran", 0, 100, 50)
bimbel = st.pills("bimbel", ["ya", "tidak"], default="ya")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[jam_belajar_per_hari,persen_kehadiran, bimbel]], columns= ("jam_belajar_per_hari", "persen_kehadiran", "bimbel"))
	prediksi = model.predict (data_baru)[0]
	prediksi = prediksi.clip(0, 100)
	prediksi = float(prediksi)

	st.success(f"model memprediksi nilai tka: {prediksi:.0f}")
