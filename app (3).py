import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Cicilan & Kelayakan Mobil", page_icon="🚗", layout="wide")

# Load model dan katalog mobil
@st.cache_resource
def load_assets():
    model_reg = joblib.load("model_cicilan.pkl")
    model_clf = joblib.load("model_kelayakan.pkl")
    katalog = pd.read_csv("katalog_mobil.csv")
    return model_reg, model_clf, katalog

model_reg, model_clf, katalog = load_assets()

st.title("🚗 Simulasi Cicilan & Prediksi Kelayakan Membeli Mobil")
st.write("Hitung perkiraan cicilan bulanan dan cek apakah pengajuan kredit mobil sesuai dengan batas aman rasio cicilan (maksimal 35% dari gaji bulanan).")

# --- SIDEBAR INPUT ---
st.sidebar.header("1. Data Penghasilan")
monthly_income = st.sidebar.number_input(
    "Gaji Bulanan (Rp)",
    min_value=1_000_000,
    max_value=500_000_000,
    value=15_000_000,
    step=500_000,
    format="%d"
)

st.sidebar.header("2. Pilihan Mobil & Skema Kredit")
daftar_merek = sorted(katalog["company"].unique())
selected_company = st.sidebar.selectbox("Merek Mobil", daftar_merek)

katalog_merek = katalog[katalog["company"] == selected_company]
selected_model = st.sidebar.selectbox("Model Mobil", katalog_merek["model"].tolist())

detail_mobil = katalog_merek[katalog_merek["model"] == selected_model].iloc[0]
price_idr = int(detail_mobil["price_idr"])
body_style = detail_mobil["body_style"]

dp_percent = st.sidebar.slider("Persentase Uang Muka / DP (%)", min_value=20, max_value=40, value=25, step=5)
loan_tenure = st.sidebar.selectbox("Tenor Cicilan (Bulan)", [12, 24, 36, 48, 60], index=2)

down_payment = int(price_idr * (dp_percent / 100))

# --- RINGKASAN PILIHAN ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mobil Dipilih", f"{selected_company} {selected_model}")
col2.metric("Tipe Bodi", body_style)
col3.metric("Harga OTR", f"Rp {price_idr:,.0f}")
col4.metric(f"Uang Muka ({dp_percent}%)", f"Rp {down_payment:,.0f}")

st.divider()

if st.button("Hitung Simulasi & Cek Kelayakan", type="primary"):
    # Siapkan input sesuai fitur training
    X_reg_input = pd.DataFrame([{
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    X_clf_input = pd.DataFrame([{
        "monthly_income": monthly_income,
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    pred_cicilan = float(model_reg.predict(X_reg_input)[0])
    pred_status = int(model_clf.predict(X_clf_input)[0])
    dti_ratio = (pred_cicilan / monthly_income) * 100
    batas_aman = monthly_income * 0.35

    res1, res2, res3 = st.columns(3)
    res1.metric("Estimasi Cicilan / Bulan", f"Rp {pred_cicilan:,.0f}")
    res2.metric("Rasio Cicilan terhadap Gaji (DTI)", f"{dti_ratio:.1f}%", delta=f"Batas Aman: 35%", delta_color="off")
    res3.metric("Maksimal Cicilan Ideal (35% Gaji)", f"Rp {batas_aman:,.0f}")

    if pred_status == 1 and dti_ratio <= 35.0:
        st.success(f"✅ **LAYAK / MOBIL DIDAPAT** — Cicilan sebesar **Rp {pred_cicilan:,.0f}/bulan** ({dti_ratio:.1f}% dari gaji) masih berada di bawah batas aman 35%.")
    else:
        st.error(f"❌ **TIDAK LAYAK / BERISIKO** — Cicilan sebesar **Rp {pred_cicilan:,.0f}/bulan** menyerap **{dti_ratio:.1f}%** gaji bulananmu (melebihi batas aman 35%).")

        # Cari rekomendasi mobil alternatif dari katalog dengan skema DP & tenor yang sama
        katalog_simulasi = katalog.copy()
        katalog_simulasi["dp_sim"] = (katalog_simulasi["price_idr"] * (dp_percent / 100)).astype(int)
        X_alt = pd.DataFrame({
            "price_idr": katalog_simulasi["price_idr"],
            "down_payment": katalog_simulasi["dp_sim"],
            "loan_tenure": loan_tenure
        })
        katalog_simulasi["estimasi_cicilan"] = model_reg.predict(X_alt)
        rekomendasi = katalog_simulasi[katalog_simulasi["estimasi_cicilan"] <= batas_aman].sort_values("price_idr", ascending=False)

        st.subheader("💡 Rekomendasi Mobil Alternatif yang Sesuai Gaji Kamu")
        if not rekomendasi.empty:
            tampil = rekomendasi[["company", "model", "body_style", "price_idr", "estimasi_cicilan"]].head(5).copy()
            tampil.columns = ["Merek", "Model", "Tipe Bodi", "Harga (Rp)", "Estimasi Cicilan/Bln (Rp)"]
            tampil["Harga (Rp)"] = tampil["Harga (Rp)"].map(lambda x: f"Rp {x:,.0f}")
            tampil["Estimasi Cicilan/Bln (Rp)"] = tampil["Estimasi Cicilan/Bln (Rp)"].map(lambda x: f"Rp {x:,.0f}")
            st.dataframe(tampil, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada unit mobil di katalog yang cicilannya <= 35% gaji dengan skema DP dan tenor ini. Coba naikkan persentase DP atau pilih tenor 60 bulan.")
