import os, json, tempfile, subprocess, shutil
import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shock_rate import predict_shock

# =========================================
# UI Header
# =========================================
st.title("SHIELD")
st.caption("HRV Sepsis Early Warning System Powerd by AI")

risk_placeholder = st.empty()
ecg_hrv_placeholder = st.empty()

qp = st.experimental_get_query_params()
token_q = qp.get("token", [""])[0]
obs_q   = qp.get("obs", [""])[0]

# =========================================
# Check Models
# =========================================
@st.cache_resource
def _check_models_exist():
    assert os.path.exists("models/model_focalloss.h5"), "Missing models/model_focalloss.h5"
    assert os.path.exists("models/xgb_model.json"), "Missing models/xgb_model.json"

_check_models_exist()

# =========================================
# FHIR Fetch
# =========================================
def fetch_observation(token, obs_url):
    r = requests.get(
        obs_url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        timeout=20
    )
    r.raise_for_status()
    return r.json()

# =========================================
# Patient Data Placeholder
# =========================================
st.markdown("---")
patient_data_placeholder = st.empty()
with patient_data_placeholder.container():
    st.expander("Patient Data (Click to Expand)", expanded=False)

# =========================================
# Token & Observation URL
# =========================================
token = st.text_input("Token", value=token_q, type="password")
obs_url = st.text_input("Observation URL", value=obs_q)

# =========================================
# Reset cache if token/obs_url changed
# =========================================
current_key = f"{token}||{obs_url}"
if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = ""
if st.session_state.analysis_key != current_key:
    for k in [
        "analysis_done", "obs", "ecg_signal", "hrv_df", "preds",
        "risk_pct", "risk_label", "risk_color", "hr_signal",
        "ecg_download_path"
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.analysis_key = current_key

# =========================================
# Auto Run Logic
# =========================================
if token and obs_url:

    if "analysis_done" not in st.session_state:
        try:
            with st.spinner("Fetching Patient Data..."):
                obs = fetch_observation(token, obs_url)

            st.session_state.obs = obs

            with tempfile.TemporaryDirectory() as td:
                obs_path = os.path.join(td, "obs.json")
                ecg_csv  = os.path.join(td, "ECG_5min.csv")
                h0_csv   = os.path.join(td, "h0.csv")

                with open(obs_path, "w") as f:
                    json.dump(obs, f)

                # ----- Parse ECG -----
                with st.spinner("Parsing ECG..."):
                    proc = subprocess.run(
                        ["python", "parse_fhir_ecg_to_csv.py", obs_path, ecg_csv],
                        capture_output=True,
                        text=True
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr or "parse_fhir_ecg_to_csv.py failed")

                    if not os.path.exists(ecg_csv):
                        raise RuntimeError("ECG CSV not created")

                    ecg_df = pd.read_csv(ecg_csv, header=None)
                    ecg_signal = (
                        pd.to_numeric(ecg_df.iloc[:, 0], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                        .ravel()
                    )
                    if ecg_signal.size == 0:
                        raise RuntimeError("ECG signal empty")

                # ===== COPY FOR DOWNLOAD (關鍵修正) =====
                download_path = os.path.join(os.getcwd(), "ECG_5min.csv")
                shutil.copy(ecg_csv, download_path)
                st.session_state.ecg_download_path = download_path
                # =======================================

                # ----- Generate HRV Features -----
                with st.spinner("Generating HRV features..."):
                    proc = subprocess.run(
                        ["python", "generate_HRV_10_features.py", ecg_csv, h0_csv],
                        capture_output=True,
                        text=True
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr or "generate_HRV_10_features.py failed")

                    h0_json = proc.stdout.splitlines()[-1]
                    hrv_df = pd.read_json(h0_json, orient="records")

                # ----- Predict Shock Risk -----
                with st.spinner("Predicting shock risk..."):
                    preds = predict_shock(h0_csv)

            # ===== Save results =====
            st.session_state.ecg_signal = ecg_signal
            st.session_state.hrv_df = hrv_df
            st.session_state.preds = preds

            risk_pct = round(float(preds[0]) * 100, 2)
            if risk_pct < 20:
                risk_label, risk_color = "LOW RISK", "#2ecc71"
            elif risk_pct < 40:
                risk_label, risk_color = "MODERATE RISK", "#f39c12"
            else:
                risk_label, risk_color = "HIGH RISK", "#e74c3c"

            st.session_state.risk_pct = risk_pct
            st.session_state.risk_label = risk_label
            st.session_state.risk_color = risk_color
            st.session_state.analysis_done = True

            st.success("Done")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # =========================================
    # Patient Data
    # =========================================
    with patient_data_placeholder.container():
        with st.expander("Patient Data (Click to Expand)", expanded=False):
            st.json(st.session_state.get("obs", {}))

    # =========================================
    # Risk Visualization
    # =========================================
    risk_pct = st.session_state.risk_pct
    risk_label = st.session_state.risk_label
    risk_color = st.session_state.risk_color

    with risk_placeholder.container():
        pie_col, value_col = st.columns([1, 2], gap="large")
        with pie_col:
            components.html(
                f"""
                <div style="width:120px;height:120px;border-radius:50%;
                background:conic-gradient({risk_color} {risk_pct}%,#2c2c2c 0);">
                </div>
                """,
                height=140,
            )
        with value_col:
            st.markdown(
                f"<h1>{risk_pct:.2f}%</h1><h3 style='color:{risk_color}'>{risk_label}</h3>",
                unsafe_allow_html=True,
            )

    # =========================================
    # Download ECG CSV (修好版)
    # =========================================
    st.markdown("---")
    st.subheader("Download Raw ECG")

    if "ecg_download_path" in st.session_state:
        with open(st.session_state.ecg_download_path, "rb") as f:
            st.download_button(
                "⬇️ Download ECG_5min.csv",
                data=f,
                file_name="ECG_5min.csv",
                mime="text/csv"
            )

    # =========================================
    # ECG Plot
    # =========================================
    with ecg_hrv_placeholder.container():
        st.markdown("---")
        st.subheader("ECG Input")

        ecg_signal = st.session_state.ecg_signal
        hr = np.asarray(ecg_signal)
        n = len(hr)

        start = st.slider("View start index", 0, max(0, n - 500), 0)
        win = hr[start:start + 500]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(win)
        ax.set_title("ECG (index-based)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("Please enter Token and Observation URL")
