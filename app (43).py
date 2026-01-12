import os, json, tempfile, subprocess
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
# Reset cache if token/obs_url changed (MINIMAL but IMPORTANT)
# =========================================
current_key = f"{token}||{obs_url}"
if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = ""
if st.session_state.analysis_key != current_key:
    # 清掉舊資料，避免換病人仍顯示舊結果
    for k in [
        "analysis_done", "obs", "ecg_signal", "hrv_df", "preds",
        "risk_pct", "risk_label", "risk_color", "hr_signal"
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.analysis_key = current_key

# =========================================
# Auto Run Logic
# =========================================
if token and obs_url:

    # -----------------------------------------
    # Heavy pipeline: run ONCE
    # -----------------------------------------
    if "analysis_done" not in st.session_state:
        try:
            with st.spinner("Fetching Patient Data..."):
                obs = fetch_observation(token, obs_url)

            # 先把 patient JSON 顯示出來
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
                        raise RuntimeError("ECG CSV not created by parse_fhir_ecg_to_csv.py")

                    ecg_df = pd.read_csv(ecg_csv, header=None)
                    ecg_signal = (
                        pd.to_numeric(ecg_df.iloc[:, 0], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                        .ravel()
                    )
                    if ecg_signal.size == 0:
                        raise RuntimeError("ECG signal is empty after parsing")

                # ----- Generate HRV Features -----
                with st.spinner("Generating HRV features..."):
                    proc = subprocess.run(
                        ["python", "generate_HRV_10_features.py", ecg_csv, h0_csv],
                        capture_output=True,
                        text=True
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr or "generate_HRV_10_features.py failed")

                    # 從 stdout 讀回 HRV dataframe（你原本做法）
                    h0_json = proc.stdout.splitlines()[-1]
                    hrv_df = pd.read_json(h0_json, orient="records")

                # ----- Predict Shock Risk -----
                with st.spinner("Predicting shock risk..."):
                    preds = predict_shock(h0_csv)

            # ===== 存進 session_state（關鍵）=====
            st.session_state.ecg_signal = ecg_signal
            st.session_state.hrv_df = hrv_df
            st.session_state.preds = preds

            # Risk 衍生值也只算一次（避免 slider 以後再算）
            risk_pct = round(float(preds[0]) * 100, 2)
            if risk_pct < 20:
                risk_label = "LOW RISK"
                risk_color = "#2ecc71"
            elif risk_pct < 40:
                risk_label = "MODERATE RISK"
                risk_color = "#f39c12"
            else:
                risk_label = "HIGH RISK"
                risk_color = "#e74c3c"

            st.session_state.risk_pct = risk_pct
            st.session_state.risk_label = risk_label
            st.session_state.risk_color = risk_color

            st.session_state.analysis_done = True
            st.success("Done")

        except Exception as e:
            # 讓你不會「看起來沒動作」
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # -----------------------------------------
    # Always show Patient Data (no heavy rerun)
    # -----------------------------------------
    with patient_data_placeholder.container():
        with st.expander("Patient Data (Click to Expand)", expanded=False):
            st.json(st.session_state.get("obs", {}))

    # -----------------------------------------
    # Risk Visualization (values are fixed)
    # -----------------------------------------
    risk_pct = st.session_state.risk_pct
    risk_label = st.session_state.risk_label
    risk_color = st.session_state.risk_color

    with risk_placeholder.container():
        pie_col, value_col = st.columns([1, 2], gap="large")

        with pie_col:
            components.html(
                f"""
                <style>
                .pie {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    background: conic-gradient(
                        {risk_color} {risk_pct}%,
                        #2c2c2c {risk_pct}% 100%
                    );
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .pie-inner {{
                    width: 70px;
                    height: 70px;
                    background: var(--secondary-background-color);
                    border-radius: 50%;
                }}
                </style>
                <div style="display:flex; justify-content:center;">
                    <div class="pie">
                        <div class="pie-inner"></div>
                    </div>
                </div>
                """,
                height=140,
            )

        with value_col:
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:18px;">
                    <div style="font-size:42px; font-weight:800;">
                        {risk_pct:.2f}%
                    </div>
                    <div style="font-size:20px; font-weight:700; color:{risk_color};">
                        {risk_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================
    # ECG Input & HRV Features
    # =========================================
    with ecg_hrv_placeholder.container():
        st.markdown("---")
        st.subheader("ECG Input & HRV Features")

        # ----- HR Plot (slider only affects display) -----
        try:
            ecg_signal = st.session_state.ecg_signal

            # 資料只準備一次（避免每次 slider 都 np.asarray）
            if "hr_signal" not in st.session_state:
                st.session_state.hr_signal = np.asarray(ecg_signal, dtype=float).ravel()

            hr = st.session_state.hr_signal
            n = len(hr)
            x = np.arange(n)

            # idx = 750
            # if 0 <= idx < n:
            #     st.write(f"HR at index {idx}: {hr[idx]:.2f} bpm")

            start_idx = st.slider(
                "View start index",
                min_value=0,
                max_value=max(0, n - 500),
                value=min(750, max(0, n - 50)),
                step=1
            )

            window_size = 500
            end_idx = min(n, start_idx + window_size)

            hr_win = hr[start_idx:end_idx]
            x_win = x[start_idx:end_idx]

            ymin, ymax = float(hr_win.min()), float(hr_win.max())
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            pad = 0.05 * (ymax - ymin)

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(x_win, hr_win, linewidth=1)
            ax.set_title("Heart Rate (index-based view)")
            ax.set_xlabel("Index(125HZ/S)")
            ax.set_ylabel("Voltage(mV)")
            ax.set_xlim(start_idx, end_idx)
            ax.set_ylim(ymin - pad, ymax + pad)

            # if start_idx <= idx <= end_idx:
            #     ax.axvline(x=idx, linestyle="--", alpha=0.5)

            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.warning(f"Failed to plot HR: {e}")

        # ----- HRV Features (2 rows × 5 metrics) -----
        try:
            hrv_df = st.session_state.hrv_df

            st.markdown("Generated HRV Features")

            row = hrv_df.iloc[0]
            feature_names = list(row.index)[:10]
            feature_values = row.values[:10]

            cols1 = st.columns(5)
            for i in range(5):
                with cols1[i]:
                    st.metric(feature_names[i], f"{feature_values[i]:.3f}")

            cols2 = st.columns(5)
            for i in range(5, 10):
                with cols2[i - 5]:
                    st.metric(feature_names[i], f"{feature_values[i]:.3f}")
            st.markdown(
                "🔗 Reference of Features: "
                "[https://doi.org/10.1016/j.bspc.2024.106854]"
                "(https://doi.org/10.1016/j.bspc.2024.106854)"
            )

        except Exception as e:
            st.warning(f"Failed to render HRV features: {e}")

else:
    st.info("Please enter Token and Observation URL to start calculation")
