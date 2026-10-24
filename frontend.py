import io
import requests
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="On-Device Garbage Classifier & Drift Monitor",
    layout="wide"
)

st.title("🗑️ Smart Waste Classifier — 'It's Getting Worse' Drift Monitor")
st.caption("100% Local ResNet-50 Inference • Two-Stage Hybrid Baseline & Entropy Alerting")

# --- Sidebar: Thresholds & Hackathon Drift Simulator ---
st.sidebar.header("⚙️ Alert Thresholds")
entropy_threshold = st.sidebar.slider(
    "Max Prediction Entropy (nats)", min_value=0.20, max_value=2.40, value=1.10, step=0.05,
    help="Max theoretical entropy for 12 classes is ln(12) ≈ 2.48. Higher = flatter probability distribution."
)
distance_threshold = st.sidebar.slider(
    "Max Embedding L2 Distance", min_value=0.10, max_value=1.40, value=0.55, step=0.05,
    help="L2 distance between incoming image features and the Hybrid Baseline signature."
)
rolling_alert_rate = st.sidebar.slider(
    "Rolling Window Alert Trigger (%)", min_value=10, max_value=100, value=40, step=10,
    help="Fires the global system degradation banner if this % of the last 5 uploads breach thresholds."
)

st.sidebar.divider()
st.sidebar.header("🧪 Simulate Real-World Drift")
st.sidebar.caption("Deliberately induce environmental drift before sending to the model:")
blur_radius = st.sidebar.slider("Lens Blur (Gaussian Radius)", 0.0, 10.0, 0.0, 0.5)
brightness_factor = st.sidebar.slider("Lighting Condition (1.0 = Normal)", 0.1, 2.5, 1.0, 0.1)

if st.sidebar.button("🗑️ Reset Database & Baseline", use_container_width=True):
    requests.post(f"{API_URL}/reset")
    st.toast("Cleared database history and dynamic baseline!", icon="🧹")
    st.rerun()

# --- Helper to apply simulated drift ---
def apply_simulated_drift(img: Image.Image) -> Image.Image:
    out = img.convert("RGB")
    if blur_radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(out)
        out = enhancer.enhance(brightness_factor)
    return out

# --- Drag-and-Drop Image Upload Section ---
st.subheader("📤 Drop Waste Images for Classification")
uploaded_files = st.file_uploader(
    "Drag and drop one or more images (plastic, metal, paper, glass, battery, etc.)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button(f"🔍 Run On-Device Inference ({len(uploaded_files)} image(s))", type="primary"):
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, up_file in enumerate(uploaded_files):
            orig_img = Image.open(up_file)
            processed_img = apply_simulated_drift(orig_img)

            buf = io.BytesIO()
            processed_img.save(buf, format="JPEG")
            buf.seek(0)

            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (up_file.name, buf, "image/jpeg")},
                data={
                    "entropy_threshold": entropy_threshold,
                    "distance_threshold": distance_threshold
                }
            )

            if resp.status_code == 200:
                res = resp.json()
                with cols[idx % 4]:
                    st.image(processed_img, caption=f"{up_file.name}", use_container_width=True)
                    st.markdown(f"**Pred:** `{res['predicted_class']}` ({res['confidence']*100:.1f}%)")
                    st.caption(f"Entropy: `{res['entropy']}` | Drift L2: `{res['embedding_distance']}`")
                    if res["is_alert"]:
                        st.error(f"⚠️ {res['alert_reason']}")
                        st.toast(f"Alert triggered on {up_file.name}: {res['alert_reason']}", icon="🚨")
                    else:
                        st.success("✅ Reliable Prediction")

st.divider()

# --- Fetch History & Render Dashboard ---
try:
    hist_resp = requests.get(f"{API_URL}/history?limit=50", timeout=3)
    history = hist_resp.json() if hist_resp.status_code == 200 else []
except Exception:
    st.warning("FastAPI backend is not reachable at http://127.0.0.1:8000. Start `uvicorn backend:app --reload` first.")
    st.stop()

if not history:
    st.info("No predictions logged yet. Drop a few clean images above to calibrate the hybrid baseline!")
    st.stop()

df = pd.DataFrame(history).sort_values("id")

# --- Global Timed Alert Notification Banner ---
recent_window = df.tail(5)
recent_breach_pct = (recent_window["is_alert"].sum() / len(recent_window)) * 100
latest_row = df.iloc[-1]

if recent_breach_pct >= rolling_alert_rate or bool(latest_row["is_alert"]):
    st.error(
        f"🚨 **TIMED SYSTEM ALERT [{latest_row['timestamp']}]: MODEL RELIABILITY DROPPING!**  \n"
        f"**Latest Cause:** {latest_row['alert_reason']}  \n"
        f"**Rolling 5-Sample Breach Rate:** `{recent_breach_pct:.0f}%` (Threshold: `{rolling_alert_rate}%`). "
        f"Please check camera lighting, clean the lens, or recalibrate the local baseline."
    )
else:
    st.success(f"🟢 **System Healthy** (Last checked: {latest_row['timestamp']}) — Rolling breach rate: `{recent_breach_pct:.0f}%`")

# --- Top-Level KPI Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Images Stored", len(df))
m2.metric("Latest Prediction Entropy", f"{latest_row['entropy']:.2f} nats", 
          delta=f"{latest_row['entropy'] - entropy_threshold:.2f} vs limit", delta_color="inverse")
m3.metric("Latest Embedding L2 Drift", f"{latest_row['embedding_distance']:.2f}", 
          delta=f"{latest_row['embedding_distance'] - distance_threshold:.2f} vs limit", delta_color="inverse")
m4.metric("Total Alerts Fired", int(df["is_alert"].sum()))

# --- Telemetry Charts ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("📉 Prediction Entropy Over Time")
    chart_df_ent = df[["id", "entropy"]].set_index("id")
    chart_df_ent["Threshold"] = entropy_threshold
    st.line_chart(chart_df_ent)

with c2:
    st.subheader("🧭 Input Distribution Shift (L2 Distance)")
    chart_df_dist = df[["id", "embedding_distance"]].set_index("id")
    chart_df_dist["Threshold"] = distance_threshold
    st.line_chart(chart_df_dist)

# --- Stored Image Database Gallery & Audit Log ---
st.subheader("🗄️ Database Image Gallery & Telemetry Log")
gallery_cols = st.columns(6)
for i, row in enumerate(df.sort_values("id", ascending=False).head(12).to_dict("records")):
    with gallery_cols[i % 6]:
        img_url = f"{API_URL}/images/{row['image_path']}"
        st.image(img_url, use_container_width=True)
        badge = "🚨" if row["is_alert"] else "✅"
        st.markdown(f"{badge} **{row['predicted_class']}** (`{row['confidence']*100:.0f}%`)")
        st.caption(f"ID #{row['id']} • H:{row['entropy']:.2f} • D:{row['embedding_distance']:.2f}")

with st.expander("📋 View Full SQLite Database Table"):
    st.dataframe(
        df[["id", "timestamp", "filename", "predicted_class", "confidence", "entropy", "embedding_distance", "is_alert", "alert_reason"]]
        .sort_values("id", ascending=False),
        use_container_width=True
    )
