import streamlit as st
import numpy as np
import plotly.express as px
import sys; sys.path.append('src')
from fft_core import compress_image_rgb, get_spectrum
from utils import load_image, to_pil, normalize_spectrum
from metrics import full_report
from PIL import Image
import io

# ── Page Config ──
st.set_page_config(
    page_title="FFT Compression Lab",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
.stApp { background: #05060a; color: #e8eaf0; }
.metric-card { background:#10121c; border:1px solid #1a1d2e;
border-radius:8px; padding:20px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.title("🌊 FFT Compression Lab")
st.sidebar.markdown("Upload an image and control compression in real-time.")
uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"])
keep = st.sidebar.slider("Data to Keep (%)", 1, 100, 10) / 100
show_spectrum = st.sidebar.checkbox("Show Frequency Spectrum", True)

# ── Main App ──
st.title("Fourier Transform Image Compression")

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_arr = np.array(img_pil, dtype=np.float64)

    # Run compression
    with st.spinner("Applying FFT..."):
        compressed = compress_image_rgb(img_arr, keep_fraction=keep)
        report = full_report(img_arr, compressed, keep)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSNR", f"{report['PSNR (dB)']} dB")
    c2.metric("MSE", report["MSE"])
    c3.metric("Data Removed", f"{report['Compressed (%)']}%")
    c4.metric("Data Kept", f"{report['Data Kept (%)']}%")

    # Image comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader(f"Compressed ({keep*100:.0f}% data)")
        st.image(to_pil(compressed), use_container_width=True)

    # Frequency spectrum
    if show_spectrum:
        st.subheader("🔬 Frequency Spectrum")
        gray = img_arr.mean(axis=2)
        spec = get_spectrum(gray)
        fig = px.imshow(spec, color_continuous_scale="viridis",
            title="FFT Magnitude Spectrum (log scale)")
        st.plotly_chart(fig, use_container_width=True)

    # Download compressed image
    buf = io.BytesIO()
    to_pil(compressed).save(buf, format="PNG")
    st.download_button("⬇️ Download Compressed Image",
        buf.getvalue(), "compressed.png", "image/png")
else:
    st.info("👈 Upload an image from the sidebar to begin")
    st.subheader("📊 Quality vs Compression Curve")
    st.stop()
fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
psnr_vals = []
for f in fractions:
    comp = compress_image_rgb(img_arr, keep_fraction=f)
    from metrics import psnr as calc_psnr
    psnr_vals.append(round(calc_psnr(img_arr, comp), 2))

import plotly.graph_objects as go
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=[f*100 for f in fractions],
    y=psnr_vals,
    mode='lines+markers',
    line=dict(color='#00f5c4', width=3),
    marker=dict(size=8)
))
fig2.update_layout(
    xaxis_title="Data Kept (%)",
    yaxis_title="PSNR (dB)",
    paper_bgcolor="#05060a",
    plot_bgcolor="#0c0e14",
    font_color="#e8eaf0"
)
st.plotly_chart(fig2, use_container_width=True)