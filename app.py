import os
import sys
import io
import time
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model import PixelSiliconNet


# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PixelSilicon — Semiconductor Image Super-Resolution & Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for hackathon presentation polish
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .info-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        color: #E2E8F0;
        font-size: 0.95rem;
    }
    .metric-badge {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_pixelsilicon_model():
    """
    Loads PixelSiliconNet architecture and best stable checkpoint weights once.
    Caches the model in memory across Streamlit re-runs.
    """
    checkpoint_path = os.path.join(project_root, "models", "pixelsilicon_best_stable.pth")
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint file not found at: `{checkpoint_path}`")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup GPU
    dummy_input = torch.zeros(1, 1, 128, 128, device=device)
    with torch.inference_mode():
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return model, device, gpu_name, total_params, checkpoint_path


def main():
    # Load cached model
    model, device, gpu_name, total_params, checkpoint_path = load_cached_pixelsilicon_model()

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("🔬 PixelSilicon")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model System Summary")
    st.sidebar.markdown(f"**Architecture**: `PixelSiliconNet`")
    st.sidebar.markdown(f"**Parameters**: `0.74M` (`{total_params:,}`)")
    st.sidebar.markdown(f"**Upscaling Factor**: `2×`")
    st.sidebar.markdown(f"**Training Data**: `KLA Paired Dataset`")
    st.sidebar.markdown(f"**Device Acceleration**: `{device.type.upper()}`")
    if device.type == "cuda":
        st.sidebar.caption(f"GPU: {gpu_name}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project Info")
    st.sidebar.info("Designed for real-time high-throughput semiconductor wafer inspection restoration.")

    # --- HEADER & EXPLANATION ---
    st.markdown('<div class="main-title">PixelSilicon — Semiconductor Image Super-Resolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Based Restoration & 2× Super-Resolution for Degraded Semiconductor Inspection Images</div>', unsafe_allow_html=True)

    # Short Explanation Box
    st.markdown(
        '<div class="info-box">'
        '<strong>About PixelSilicon:</strong> PixelSilicon learns a supervised mapping from noisy '
        'low-resolution semiconductor inspection images to clean high-resolution references, '
        'recovering structural edges and details that traditional bicubic interpolation misses.'
        '</div>',
        unsafe_allow_html=True
    )

    # --- FILE UPLOADER & DEMO SELECTION ---
    st.markdown("### 📤 Upload Inspection Image")
    uploaded_file = st.file_uploader(
        "Upload a KLA NoisyLR (.npy) NumPy array file:",
        type=["npy"],
        help="Upload a single 128x128 float32 .npy file from the KLA dataset."
    )

    # Sample dropdown demo helper if user doesn't upload a custom file
    test_dir = os.path.join(project_root, "dataset", "KLA", "Test_NoisyLR", "NoisyLR")
    demo_selected_path = None
    demo_name = None

    if uploaded_file is None and os.path.exists(test_dir):
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".npy")])
        if test_files:
            st.markdown("##### *Or select a sample file from the KLA Test Set:*")
            selected_sample = st.selectbox("Select test sample:", ["-- None --"] + test_files[:20])
            if selected_sample != "-- None --":
                demo_selected_path = os.path.join(test_dir, selected_sample)
                demo_name = selected_sample

    # Determine input data source
    input_arr = None
    source_filename = None

    if uploaded_file is not None:
        source_filename = uploaded_file.name
        try:
            bytes_io = io.BytesIO(uploaded_file.getvalue())
            input_arr = np.load(bytes_io)
        except Exception as e:
            st.error(f"❌ **Corrupted File Error**: Could not load the uploaded file as a NumPy `.npy` array. Details: {e}")
            st.stop()

    elif demo_selected_path is not None:
        source_filename = demo_name
        try:
            input_arr = np.load(demo_selected_path)
        except Exception as e:
            st.error(f"❌ **File Load Error**: Could not load `{demo_name}`. Details: {e}")
            st.stop()

    # If an image array is loaded, run validation & inference
    if input_arr is not None:
        # Error handling for array dimensions & shape
        if not isinstance(input_arr, np.ndarray):
            st.error("❌ **Invalid File Error**: The loaded `.npy` file does not contain a valid NumPy ndarray.")
            st.stop()

        if input_arr.ndim != 2 or input_arr.shape != (128, 128):
            st.error(
                f"❌ **Dimension Mismatch Error**: Uploaded image has shape `{input_arr.shape}` with `{input_arr.ndim}` dimensions. "
                f"PixelSiliconNet requires an exact 2D NumPy array of shape `(128, 128)`."
            )
            st.stop()

        # Convert array to float32 PyTorch tensor [1, 1, 128, 128]
        noisy_lr_float = input_arr.astype(np.float32)
        input_tensor = torch.from_numpy(noisy_lr_float).unsqueeze(0).unsqueeze(0).to(device)

        # 7 & 10. Measure model inference time
        start_time = time.time()
        with torch.inference_mode():
            output_tensor = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
        inf_time_ms = (time.time() - start_time) * 1000.0

        # Extract 2D predictions [256, 256] clamped to [0, 1]
        pred_np = torch.clamp(output_tensor, 0.0, 1.0).squeeze().cpu().numpy()
        uint8_pred = np.clip(pred_np * 255.0, 0.0, 255.0).astype(np.uint8)

        # 8. Generate Bicubic baseline [256, 256]
        bicubic_np = cv2.resize(noisy_lr_float, (256, 256), interpolation=cv2.INTER_CUBIC)
        bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)
        uint8_bicubic = np.clip(bicubic_clamped * 255.0, 0.0, 255.0).astype(np.uint8)

        # Input NoisyLR visualization array [128, 128] clamped to [0, 1]
        noisy_clamped = np.clip(noisy_lr_float, 0.0, 1.0)
        uint8_noisy = np.clip(noisy_clamped * 255.0, 0.0, 255.0).astype(np.uint8)

        st.markdown("---")

        # 10. Display Metrics Summary Badges
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-badge"><div class="metric-val">128 × 128</div><div class="metric-lbl">Input Resolution</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-badge"><div class="metric-val">256 × 256</div><div class="metric-lbl">Output Resolution</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-badge"><div class="metric-val">739,777</div><div class="metric-lbl">Model Parameters</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-badge"><div class="metric-val">{inf_time_ms:.2f} ms</div><div class="metric-lbl">Inference Latency</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 9 & 11. Display Three Grayscale Images Side-by-Side
        st.markdown(f"### 📊 Visual Comparison for `{source_filename}`")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. Original NoisyLR")
            st.image(uint8_noisy, caption="Input: 128×128 (Raw Sensor Noise)", use_container_width=True)

        with col2:
            st.subheader("2. Bicubic Baseline")
            st.image(uint8_bicubic, caption="Baseline: 256×256 (Interpolated)", use_container_width=True)

        with col3:
            st.subheader("3. PixelSilicon Restoration")
            st.image(uint8_pred, caption="Restoration: 256×256 (PixelSiliconNet)", use_container_width=True)

        st.markdown("---")

        # 17. Download Restored PNG Button
        buf = io.BytesIO()
        pil_img = Image.fromarray(uint8_pred)
        pil_img.save(buf, format="PNG")

        download_filename = f"pixelsilicon_restored_{source_filename.replace('.npy', '')}.png"

        st.download_button(
            label="⬇️ Download Restored 256×256 Image (PNG)",
            data=buf.getvalue(),
            file_name=download_filename,
            mime="image/png"
        )

    else:
        st.info("👆 Please upload a `.npy` file or select a sample from the test set dropdown above to run live restoration.")


if __name__ == "__main__":
    main()
