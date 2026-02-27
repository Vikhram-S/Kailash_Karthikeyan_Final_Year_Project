import io
import os
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import FaceDetector
from ui_theme import inject_theme, metric_card


def load_image_file(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.read()
    pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def maybe_resize(image_bgr: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Downscale very large images to keep inference fast and avoid timeouts."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image_bgr

    scale = max_size / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def name_from_filename(filename: str) -> str:
    """Extract a display name from an uploaded filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Replace common separators and title-case
    base = base.replace("_", " ").replace("-", " ").strip()
    return base.title() if base else "Unknown"


def run_image_mode(detector: FaceDetector):
    tabs = st.tabs(["Workspace", "Insights"])
    with tabs[0]:
        st.markdown(
            """
            <div style="margin-bottom: 0.75rem; font-size: 0.9rem; color: #9CA3AF;">
                <strong>Tip</strong>: On mobile, use landscape mode for a wider preview.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload images for face detection",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info(
                "Upload one or more images to get started. "
                "On mobile, tap here to open your gallery."
            )
            return

        total_faces = 0
        total_inference = 0.0
        results = []

        for f in uploaded_files:
            img_bgr = load_image_file(f)
            if img_bgr is None:
                continue

            img_bgr = maybe_resize(img_bgr)
            person_name = name_from_filename(f.name)

            t0 = time.time()
            boxes = detector.detect_faces(img_bgr)
            # Compatible with older deployments that don't accept `label` arg
            try:
                annotated = detector.draw_detections(
                    img_bgr,
                    boxes,
                    person_name,  # positional to avoid keyword issues
                )
            except TypeError:
                # Fallback: draw without name if remote code is older
                annotated = detector.draw_detections(img_bgr, boxes)
            dt = (time.time() - t0) * 1000.0

            total_faces += len(boxes)
            total_inference += dt

            results.append(
                {
                    "name": f.name,
                    "label": person_name,
                    "faces": len(boxes),
                    "latency_ms": dt,
                    "image": annotated,
                }
            )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            metric_card("Total Faces", str(total_faces), "Across all images")
        with col2:
            avg = total_inference / max(1, len(results))
            metric_card("Avg Latency", f"{avg:.1f} ms", "Per image")
        with col3:
            metric_card("Images Processed", str(len(results)), "Batch size")

        st.markdown("<hr style='opacity:0.2;margin:1.25rem 0;'/>", unsafe_allow_html=True)
        for r in results:
            st.markdown(
                f"<h4 style='margin-bottom:0.25rem;'>{r['name']}</h4>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(
                    cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB),
                    caption=f"{r['faces']} faces detected - {r['label']}",
                    use_column_width=True,
                )
            with c2:
                st.caption(f"Latency: {r['latency_ms']:.1f} ms")

    with tabs[1]:
        st.subheader("How to use in production")
        st.markdown(
            """
            - Use this as an internal tool or demo dashboard.
            - For production APIs, wrap `FaceDetector.detect_faces` in a FastAPI or Flask endpoint.
            """
        )


def run_webcam_mode(detector: FaceDetector):
    st.info("Capture a frame from your webcam. Each capture runs detection.")
    cam_image = st.camera_input("Camera", key="webcam_capture")

    if cam_image is None:
        return

    image_bgr = load_image_file(cam_image)
    if image_bgr is None:
        st.error("Could not read camera frame.")
        return

    image_bgr = maybe_resize(image_bgr)
    t0 = time.time()
    boxes = detector.detect_faces(image_bgr)
    annotated = detector.draw_detections(image_bgr, boxes)
    dt = (time.time() - t0) * 1000.0

    col1, col2 = st.columns(2)
    with col1:
        metric_card("Faces in frame", str(len(boxes)))
    with col2:
        metric_card("Latency", f"{dt:.1f} ms")

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption=f"{len(boxes)} faces detected",
        use_column_width=True,
    )


def main():
    inject_theme()

    header = st.container()
    with header:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 0.75rem;">
                <div style="font-size: 1.8rem; font-weight: 650; letter-spacing: 0.03em;">
                    AI powered smart door bell Security System using Tiny ML
                </div>
                <div style="font-size: 0.9rem; margin-top: 0.25rem; color:#9CA3AF;">
                    Real‑time, multi‑device face detection for images & webcam.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    mode = st.segmented_control(
        "Mode",
        ["Images", "Webcam"],
        default="Images",
        label_visibility="collapsed",
    )

    st.sidebar.title("Controls")
    st.sidebar.caption(
        "Upload single or multiple images, or use live webcam. "
        "Settings update in real-time."
    )

    conf = st.sidebar.slider(
        "Minimum confidence (only for advanced models)",
        0.3,
        0.9,
        0.6,
        0.05,
    )
    model_sel = st.sidebar.selectbox(
        "Model detail",
        options=[0, 1],
        index=1,
        format_func=lambda x: "Short-range (selfies)" if x == 0 else "Full-range",
    )

    detector = get_detector(conf, model_sel)

    if mode == "Webcam":
        run_webcam_mode(detector)
    else:
        run_image_mode(detector)


@st.cache_resource(show_spinner=False)
def get_detector(conf: float, model_sel: int) -> FaceDetector:
    """Create and cache the face detector so it's not reloaded on every rerun."""
    return FaceDetector(min_confidence=conf, model_selection=model_sel)
if __name__ == "__main__":
    main()

