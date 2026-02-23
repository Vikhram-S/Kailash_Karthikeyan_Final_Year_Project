import io
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


def run_image_mode(detector: FaceDetector):
    tabs = st.tabs(["Upload Image", "Insights"])
    with tabs[0]:
        uploaded_files = st.file_uploader(
            "Drop one or more images here",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Upload at least one image to begin.")
            return

        total_faces = 0
        total_inference = 0.0
        results = []

        for f in uploaded_files:
            img_bgr = load_image_file(f)
            if img_bgr is None:
                continue

            t0 = time.time()
            boxes = detector.detect_faces(img_bgr)
            annotated = detector.draw_detections(img_bgr, boxes)
            dt = (time.time() - t0) * 1000.0

            total_faces += len(boxes)
            total_inference += dt

            results.append(
                {
                    "name": f.name,
                    "faces": len(boxes),
                    "latency_ms": dt,
                    "image": annotated,
                }
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Total Faces", str(total_faces), "Across all images")
        with col2:
            avg = total_inference / max(1, len(results))
            metric_card("Avg Latency", f"{avg:.1f} ms", "Per image")
        with col3:
            metric_card("Images Processed", str(len(results)), "Batch size")

        st.markdown("---")
        for r in results:
            st.markdown(f"#### {r['name']}")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(
                    cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB),
                    caption=f"{r['faces']} faces detected",
                    use_column_width=True,
                )
            with c2:
                st.caption(f"Latency: {r['latency_ms']:.1f} ms")

    with tabs[1]:
        st.subheader("How to use in production")
        st.markdown(
            """
            - This UI is suitable as an internal tool.
            - For production APIs, wrap `FaceDetector.detect_faces` into a FastAPI endpoint.

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

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "<h2 style='margin-bottom:0.1rem;'>Pro Face Detection Studio</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='color:#9CA3AF;'>Ultra-fast, production-grade face detection for images & webcam.</span>",
            unsafe_allow_html=True,
        )
    with right:
        mode = st.toggle("Live webcam mode", value=False)

    st.sidebar.title("Controls")
    st.sidebar.caption(
        "Upload single or multiple images, or use live webcam. "
        "Settings update in real-time."
    )

    conf = st.sidebar.slider(
        "Minimum confidence",
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

    detector = FaceDetector(min_confidence=conf, model_selection=model_sel)

    if mode:
        run_webcam_mode(detector)
    else:
        run_image_mode(detector)


if __name__ == "__main__":
    main()

