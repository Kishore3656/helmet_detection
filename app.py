from ultralytics import YOLO
import streamlit as st
import cv2
import os
import numpy as np

# ------------------ Setup ------------------
st.set_page_config(page_title="Helmet Detection System", layout="wide")
st.title("🪖 Helmet Detection – Image | Video | Live")

# Load YOLO model (use ONE correct path)
model = YOLO(r"D:\project\helmet detection\helmet_001.pt")

# ------------------ Sidebar ------------------
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Image", "Video", "Folder", "Live Camera"]
)

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ------------------ Image ------------------
if option == "Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
        results = model(img, conf=conf)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR")

# ------------------ Video ------------------
elif option == "Video":
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if video_file:
        with open("temp.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()

# ------------------ Folder ------------------
elif option == "Folder":
    folder = st.text_input("Enter folder path with images")

    if folder and os.path.exists(folder):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            results = model(img, conf=conf)
            annotated = results[0].plot()

            st.image(annotated, caption=img_name, channels="BGR")

# ------------------ Live Camera ------------------
elif option == "Live Camera":
    cam_id = st.number_input("Camera ID (0 = webcam)", value=0, step=1)

    if st.button("Start Camera"):
        cap = cv2.VideoCapture(cam_id)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()
