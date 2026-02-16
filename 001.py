from ultralytics import YOLO
import cv2
import os
import numpy as np

# ================= CONFIG =================
MODEL_PATH = r"D:\project\helmet detection\helmet_001.pt"
CONF = 0.4

model = YOLO(MODEL_PATH)

# ================= MENU =================
print("\n🪖 Helmet Detection System")
print("1. Image")
print("2. Video")
print("3. Folder")
print("4. Live Camera")

choice = input("\nSelect option (1/2/3/4): ").strip()

# ================= IMAGE =================
if choice == "1":
    img_path = input("Enter image path: ").strip()
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Invalid image path")
        exit()

    results = model(img, conf=CONF)
    annotated = results[0].plot()

    cv2.imshow("Helmet Detection - Image", annotated)
    print("Press CTRL+Q or any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= VIDEO =================
elif choice == "2":
    video_path = input("Enter video path: ").strip()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        exit()

    print("Press CTRL+Q to stop video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF)
        annotated = results[0].plot()
        cv2.imshow("Helmet Detection - Video", annotated)

        if cv2.waitKey(1) & 0xFF == 17:  # CTRL + Q
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= FOLDER =================
elif choice == "3":
    folder_path = input("Enter folder path: ").strip()

    if not os.path.exists(folder_path):
        print("❌ Folder not found")
        exit()

    print("Press CTRL+Q to exit folder view")

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model(img, conf=CONF)
        annotated = results[0].plot()
        cv2.imshow("Helmet Detection - Folder", annotated)

        if cv2.waitKey(0) & 0xFF == 17:  # CTRL + Q
            break

    cv2.destroyAllWindows()

# ================= LIVE CAMERA =================
elif choice == "4":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        exit()

    print("Live Camera started | Press CTRL+Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF)
        annotated = results[0].plot()
        cv2.imshow("Helmet Detection - Live Camera", annotated)

        if cv2.waitKey(1) & 0xFF == 17:  # CTRL + Q
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("❌ Invalid option selected")
