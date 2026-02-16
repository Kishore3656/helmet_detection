import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

# Load the trained YOLOv8 model for helmet detection
model_path = r"D:\project\helmet detection\helmet_001.pt"
model = YOLO(model_path)

# ================= STOP FLAG =================
stop_flag = False

# Function to detect no helmet in an image
def detect_no_helmet_in_image(image_path):
    status_label.config(text="Processing...", fg="blue")
    root.update()

    image = cv2.imread(image_path)

    if image is None:
        status_label.config(text="❌ Error: Could not read image", fg="red")
        return

    results = model(image)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if confidence > 0.5:
                    if class_id == 0:
                        label = f'No Helmet {confidence:.2f}'
                        color = (0, 0, 255)
                    else:
                        label = f'Helmet {confidence:.2f}'
                        color = (0, 255, 0)

                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    resized_image = cv2.resize(image, (1280, 720))
    cv2.imshow("No Helmet Detection - Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    status_label.config(text="Processing complete!", fg="green")

# Function to detect no helmet in a video or live feed
def detect_no_helmet_in_video(video_source):
    global stop_flag
    stop_flag = False

    status_label.config(text="Processing...", fg="blue")
    root.update()

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        status_label.config(text="❌ Error: Could not open video source", fg="red")
        return

    while True:
        # 🔴 THIS LINE IS THE KEY FIX
        root.update()

        if stop_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if confidence > 0.5:
                        if class_id == 0:
                            label = f'No Helmet {confidence:.2f}'
                            color = (0, 0, 255)
                        else:
                            label = f'Helmet {confidence:.2f}'
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("No Helmet Detection - Video", resized_frame)

        # Press 'q' to exit OR Stop button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if stop_flag:
        status_label.config(text="⏹ Stopped by user", fg="orange")
    else:
        status_label.config(text="Processing complete!", fg="green")

# ================= STOP FUNCTION =================
def stop_processing():
    global stop_flag
    stop_flag = True
    status_label.config(text="Stopping...", fg="orange")

# Function to handle drag and drop
def handle_drop(event):
    file_path = event.data.strip('{}')
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        path_label.config(text=f"Image: {file_path}")
        detect_no_helmet_in_image(file_path)
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        path_label.config(text=f"Video: {file_path}")
        detect_no_helmet_in_video(file_path)
    else:
        status_label.config(text="❌ Error: Unsupported file type", fg="red")

# Function to start live feed processing
def start_live_feed():
    detect_no_helmet_in_video(0)

# Function to quit the application
def quit_app():
    global stop_flag
    stop_flag = True
    root.destroy()

# ================= GUI =================
root = TkinterDnD.Tk()
root.title("Helmet Detection - Drag and Drop")
root.geometry("500x300")

instruction_label = tk.Label(
    root,
    text="Drag and drop an image/video or click 'Live Feed'",
    font=("Arial", 14),
    fg="black"
)
instruction_label.pack(pady=10)

path_label = tk.Label(root, text="", font=("Arial", 12), fg="gray")
path_label.pack(pady=5)

status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.pack(pady=10)

# ---------- BUTTONS ----------
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

live_feed_button = tk.Button(
    btn_frame, text="Live Feed", font=("Arial", 12),
    command=start_live_feed, bg="blue", fg="white", width=12
)
live_feed_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(
    btn_frame, text="Stop", font=("Arial", 12),
    command=stop_processing, bg="orange", fg="black", width=12
)
stop_button.grid(row=0, column=1, padx=5)

quit_button = tk.Button(
    btn_frame, text="Quit", font=("Arial", 12),
    command=quit_app, bg="red", fg="white", width=12
)
quit_button.grid(row=0, column=2, padx=5)

# ---------- DRAG & DROP ----------
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', handle_drop)

# ---------- RUN ----------
root.mainloop()
