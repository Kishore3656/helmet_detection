import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

# Load the trained YOLOv8 model for helmet detection
model_path = r"D:\project\helmet detection\helmet_001.pt"
model = YOLO(model_path)

# Function to detect no helmet in an image
def detect_no_helmet_in_image(image_path):
    # Update status label
    status_label.config(text="Processing...", fg="blue")
    root.update()  # Refresh the GUI

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        status_label.config(text="❌ Error: Could not read image", fg="red")
        return

    # Perform YOLO inference
    results = model(image)

    # Process detection results
    for result in results:
        if result.boxes is not None:  # Ensure there are detections
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert coordinates to integers
                confidence = float(box.conf[0])  # Get confidence score
                class_id = int(box.cls[0])  # Get class ID

                # Filter by confidence score (e.g., only consider detections with confidence > 0.5)
                if confidence > 0.5:
                    # Assuming class_id 0 is for no-helmet and 1 is for helmet
                    if class_id == 0:  # No helmet class
                        label = f'No Helmet {confidence:.2f}'
                        color = (0, 0, 255)  # Red color for no helmet
                    else:  # Helmet class
                        label = f'Helmet {confidence:.2f}'
                        color = (0, 255, 0)  # Green color for helmet

                    # Draw bounding box and label on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Resize the output image to make it bigger
    resized_image = cv2.resize(image, (1280, 720))  # Resize to 1280x720 (adjust as needed)

    # Display the output image with detections
    cv2.imshow("No Helmet Detection - Image", resized_image)
    cv2.waitKey(0)  # Wait until the user closes the window
    cv2.destroyAllWindows()  # Close the OpenCV window

    # Update status label
    status_label.config(text="Processing complete!", fg="green")

# Function to detect no helmet in a video or live feed
def detect_no_helmet_in_video(video_source):
    # Update status label
    status_label.config(text="Processing...", fg="blue")
    root.update()  # Refresh the GUI

    # Open the video source (file or webcam)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        status_label.config(text="❌ Error: Could not open video source", fg="red")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO inference
        results = model(frame)

        # Process detection results
        for result in results:
            if result.boxes is not None:  # Ensure there are detections
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert coordinates to integers
                    confidence = float(box.conf[0])  # Get confidence score
                    class_id = int(box.cls[0])  # Get class ID

                    # Filter by confidence score (e.g., only consider detections with confidence > 0.5)
                    if confidence > 0.5:
                        # Assuming class_id 0 is for no-helmet and 1 is for helmet
                        if class_id == 0:  # No helmet class
                            label = f'No Helmet {confidence:.2f}'
                            color = (0, 0, 255)  # Red color for no helmet
                        else:  # Helmet class
                            label = f'Helmet {confidence:.2f}'
                            color = (0, 255, 0)  # Green color for helmet

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Resize the output frame to make it bigger
        resized_frame = cv2.resize(frame, (1280, 720))  # Resize to 1280x720 (adjust as needed)

        # Display the frame with detections
        cv2.imshow("No Helmet Detection - Video", resized_frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Update status label
    status_label.config(text="Processing complete!", fg="green")

# Function to handle drag and drop
def handle_drop(event):
    file_path = event.data.strip('{}')  # Remove curly braces from the path
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image
        path_label.config(text=f"Image: {file_path}")  # Display the image path
        detect_no_helmet_in_image(file_path)  # Process the image
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Check if it's a video
        path_label.config(text=f"Video: {file_path}")  # Display the video path
        detect_no_helmet_in_video(file_path)  # Process the video
    else:
        status_label.config(text="❌ Error: Unsupported file type", fg="red")

# Function to start live feed processing
def start_live_feed():
    detect_no_helmet_in_video(0)  # 0 for webcam

# Function to quit the application
def quit_app():
    root.destroy()  # Close the GUI window

# Create a GUI window
root = TkinterDnD.Tk()
root.title("Helmet Detection - Drag and Drop")  # Custom window title
root.geometry("500x250")  # Set window size (width x height)

# Add a label for instructions
instruction_label = tk.Label(root, text="Drag and drop an image/video or click 'Live Feed'", font=("Arial", 14), fg="black")
instruction_label.pack(pady=10)

# Add a label to display the dropped file path
path_label = tk.Label(root, text="", font=("Arial", 12), fg="gray")
path_label.pack(pady=5)

# Add a status label
status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.pack(pady=10)

# Add a "Live Feed" button
live_feed_button = tk.Button(root, text="Live Feed", font=("Arial", 12), command=start_live_feed, bg="blue", fg="white")
live_feed_button.pack(pady=10)

# Add a "Quit" button
quit_button = tk.Button(root, text="Quit", font=("Arial", 12), command=quit_app, bg="red", fg="white")
quit_button.pack(pady=10)

# Enable drag and drop
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', handle_drop)

# Run the GUI
root.mainloop()