import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import datetime
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------------------
# Load trained model & actions
# ---------------------------
model = load_model("sign_lang_model.h5")
actions = np.array(["hello", "thanks", "iloveyou"])  # <-- Change with your classes

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Store last 30 frames
sequence = deque(maxlen=30)

# ---------------------------
# Keypoint Extraction Function
# ---------------------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ---------------------------
# Prediction Function
# ---------------------------
def predict_keypoints(frame, holistic):
    """Run Mediapipe, extract keypoints, make prediction if enough frames"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True

    # Extract keypoints
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    # Only predict when we have 30 frames
    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)  # (1,30,1662)
        preds = model.predict(input_data, verbose=0)[0]
        label = actions[np.argmax(preds)]
        confidence = np.max(preds)
        return f"{label} ({confidence*100:.2f}%)"
    return "Collecting frames..."

# ---------------------------
# GUI Class
# ---------------------------
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        self.root.geometry("800x600")

        # Title
        self.label = tk.Label(root, text="Sign Language Detection", font=("Arial", 20, "bold"))
        self.label.pack(pady=10)

        # Canvas for video
        self.canvas = tk.Label(root)
        self.canvas.pack()

        # Result label
        self.result_label = tk.Label(root, text="Prediction: None", font=("Arial", 16))
        self.result_label.pack(pady=10)

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=20)

        self.start_btn = tk.Button(self.btn_frame, text="Start Video", command=self.start_video, font=("Arial", 12))
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop Video", command=self.stop_video, font=("Arial", 12))
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.upload_btn = tk.Button(self.btn_frame, text="Upload Image", command=self.upload_image, font=("Arial", 12))
        self.upload_btn.grid(row=0, column=2, padx=10)

        # Video state
        self.cap = None
        self.video_running = False

    def check_time_allowed(self):
        now = datetime.datetime.now().time()
        start = datetime.time(18, 0, 0)  # 6 PM
        end = datetime.time(22, 0, 0)    # 10 PM
        return start <= now <= end

    def start_video(self):
        if not self.check_time_allowed():
            self.result_label.config(text="⏰ Allowed only between 6 PM - 10 PM!")
            return
        self.cap = cv2.VideoCapture(0)
        self.video_running = True
        self.show_video()

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.canvas.config(image="")

    def show_video(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    result = predict_keypoints(frame, holistic)

                self.result_label.config(text=f"Prediction: {result}")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((500, 400))
                imgtk = ImageTk.PhotoImage(img)
                self.canvas.imgtk = imgtk
                self.canvas.configure(image=imgtk)

            self.root.after(20, self.show_video)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                keypoints = extract_keypoints(results)

                # Fake a sequence (repeat same frame 30 times)
                input_data = np.expand_dims([keypoints]*30, axis=0)
                preds = model.predict(input_data, verbose=0)[0]
                label = actions[np.argmax(preds)]
                confidence = np.max(preds)

                self.result_label.config(text=f"Prediction: {label} ({confidence*100:.2f}%)")

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((500, 400))
            imgtk = ImageTk.PhotoImage(img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)


# ---------------------------
# Run GUI
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
