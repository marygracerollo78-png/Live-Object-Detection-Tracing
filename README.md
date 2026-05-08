# 🎥 Live Object Detection & Tracing

A real-time AI-powered object detection system using **YOLOv8**, **Streamlit**, and **WebRTC**.  
The app detects, tracks, and labels objects from a live camera feed with visual bounding boxes and alert system.

---

## 🚀 Features

- 🎯 Real-time object detection using YOLOv8
- 📹 Live webcam streaming (WebRTC)
- 📦 Object tracking across frames
- 🚨 Auto alert when target object is detected
- 💾 Save detected frames as images
- 🎨 Simple light violet UI design
- 📊 Live object counting display

---

## 🧠 Technologies Used

- Python
- Streamlit
- Streamlit WebRTC
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- NumPy

---

## 📂 Project Structure


project/
│
├── app.py
├── requirements.txt
├── frames/ # saved detected images
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository

git clone [https://github.com/your-username/live-object-detection.git](https://github.com/marygracerollo78-png/Live-Object-Detection-Tracing.git)

cd live-object-detection-tracing


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the app

streamlit run app.py


---

## 📦 Requirements


streamlit
streamlit-webrtc
ultralytics
opencv-python-headless
av
numpy
pillow
torch
torchvision


---

## 🎯 How It Works

1. The webcam feed is captured using WebRTC  
2. YOLOv8 model detects objects in real time  
3. Bounding boxes are drawn around detected objects  
4. If the selected target object appears, a red alert is triggered  
5. Frames can optionally be saved locally  

---

## 🚨 Alert System

- Automatically highlights target objects in **red**
- Displays alert text when detection is active
- Works in real-time without manual toggle

---

## 📸 Output Example

- Person detection
- Objects labeled with confidence
- Real-time tracking boxes
- Live object count display

---
