

🎮 Infosys GestureVolume

Hand-Gesture-Controlled Microphone Volume System (Python + MediaPipe + PyCAW + Tkinter UI)

A futuristic Iron-Man-style gesture interface that detects real-time hand landmarks using MediaPipe and converts finger gestures into live microphone volume control.
Comes with a 3-panel advanced HUD:
	•	🔵 Arc Reactor Volume Meter (Circular polar progress bar)
	•	🟢 Finger Radar Scanner (5-finger gesture visualization polygon)
	•	🔶 Z-Axis Proximity Sensor (Hand-to-camera distance indicator)
	•	🎥 Integrated Live Webcam Feed with gesture overlays
	•	🎚️ Real-time mic volume control via PyCAW
	•	🎛️ Smart smoothing using rolling frame buffers

Built for Infosys Springboard Internship 6.0 Batch A.

⸻

🚀 Features

🎯 1. Hand Gesture Recognition
	•	Uses MediaPipe Hands (21 landmarks)
	•	Detects thumb, index, middle, ring, and pinky independently
	•	Converts number of open fingers into a volume percentage:

0 fingers → 0%
1 finger  → 20%
2 fingers → 40%
3 fingers → 60%
4 fingers → 80%
5 fingers → 100%



⸻

🌀 2. Iron-Man Style Arc Reactor Volume Meter
	•	Fully animated circular graph
	•	Dynamic color transitions:
	•	Cyan (<50%)
	•	Magenta (50–80%)
	•	Red (>80%)
	•	Smooth radial expansion based on volume

⸻

🛰️ 3. Finger Radar Scanner (New)

A futuristic radar that visualizes which fingers are open.
	•	Polar chart labeled: Thumb, Index, Mid, Ring, Pinky
	•	Auto-fills radar polygon in real-time
	•	High responsiveness

⸻

📡 4. Z-Axis Proximity Sensor

Estimates hand depth using landmark distance.
	•	Green → Far
	•	Yellow → Medium
	•	Red → Very close
	•	Smooth, horizontal bar graph

⸻

🎥 5. Live Webcam Video Feed
	•	With real-time MediaPipe skeleton drawing
	•	FPS-optimized display
	•	Embedded directly inside Tkinter UI

⸻

🔊 6. Microphone Volume Control (PyCAW)
	•	Direct control of Windows Microphone endpoint
	•	Supports:
	•	Get volume %
	•	Set volume %
	•	Get mute state
	•	Robust exception handling

⸻

🧠 7. Smart Smoothing

Uses a rolling buffer of last N frames (deque(maxlen=6)) to avoid flickering volume jumps.

⸻

🖥️ 8. Modern Tkinter Dark UI
	•	3-graph HUD panel
	•	Real-time status bar
	•	Responsive window layout
	•	Custom font + color theme

⸻

🛠️ Technology Stack

Core

Component	Technology
Gesture Recognition	MediaPipe Hands
UI Framework	Tkinter
Camera Handling	OpenCV
Volume Control	PyCAW (Windows Only)
Graphs	Matplotlib
Image Processing	PIL (Pillow)
State Smoothing	collections.deque


⸻

📂 Project Structure

GestureVolume/  
│  
├── main.py                # Main UI + gesture + graph + audio system  
├── README.md              # Documentation  
├── requirements.txt       # Dependencies  
└── assets/ (optional)     # Icons / UI assets  


⸻

📦 Installation & Setup

1️⃣ Install Dependencies

pip install opencv-python mediapipe matplotlib pillow pycaw comtypes

2️⃣ Run the App

python main.py


⸻

🧭 Usage Instructions

Gesture	Mic Volume
✊ 0 fingers	0%
☝️ 1 finger	20%
✌️ 2 fingers	40%
🤟 3 fingers	60%
🖖 4 fingers	80%
🖐️ 5 fingers	100%

Press Q to quit.

⸻

🧪 How It Works (Logic Flow)

1. Read camera frame

↓

2. Process using MediaPipe Hands

↓

3. Determine each finger’s open/closed state

↓

4. Convert finger count → volume %

↓

5. Update:
	•	Arc Reactor
	•	Radar Chart
	•	Z-Axis Bar
	•	Mic Volume

↓

6. Display final UI

⸻

🖼️ UI Preview (Describe in README)

(You can upload screenshots here)

+-----------------------------------------------------------+
|  Arc Reactor | Radar Scanner | Z-Axis Sensor              |
+-----------------------------------------------------------+
|                 Live Webcam Feed                         |
+-----------------------------------------------------------+
|                Status Bar (Volume, Radar Data)            |
+-----------------------------------------------------------+


⸻

🛡 Known Limitations
	•	Works only on Windows (PyCAW requirement)
	•	Accuracy may drop with poor lighting
	•	Requires a functioning webcam

⸻

💡 Future Enhancements
	•	Add gesture → system actions (mute, zoom control, PPT navigation)
	•	Add left-hand support
	•	Voice feedback (“Volume set to 60%”)
	•	Custom theme packs for UI

⸻

🏢 Author & Credits

Developed By:
Batch A – Infosys Springboard Internship 6.0 (2025)


Hand-Tracking Powered by MediaPipe • Audio API via PyCAW

⸻

