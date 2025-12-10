import cv2
import mediapipe as mp
import time
import math
import sys
import os
import tkinter as tk
from collections import deque, Counter
from PIL import Image, ImageTk

# Matplotlib for the UI graphs
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Audio Control Libraries (Windows)
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- 1. GLOBAL CONSTANTS ---
WIN_W = 1280
WIN_H = 768  # Increased slightly to fit the 3rd graph comfortably

# --- 2. MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 3. AUDIO CONTROL SETUP ---
volume_interface = None
try:
    enum = AudioUtilities.GetDeviceEnumerator()
    device = enum.GetDefaultAudioEndpoint(1, 1) # 1 = Capture/Mic
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
    print("Audio Driver: Connected to MICROPHONE successfully.")
except Exception as e:
    print(f"Audio Driver Error: {e}")
    volume_interface = None

# --- 4. HELPER FUNCTIONS ---
def get_mic_volume_percent():
    if volume_interface:
        try:
            return int(round(volume_interface.GetMasterVolumeLevelScalar() * 100))
        except: pass
    return 0

def set_mic_volume_percent(target_pct):
    if volume_interface:
        try:
            val = max(0.0, min(1.0, target_pct / 100.0))
            volume_interface.SetMasterVolumeLevelScalar(val, None)
        except: pass

def get_mic_is_muted():
    if volume_interface:
        try: return volume_interface.GetMute() == 1
        except: pass
    return False

def fingers_to_percent(fingers):
    mapping = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
    return mapping.get(fingers, 0)

# --- 5. APP CLASS ---
class App:
    def __init__(self, stable_frames=6, cam_index=0, win_w=1280, win_h=720):
        self.stable_frames = stable_frames
        self.cam_index = cam_index
        self.win_w = win_w
        self.win_h = win_h

        # 1. Camera
        try: self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        except: self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened(): raise RuntimeError("Cannot open camera.")

        # 2. MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1
        )

        # 3. Logic & State
        self.history = deque(maxlen=self.stable_frames)
        self.current_applied = get_mic_volume_percent()
        self.last_observed = 0
        self.running = True
        
        # UI Setup
        self.root = tk.Tk()
        self.root.title("Infosys_GestureVolume: Finger -> Mic Volume")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="#050505")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Grid Layout
        self.root.rowconfigure(0, weight=0) # Header
        self.root.rowconfigure(1, weight=1) # Main
        self.root.rowconfigure(2, weight=0) # Footer
        self.root.columnconfigure(0, weight=1)

        # --- HEADER ---
        self.header_frame = tk.Frame(self.root, bg="#111", height=70)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 5))
        self.header_frame.grid_propagate(False)
        
        tk.Label(self.header_frame, text="Infosys_GestureVolume: Mic Control with Hand Gestures",
                 font=("Consolas", 18, "bold"), fg="#00ffcc", bg="#111", anchor="w", padx=20).pack(fill=tk.X, pady=(5, 0))
        tk.Label(self.header_frame, text="Project by BATCH A | RADAR ANALYSIS ENABLED",
                 font=("Consolas", 10), fg="#cccccc", bg="#111", anchor="w", padx=20).pack(fill=tk.X, pady=(0, 5))

        # --- MAIN CONTENT ---
        self.main_frame = tk.Frame(self.root, bg="#050505")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.main_frame.columnconfigure(0, weight=1) # HUD
        self.main_frame.columnconfigure(1, weight=3) # Video
        self.main_frame.rowconfigure(0, weight=1)

        # LEFT PANEL (HUD)
        self.hud_panel = tk.Frame(self.main_frame, bg="#050505")
        self.hud_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.hud_panel.rowconfigure(0, weight=2) # Arc
        self.hud_panel.rowconfigure(1, weight=2) # Radar (New)
        self.hud_panel.rowconfigure(2, weight=1) # Proximity

        # --- GRAPH 1: ARC REACTOR ---
        self.fig_arc = Figure(figsize=(3, 3), dpi=100, facecolor='#050505')
        self.ax_arc = self.fig_arc.add_subplot(111, projection='polar')
        self.ax_arc.set_facecolor('#050505')
        self.ax_arc.axis('off')
        
        # Background & Dynamic Bar
        self.ax_arc.bar([0], [2.4], width=2*math.pi, color='#151515', bottom=0.0)
        self.arc_bar = self.ax_arc.bar([0], [2.4], width=0, color='#00ffff', bottom=0.0)[0]
        self.ax_arc.set_ylim(0, 2.5) 
        self.text_vol = self.ax_arc.text(0, 0, "0%", ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        self.canvas_arc = FigureCanvasTkAgg(self.fig_arc, master=self.hud_panel)
        self.canvas_arc.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=5)

        # --- GRAPH 2: FINGER RADAR (NEW) ---
        self.fig_radar = Figure(figsize=(3, 3), dpi=100, facecolor='#050505')
        self.ax_radar = self.fig_radar.add_subplot(111, projection='polar')
        self.ax_radar.set_facecolor('#050505')
        
        # Radar Setup
        self.radar_categories = ['Thumb', 'Index', 'Mid', 'Ring', 'Pinky']
        self.radar_angles = np.linspace(0, 2 * np.pi, len(self.radar_categories), endpoint=False).tolist()
        self.radar_angles += self.radar_angles[:1] # Close the loop
        
        # Axis formatting
        self.ax_radar.set_xticks(self.radar_angles[:-1])
        self.ax_radar.set_xticklabels(self.radar_categories, color="#00ffcc", fontsize=8)
        self.ax_radar.set_yticks([0.5, 1.0])
        self.ax_radar.set_yticklabels([]) # Hide radial numbers
        self.ax_radar.spines['polar'].set_color('#333333')
        self.ax_radar.grid(color='#333333', linestyle='--', linewidth=0.5)
        self.ax_radar.set_ylim(0, 1.1)

        # Initial Empty Plot
        self.radar_values = [0, 0, 0, 0, 0]
        self.radar_values_closed = self.radar_values + [self.radar_values[0]]
        self.radar_line, = self.ax_radar.plot(self.radar_angles, self.radar_values_closed, color='#00ffcc', linewidth=2)
        self.radar_fill, = self.ax_radar.fill(self.radar_angles, self.radar_values_closed, color='#00ffcc', alpha=0.25)
        
        self.canvas_radar = FigureCanvasTkAgg(self.fig_radar, master=self.hud_panel)
        self.canvas_radar.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=5)

        # --- GRAPH 3: Z-AXIS PROXIMITY ---
        self.fig_prox = Figure(figsize=(3, 1.5), dpi=100, facecolor='#050505')
        self.ax_prox = self.fig_prox.add_subplot(111)
        self.ax_prox.set_facecolor('#0f0f0f')
        self.ax_prox.set_title("Z-AXIS SENSOR", color='#00ffcc', fontsize=8)
        self.ax_prox.set_yticks([])
        self.ax_prox.set_xticks([])
        self.ax_prox.spines['bottom'].set_color('#333')
        self.ax_prox.spines['top'].set_color('#333')
        self.ax_prox.spines['left'].set_color('#333')
        self.ax_prox.spines['right'].set_color('#333')

        self.bar_prox = self.ax_prox.barh([0], [0], height=0.5, color='#00ff00')[0]
        self.ax_prox.set_xlim(0, 1.0)
        self.ax_prox.set_ylim(-0.5, 0.5)
        self.canvas_prox = FigureCanvasTkAgg(self.fig_prox, master=self.hud_panel)
        self.canvas_prox.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=5)

        # RIGHT: VIDEO
        self.video_label = tk.Label(self.main_frame, bg="black")
        self.video_label.grid(row=0, column=1, sticky="nsew")

        # --- FOOTER ---
        self.status_label = tk.Label(self.root, text="System Ready", bg="#111", fg="#00ff99", font=("Consolas", 12), anchor="w", padx=10)
        self.status_label.grid(row=2, column=0, sticky="ew")

        self.root.bind("<Key>", self._on_keypress)
        self.root.after(10, self._update_frame)

    def _on_keypress(self, event):
        if hasattr(event, "char") and event.char and event.char.lower() == "q": self._on_close()

    def _update_frame(self):
        if not self.running: return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(50, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Data placeholders
        current_finger_states = [0, 0, 0, 0, 0] # T, I, M, R, P
        wrist_z_estimate = 0.0 

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            
            # --- INDIVIDUAL FINGER DETECTION ---
            # 1. Thumb (Index 0)
            # Check if thumb tip (4) is to the left/right of index base (2) depending on hand orientation
            # Simplifying: For right hand (flipped), thumb tip x < thumb mcp x usually means open
            try:
                if lm[4].x < lm[2].x if lm[17].x > lm[2].x else lm[4].x > lm[2].x:
                    current_finger_states[0] = 1
            except: pass

            # 2. Other Fingers (Index 1-4) - Check Tip y < Pip y
            tips_pips = [(8,6), (12,10), (16,14), (20,18)]
            for i, (tip, pip) in enumerate(tips_pips):
                if lm[tip].y < lm[pip].y:
                    current_finger_states[i+1] = 1

            # --- Z-PROXIMITY ---
            try:
                dist = math.sqrt((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2)
                wrist_z_estimate = max(0.0, min(1.0, (dist - 0.1) * 3.5))
            except: pass

        fingers_found = sum(current_finger_states)

        # Smoothing Logic
        self.history.append(fingers_found)
        chosen = fingers_found # Instant feedback for radar, smoothed for volume
        
        # Determine stable count for volume control
        if len(self.history) == self.history.maxlen:
            counts = Counter(self.history)
            most_common = counts.most_common()
            if most_common: chosen_stable = most_common[0][0]
            else: chosen_stable = fingers_found
        else: chosen_stable = fingers_found

        # Volume Control
        target_pct = fingers_to_percent(chosen_stable)
        if target_pct != self.current_applied:
            set_mic_volume_percent(target_pct)
            self.current_applied = get_mic_volume_percent()

        self.last_observed = chosen_stable
        muted = get_mic_is_muted()
        
        # --- UI UPDATES ---
        
        # 1. ARC REACTOR
        try:
            vol_radians = (self.current_applied / 100.0) * (2 * math.pi)
            self.arc_bar.set_width(vol_radians)
            col = '#00ffff' if self.current_applied < 50 else '#ff00ff' if self.current_applied < 80 else '#ff3333'
            self.arc_bar.set_color(col)
            self.text_vol.set_text(f"{int(self.current_applied)}%")
            self.canvas_arc.draw_idle()
        except: pass

        # 2. RADAR CHART (Update)
        try:
            # We use the raw 'current_finger_states' for the radar to make it feel responsive
            # Append start to end to close the polygon
            new_values = current_finger_states + [current_finger_states[0]]
            self.radar_line.set_ydata(new_values)
            # To update fill, we have to remove the old polygon and add a new one (limitation of mpl fill)
            self.radar_fill.remove()
            self.radar_fill, = self.ax_radar.fill(self.radar_angles, new_values, color='#00ffcc', alpha=0.25)
            self.canvas_radar.draw_idle()
        except: pass

        # 3. Z-PROXIMITY
        try:
            self.bar_prox.set_width(wrist_z_estimate)
            col = '#00ff00' if wrist_z_estimate < 0.5 else '#ffcc00' if wrist_z_estimate < 0.8 else '#ff0000'
            self.bar_prox.set_color(col)
            self.canvas_prox.draw_idle()
        except: pass

        # Display Video
        display_frame = self._overlay_text(frame.copy(), self.last_observed, self.current_applied, muted)
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        try:
            pil = pil.resize((self.video_label.winfo_width(), self.video_label.winfo_height()), Image.LANCZOS)
        except: pass
        imgtk = ImageTk.PhotoImage(image=pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Status Bar
        now = time.strftime("%H:%M:%S")
        status_txt = f"[{now}] | RADAR: {current_finger_states} | Mic: {self.current_applied}%"
        self.status_label.config(text=status_txt)

        self.root.after(15, self._update_frame)

    def _overlay_text(self, frame, fingers, volume, muted):
        # (Same overlay logic as before)
        h, w = frame.shape[:2]
        txt = f"Vol: {volume}%"
        cv2.putText(frame, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame

    def _on_close(self):
        self.running = False
        try: self.cap.release()
        except: pass
        try: self.hands.close()
        except: pass
        try: self.root.destroy()
        except: pass
        sys.exit(0)

if __name__ == "__main__":
    app = App()
    app.root.mainloop()
