# merged_mic_hand_tk_fixed.py
# Fixed: added missing COM-safe mic get/set helpers so GUI can seed and write mic volume.
# Run on Windows. Requires:
# pip install pycaw comtypes mediapipe opencv-python Pillow matplotlib keyboard

import sys
import time
import threading
import platform
import queue
import traceback

# -------------------------
#   ORIGINAL MC_debug04 LOGIC (UNTOUCHED)
# -------------------------

if platform.system() != "Windows":
    raise SystemExit("This script runs only on Windows.")

try:
    import keyboard   # pip install keyboard
except Exception:
    raise SystemExit("Install required package: pip install keyboard")

try:
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator
except Exception:
    raise SystemExit("Install required packages: pip install pycaw comtypes")

from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from comtypes.client import CreateObject
from comtypes import GUID

eCapture = 1
eConsole = 0
STEP_PERCENT = 2

# --- Cooldown settings ---
LAST_TRIGGER = {
    "inc": 0,
    "dec": 0
}
COOLDOWN_SECONDS = 0.15  # adjust: 0.10 = fast, 0.20 = slow, 0.30 = very slow

def allow_action(action_key):
    """Return True if cooldown time has passed for this action."""
    import time
    now = time.time()
    if now - LAST_TRIGGER[action_key] >= COOLDOWN_SECONDS:
        LAST_TRIGGER[action_key] = now
        return True
    return False


def _create_mmdevice_enumerator():
    try:
        return CreateObject("MMDeviceEnumerator.MMDeviceEnumerator", interface=IMMDeviceEnumerator)
    except Exception:
        clsid = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
        return CreateObject(clsid, interface=IMMDeviceEnumerator)

def _get_volume_interface_for_default():
    enumerator = _create_mmdevice_enumerator()
    default_device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)
    iface = default_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))

def _percent_to_scalar(p):
    return max(0.0, min(1.0, p / 100.0))

def _scalar_to_percent(s):
    return max(0.0, min(100.0, s * 100.0))

def ensure_com(func):
    def wrapper(*args, **kwargs):
        CoInitialize()
        try:
            return func(*args, **kwargs)
        except Exception:
            print("Exception in handler:", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                CoUninitialize()
            except Exception:
                pass
    wrapper.__name__ = func.__name__
    return wrapper

@ensure_com
def increase_volume():
    if not allow_action("inc"):
        return
    vol = _get_volume_interface_for_default()
    cur = float(vol.GetMasterVolumeLevelScalar())
    cur_pct = _scalar_to_percent(cur)
    new_pct = min(100.0, cur_pct + STEP_PERCENT)
    vol.SetMasterVolumeLevelScalar(_percent_to_scalar(new_pct), None)
    print(f"[+] Mic volume -> {new_pct:.0f}%")

@ensure_com
def decrease_volume():
    if not allow_action("dec"):
        return
    vol = _get_volume_interface_for_default()
    cur = float(vol.GetMasterVolumeLevelScalar())
    cur_pct = _scalar_to_percent(cur)
    new_pct = max(0.0, cur_pct - STEP_PERCENT)
    vol.SetMasterVolumeLevelScalar(_percent_to_scalar(new_pct), None)
    print(f"[-] Mic volume -> {new_pct:.0f}%")

@ensure_com
def toggle_mute():
    vol = _get_volume_interface_for_default()
    cur_mute = bool(vol.GetMute())
    vol.SetMute(0 if cur_mute else 1, None)
    print(f"[{'M' if not cur_mute else 'U'}] Mic muted -> {not cur_mute}")

def quit_program():
    print("Exiting mic hotkeys...")
    keyboard.unhook_all_hotkeys()
    sys.exit(0)

def register_hotkeys():
    keyboard.add_hotkey('ctrl+alt+up', increase_volume)
    keyboard.add_hotkey('ctrl+alt+down', decrease_volume)
    keyboard.add_hotkey('ctrl+alt+m', toggle_mute)
    keyboard.add_hotkey('ctrl+alt+q', quit_program)

# start_hotkeys / keyboard listener threads (kept as MC_debug04 style)
def start_hotkeys():
    print("Registering hotkeys...")
    register_hotkeys()
    CoInitialize()
    try:
        vol = _get_volume_interface_for_default()
        print(
            f"Initial mic volume: "
            f"{_scalar_to_percent(float(vol.GetMasterVolumeLevelScalar())):.0f}%  "
            f"muted={bool(vol.GetMute())}"
        )
    except Exception:
        pass
    finally:
        try:
            CoUninitialize()
        except Exception:
            pass

    print("Hotkeys active. Ctrl+Alt+Up/Down/M/Q")

def keyboard_listener_thread():
    # Listen to all key down/up events globally
    keyboard.hook(lambda e: None)  # keep keyboard module active for hotkeys
    keyboard.wait()  # keeps thread alive


# -------------------------
# Add the missing mic-volume helpers (COM-safe) so GUI wrappers can call them
# -------------------------

@ensure_com
def set_mic_volume_percent(vol_percent: int):
    """Set microphone (capture endpoint) master level as integer 0-100."""
    try:
        vol = _get_volume_interface_for_default()
        vol_percent = int(max(0, min(100, vol_percent)))
        vol.SetMasterVolumeLevelScalar(_percent_to_scalar(vol_percent), None)
    except Exception:
        # Keep behavior non-fatal for GUI loops
        traceback.print_exc()

@ensure_com
def get_mic_volume_percent() -> int:
    """Return current mic master level as integer 0-100. On error, return 50."""
    try:
        vol = _get_volume_interface_for_default()
        return int(_scalar_to_percent(float(vol.GetMasterVolumeLevelScalar())))
    except Exception:
        traceback.print_exc()
        return 50

@ensure_com
def get_mic_state():
    """Return (level_percent, muted) same approach as MC_debug04's get_mic_state."""
    try:
        vol = _get_volume_interface_for_default()
        level = int(_scalar_to_percent(float(vol.GetMasterVolumeLevelScalar())))
        mute = bool(vol.GetMute())
        return level, mute
    except Exception:
        return 50, False


# -------------------------
#  Tkinter + MediaPipe UI (based on provided reference)
# -------------------------

import math
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import warnings

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # kept for compatibility if needed
from comtypes import CLSCTX_ALL
import ctypes

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Configuration (from reference)
CAM_INDEX = 0
PRINT_EVERY_N_FRAMES = 15
PINCH_PIXEL_THRESHOLD = 40
PINCH_NORM_THRESHOLD = 0.03

MIN_DIST = 25
MAX_DIST = 190
VOLUME_STEP_THRESHOLD = 4

# ---------------- Mediapipe (kept)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Provide set_system_volume and get_system_volume wrappers used by the UI/process_frame code.

def set_system_volume(vol_percent):
    # route gesture writes to microphone control (MC logic preserved)
    try:
        set_mic_volume_percent(int(vol_percent))
    except Exception:
        traceback.print_exc()

def get_system_volume():
    try:
        return get_mic_volume_percent()
    except Exception:
        traceback.print_exc()
        return 50

# ---------------- Frame Processing (kept from reference, unchanged logic except using above wrappers)
def process_frame(frame, prev_pixel_dist, prev_volume):
    img_h, img_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    norm_dist = None
    pixel_dist = None
    pinch = False
    handedness_label = None
    current_volume = prev_volume
    aspect_ratio = None

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_thumb = hand_landmarks.landmark[4]
        lm_index = hand_landmarks.landmark[8]

        dx_n = lm_thumb.x - lm_index.x
        dy_n = lm_thumb.y - lm_index.y
        dz_n = lm_thumb.z - lm_index.z
        norm_dist = math.sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n)

        tx_px = int(round(lm_thumb.x * img_w))
        ty_px = int(round(lm_thumb.y * img_h))
        ix_px = int(round(lm_index.x * img_w))
        iy_px = int(round(lm_index.y * img_h))
        pixel_dist = math.hypot(tx_px - ix_px, ty_px - iy_px)

        if pixel_dist <= PINCH_PIXEL_THRESHOLD or (norm_dist is not None and norm_dist <= PINCH_NORM_THRESHOLD):
            pinch = True

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if height > 0:
            aspect_ratio = width / height

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(frame, (tx_px, ty_px), 10, (0, 0, 255), -1)
        cv2.circle(frame, (ix_px, iy_px), 10, (255, 0, 0), -1)
        cv2.line(frame, (tx_px, ty_px), (ix_px, iy_px), (0, 255, 0), 3)

        mid_x = (tx_px + ix_px) // 2
        mid_y = (ty_px + iy_px) // 2
        cv2.circle(frame, (mid_x, mid_y), 10, (0, 255, 255), -1)

        if pixel_dist is not None:
            new_volume = np.interp(pixel_dist, [MIN_DIST, MAX_DIST], [0, 100])
            new_volume = int(np.clip(new_volume, 0, 100))
            if abs(new_volume - prev_volume) >= VOLUME_STEP_THRESHOLD:
                set_system_volume(new_volume)
                current_volume = new_volume

        cv2.putText(frame, f"Volume: {current_volume}%", (mid_x - 80, mid_y - 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

        if results.multi_handedness and len(results.multi_handedness) > 0:
            try:
                handedness_label = results.multi_handedness[0].classification[0].label
            except Exception:
                handedness_label = None

    base_y = 30
    dy_text = 30
    lines = []
    if handedness_label:
        lines.append(f"Hand: {handedness_label}")
    if norm_dist is not None:
        lines.append(f"Norm: {norm_dist:.4f}")
    if pixel_dist is not None:
        lines.append(f"Pixel: {pixel_dist:.1f}px")
    lines.append(f"Pinch: {'YES' if pinch else 'NO'}")

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, base_y + i * dy_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, norm_dist, pixel_dist, pinch, current_volume, aspect_ratio


# ---------------- Tkinter GUI (based on reference)
class App:
    def __init__(self, root):
        self.root = root
        root.title("TaskManager-Themed Hand Tracker")
        root.geometry("1280x780")
        root.configure(bg="#111")

        # 🔒 Lock height but allow horizontal resizing
        root.resizable(True, False)

        # Define grid rows: header (0), main (1), status (2)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)
        root.columnconfigure(0, weight=1)

        # --- HEADER FRAME ---
        self.header_frame = tk.Frame(root, bg="#181818", height=70)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        self.header_frame.grid_propagate(False)

        # Main heading
        self.heading_label = tk.Label(
            self.header_frame,
            text="Infosys_GestureVolume: Volume Control with Hand Gestures",
            font=("Consolas", 18, "bold"),
            fg="#00ffcc",
            bg="#181818",
            anchor="e",  # right align
            padx=20
        )
        self.heading_label.pack(fill=tk.X, pady=(5, 0))

        # Subheading
        self.sub_label = tk.Label(
            self.header_frame,
            text="Project made by BATCH A | SNEHIL GHOSH, GAUTAM N CHIPKAR, AMRUTHA VARSHANI, AYUSH GEORGE",
            font=("Consolas", 12),
            fg="#cccccc",
            bg="#181818",
            anchor="e",
            padx=20
        )
        self.sub_label.pack(fill=tk.X, pady=(0, 5))

        # --- MAIN CONTENT FRAME ---
        self.main_frame = tk.Frame(root, bg="#111")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=1)

        # --- Left Graph Panel ---
        self.graph_panel = tk.Frame(self.main_frame, bg="#181818")
        self.graph_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.graph_panel.rowconfigure(0, weight=1)
        self.graph_panel.rowconfigure(1, weight=1)
        self.graph_panel.rowconfigure(2, weight=1)

        # --- Right Camera Feed ---
        self.video_label = tk.Label(self.main_frame, bg="#000")
        self.video_label.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # --- Fixed Status Bar ---
        self.status_label = tk.Label(root, text="Starting camera...", bg="#111", fg="#00ff99",
                                     font=("Consolas", 13), anchor="w", padx=10)
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        # --- Graph 1: LIVE PORT ---
        self.fig1 = Figure(figsize=(4, 2), dpi=100, facecolor="#111")
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor("#111")
        self.ax1.tick_params(colors="white")
        self.ax1.set_title("LIVE PORT", color="white")
        self.line_norm, = self.ax1.plot([], [], label="Norm")
        self.line_pix, = self.ax1.plot([], [], label="Pixel")
        self.ax1.legend(facecolor="#222", labelcolor="white")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.graph_panel)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- Graph 2: Aspect Ratio ---
        self.fig2 = Figure(figsize=(4, 2), dpi=100, facecolor="#111")
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor("#111")
        self.ax2.tick_params(colors="white")
        self.ax2.set_title("Aspect Ratio", color="white")
        self.line_aspect, = self.ax2.plot([], [], )
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graph_panel)
        self.canvas2.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # --- Graph 3: Placeholder ---
        self.fig3 = Figure(figsize=(4, 2), dpi=100, facecolor="#111")
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_facecolor("#111")
        self.ax3.text(0.5, 0.5, "Future Graph", color="gray", ha="center", va="center", fontsize=14)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.graph_panel)
        self.canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        # --- Camera Setup ---
        # Use CAP_DSHOW if available on Windows for lower latency
        try:
            self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(CAM_INDEX)

        if not self.cap.isOpened():
            self.status_label.config(text="Cannot open camera.")
            return

        self.running = True
        self.prev_pixel_dist = None
        self.prev_volume = get_system_volume()

        # Buffers
        self.timestamps = deque(maxlen=150)
        self.norm_dists = deque(maxlen=150)
        self.pixel_dists = deque(maxlen=150)
        self.aspect_ratios = deque(maxlen=150)
        self.start_time = time.time()

        root.bind("<Key>", self._on_keypress)
        root.protocol("WM_DELETE_WINDOW", self.stop_and_close)
        self._update_frame()

    def _on_keypress(self, event):
        if event.char.lower() == "q":
            self.stop_and_close()

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Failed to read frame.")
            self.stop_and_close()
            return

        frame = cv2.flip(frame, 1)
        frame, norm_dist, pixel_dist, pinch, self.prev_volume, aspect_ratio = process_frame(
            frame, self.prev_pixel_dist, self.prev_volume
        )
        self.prev_pixel_dist = pixel_dist

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Resize safely even if label size is zero
        lbl_w = self.video_label.winfo_width() or 960
        lbl_h = self.video_label.winfo_height() or 720

        # Pillow resampling compatibility
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS

        pil_img = pil_img.resize((lbl_w, lbl_h), resample)
        imgtk = ImageTk.PhotoImage(pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        elapsed = time.time() - self.start_time
        self.timestamps.append(elapsed)
        self.norm_dists.append(norm_dist if norm_dist is not None else np.nan)
        self.pixel_dists.append(pixel_dist if pixel_dist is not None else np.nan)
        self.aspect_ratios.append(aspect_ratio if aspect_ratio is not None else np.nan)

        # --- Graph 1: Live Port ---
        t_arr = np.array(self.timestamps)
        norm_arr = np.array(self.norm_dists)
        pix_arr = np.array(self.pixel_dists)
        if np.isfinite(np.nanmax(pix_arr)) and np.isfinite(np.nanmax(norm_arr)) and \
           (np.nanmax(pix_arr) > 0 and np.nanmax(norm_arr) > 0):
            factor = (np.nanmax(norm_arr) + 1e-6) / (np.nanmax(pix_arr) + 1e-6)
        else:
            factor = 1.0
        scaled_pix = pix_arr * factor
        try:
            self.line_norm.set_data(t_arr, norm_arr)
            self.line_pix.set_data(t_arr, scaled_pix)
            self.ax1.set_xlim(max(0, elapsed - 15), elapsed + 0.1)
            y_vals = np.concatenate([np.nan_to_num(norm_arr), np.nan_to_num(scaled_pix)])
            if np.any(np.isfinite(y_vals)):
                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                self.ax1.set_ylim(y_min - 0.1, y_max + 0.1)
            self.canvas1.draw_idle()
        except Exception:
            pass

        # --- Graph 2: Aspect Ratio ---
        aspect_arr = np.array(self.aspect_ratios)
        try:
            self.line_aspect.set_data(t_arr, aspect_arr)
            self.ax2.set_xlim(max(0, elapsed - 15), elapsed + 0.1)
            if len(aspect_arr) > 0 and np.any(np.isfinite(aspect_arr)):
                ymin, ymax = np.nanmin(aspect_arr[np.isfinite(aspect_arr)]), np.nanmax(aspect_arr[np.isfinite(aspect_arr)])
                if np.isfinite(ymin) and np.isfinite(ymax):
                    self.ax2.set_ylim(ymin - 0.1, ymax + 0.1)
            self.canvas2.draw_idle()
        except Exception:
            pass

        # --- Status Update ---
        now = time.strftime("%H:%M:%S")
        norm_str = f"{norm_dist:.4f}" if norm_dist is not None else "0.0000"
        pix_str = f"{pixel_dist:.1f}" if pixel_dist is not None else "0.0"
        asp_str = f"{aspect_ratio:.2f}" if aspect_ratio is not None else "N/A"
        self.status_label.config(
            text=f"[{now}] | Norm={norm_str} | Pixel={pix_str}px | Aspect={asp_str} | Volume={self.prev_volume}%"
        )

        self.root.after(50, self._update_frame)

    def stop_and_close(self):
        if not getattr(self, "running", False):
            try:
                self.root.destroy()
            except:
                pass
            return

        self.running = False
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except:
            pass
        try:
            hands.close()
        except:
            pass
        # Unhook keyboard hotkeys gracefully
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        self.root.after(50, self.root.destroy)


# ---------------- Run ----------------
def main():
    # Start the original hotkeys thread (preserving MC behavior)
    t_hot = threading.Thread(target=start_hotkeys, daemon=True)
    t_hot.start()

    # Start keyboard listener thread (keeps keyboard module hooked)
    t_kbd = threading.Thread(target=keyboard_listener_thread, daemon=True)
    t_kbd.start()

    root = tk.Tk()
    app = App(root)
    try:
        root.mainloop()
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass

if __name__ == "__main__":
    main()
