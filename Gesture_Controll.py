# SINGLE NOTEBOOK CELL: protected child + hotkeys + Tkinter+MediaPipe GUI
# Windows only. Requires: pip install pycaw comtypes mediapipe opencv-python Pillow matplotlib keyboard

import sys, os, time, tempfile, textwrap, threading, queue, subprocess, platform, traceback

if platform.system() != "Windows":
    raise SystemExit("This notebook cell runs only on Windows (pycaw/comtypes).")

# -------------------------
# Mic hotkey helpers (original style)
# -------------------------
try:
    import keyboard   # pip install keyboard
except Exception:
    raise SystemExit("Install required package: pip install keyboard")

# We'll use the child MicController for safe COM access, but still provide a start_hotkeys
# function and a keyboard listener thread (keeps keyboard module active).
# (Definitions of hot_inc, hot_dec, hot_toggle_mute, quit_all are below in the file.)

# -------------------------
# --- 1) Child code (COM + pycaw) written to a temp file ---
# -------------------------
child_code = r'''
# child_mic_server.py -- handles COM/pycaw in isolated process
import sys, traceback
def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
try:
    from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
    from comtypes.client import CreateObject
    from comtypes import GUID
    from ctypes import POINTER, cast
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator
except Exception as ex:
    print("ERR IMPORT", ex)
    eprint("IMPORT TRACEBACK:")
    traceback.print_exc()
    sys.exit(2)

def out(s):
    print(s)
    sys.stdout.flush()

try:
    CoInitialize()
except Exception as ex:
    out("ERR COINIT " + str(ex))
    eprint("COINIT TRACEBACK:")
    traceback.print_exc()
    sys.exit(3)

def _create_enum():
    try:
        from comtypes.client import CreateObject
        from pycaw.pycaw import IMMDeviceEnumerator
        try:
            return CreateObject("MMDeviceEnumerator.MMDeviceEnumerator", interface=IMMDeviceEnumerator)
        except Exception:
            clsid = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
            return CreateObject(clsid, interface=IMMDeviceEnumerator)
    except Exception:
        raise

def _get_vol_iface():
    enumerator = _create_enum()
    device = enumerator.GetDefaultAudioEndpoint(1, 0)  # eCapture=1, eConsole=0
    iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    from ctypes import cast, POINTER
    vol = cast(iface, POINTER(IAudioEndpointVolume))
    return vol

try:
    vol = _get_vol_iface()
except Exception as ex:
    out("ERR GET_IFACE " + str(ex))
    eprint("GET_IFACE TRACEBACK:")
    traceback.print_exc()
    try:
        CoUninitialize()
    except:
        pass
    sys.exit(4)

out("OK READY")

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        if line.lower() == "get":
            try:
                cur = vol.GetMasterVolumeLevelScalar()
                p = int(round(cur * 100))
                out(f"OK GET {p}")
            except Exception as ex:
                out("ERR GET " + str(ex))
                eprint("GET TRACE:")
                traceback.print_exc()
        elif line.lower().startswith("set:"):
            try:
                arg = line.split(":",1)[1].strip()
                v = int(arg)
                v = max(0, min(100, v))
                vol.SetMasterVolumeLevelScalar(v / 100.0, None)
                out(f"OK SET {v}")
            except Exception as ex:
                out("ERR SET " + str(ex))
                eprint("SET TRACE:")
                traceback.print_exc()
        elif line.lower() in ("mute","toggle_mute"):
            try:
                curmute = bool(vol.GetMute())
                if line.lower() == "mute":
                    vol.SetMute(1, None)
                    out("OK MUTE 1")
                else:
                    vol.SetMute(0 if curmute else 1, None)
                    out(f"OK TOGGLE {0 if curmute else 1}")
            except Exception as ex:
                out("ERR MUTE " + str(ex))
                eprint("MUTE TRACE:")
                traceback.print_exc()
        elif line.lower() == "quit":
            out("OK QUIT")
            break
        else:
            out("ERR UNKNOWN " + line)
    except Exception as ex:
        out("ERR LOOP " + str(ex))
        eprint("LOOP TRACE:")
        traceback.print_exc()
        break

try:
    CoUninitialize()
except Exception:
    pass
'''

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", prefix="child_mic_server_")
tmp.write(child_code.encode("utf-8"))
tmp.flush()
tmp.close()
child_path = tmp.name
print("Child script written to:", child_path)

# --- 2) Launch child subprocess (unbuffered) and start reader threads ---
proc = subprocess.Popen([sys.executable, "-u", child_path],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1)

stdout_q = queue.Queue()
stderr_q = queue.Queue()

def _read_stream(stream, q):
    try:
        for line in stream:
            q.put(line.rstrip("\n"))
    except Exception:
        pass
    finally:
        try: stream.close()
        except: pass

t_o = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_q), daemon=True)
t_e = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_q), daemon=True)
t_o.start(); t_e.start()

# Wait for OK READY
start = time.time(); ready = False
while time.time() - start < 6.0:
    try:
        line = stdout_q.get(timeout=0.2)
    except queue.Empty:
        continue
    print("CHILD:", line)
    if line.strip().upper().startswith("OK READY"):
        ready = True
        break

if not ready:
    # print any stderr
    while not stderr_q.empty():
        print("CHILD STDERR:", stderr_q.get())
    raise SystemExit("Child did not start correctly; check stderr above.")

# --- 3) MicController talking to child ---
class MicController:
    def __init__(self, proc, stdout_q):
        self.proc = proc
        self.stdout_q = stdout_q
        self.lock = threading.Lock()

    def _send(self, line, timeout=2.0):
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("Child process has exited.")
            try:
                self.proc.stdin.write(line.strip() + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                raise RuntimeError("Failed to write to child stdin: " + str(e))
            # wait for a response line
            start = time.time()
            while time.time() - start < timeout:
                try:
                    out = self.stdout_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if out is None:
                    continue
                return out
            raise TimeoutError("No reply from child")

    def get_volume(self):
        out = self._send("get")
        if out.upper().startswith("OK GET"):
            try:
                return int(out.split()[-1])
            except:
                return 50
        raise RuntimeError("Child error: " + out)

    def set_volume(self, percent):
        out = self._send(f"set:{int(percent)}")
        return out

    def toggle_mute(self):
        out = self._send("toggle_mute")
        return out

    def mute(self):
        out = self._send("mute")
        return out

    def close(self):
        try:
            self._send("quit", timeout=1.0)
        except Exception:
            pass
        try:
            self.proc.kill()
        except:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except:
            pass

mc = MicController(proc, stdout_q)

# Mic helper wrappers (safe for GUI)
def get_mic_volume_percent():
    try:
        return mc.get_volume()
    except Exception as e:
        print("get_mic_volume_percent error:", e)
        return 50

def set_mic_volume_percent(v):
    try:
        mc.set_volume(int(v))
    except Exception as e:
        print("set_mic_volume_percent error:", e)

def toggle_mic_mute():
    try:
        return mc.toggle_mute()
    except Exception as e:
        print("toggle_mic_mute error:", e)

# --- 4) Hotkeys that call the child (no COM in main kernel) ---
STEP_PERCENT = 2
LAST = {"inc":0, "dec":0}
COOLDOWN = 0.12
def allow_action(k):
    now = time.time()
    if now - LAST[k] >= COOLDOWN:
        LAST[k] = now
        return True
    return False

def hot_inc():
    if not allow_action("inc"): return
    cur = get_mic_volume_percent()
    new = min(100, cur + STEP_PERCENT)
    set_mic_volume_percent(new)
    print("[+] Mic ->", new, "%")

def hot_dec():
    if not allow_action("dec"): return
    cur = get_mic_volume_percent()
    new = max(0, cur - STEP_PERCENT)
    set_mic_volume_percent(new)
    print("[-] Mic ->", new, "%")

def hot_toggle_mute():
    try:
        r = toggle_mic_mute()
        print("MUTE TOGGLE ->", r)
    except Exception as e:
        print("mute toggle error", e)

def quit_all():
    try:
        mc.close()
    except:
        pass
    try:
        keyboard.unhook_all_hotkeys()
    except:
        pass
    try:
        # destroy Tk if running
        if 'root' in globals() and getattr(root, 'destroy', None):
            root.destroy()
    except:
        pass
    print("Exiting hotkeys and child.")

# Register the hotkeys in the child section too (keeps parity)
keyboard.add_hotkey('ctrl+alt+up', hot_inc)
keyboard.add_hotkey('ctrl+alt+down', hot_dec)
keyboard.add_hotkey('ctrl+alt+m', hot_toggle_mute)
keyboard.add_hotkey('ctrl+alt+q', quit_all)
print("Hotkeys registered: Ctrl+Alt+Up/Down/M/Q")

# -------------------------
# --- 5) GUI + MediaPipe (same logic; uses set_mic_volume_percent/get_mic_volume_percent) ---
# -------------------------
import math, cv2, mediapipe as mp, numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
CAM_INDEX = 0
PINCH_PIXEL_THRESHOLD = 40
PINCH_NORM_THRESHOLD = 0.03
MIN_DIST = 25
MAX_DIST = 190
VOLUME_STEP_THRESHOLD = 4

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

def set_system_volume(vol_percent):
    set_mic_volume_percent(int(vol_percent))

def get_system_volume():
    return get_mic_volume_percent()

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


# ---------------- Tkinter GUI (two graphs layout)
class App:
    def __init__(self, root):
        self.root = root
        root.title("Gesture Mic Volume")
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
            text="Project made by BATCH A | SNEHIL GHOSH, GAUTAM N CHIPKAR, AMRUTHA VARSHANI, AYUSH GORGE",
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

        # --- Left Graph Panel (2 rows) ---
        self.graph_panel = tk.Frame(self.main_frame, bg="#181818")
        self.graph_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.graph_panel.rowconfigure(0, weight=1)
        self.graph_panel.rowconfigure(1, weight=1)

        # --- Right Camera Feed ---
        self.video_label = tk.Label(self.main_frame, bg="#000")
        self.video_label.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # --- Fixed Status Bar ---
        self.status_label = tk.Label(root, text="Starting camera...", bg="#111", fg="#00ff99",
                                     font=("Consolas", 13), anchor="w", padx=10)
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        # --- Graph 2: Dual-axis (pixel distance ; volume ; pinch state) ---
        self.fig2 = Figure(figsize=(4, 2), dpi=100, facecolor="#111")
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor("#111")
        self.ax2.tick_params(colors="white")
        self.ax2.set_title("Pixel Dist (left)  |  Volume & Pinch (right)", color="white")
        self.ax2.set_xlabel("Time (s)")
        # primary axis (left) for pixel distance
        self.line_pix2, = self.ax2.plot([], [], label="Pixel Dist")
        # secondary axis for volume (+ pinch markers scaled)
        self.ax2b = self.ax2.twinx()
        self.ax2b.tick_params(colors="white")
        # volume line colored pink (#ff69b4)
        self.line_vol, = self.ax2b.plot([], [], label="Volume (%)", linestyle='-', linewidth=1.2, color="#ff69b4")
        self.line_pinch_markers, = self.ax2b.plot([], [], linestyle='None', marker='o', markersize=4, label="Pinch")
        lines_2, labels_2 = self.ax2.get_legend_handles_labels()
        lines_2b, labels_2b = self.ax2b.get_legend_handles_labels()
        self.ax2.legend(lines_2 + lines_2b, labels_2 + labels_2b, facecolor="#222", labelcolor="white")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graph_panel)
        self.canvas2.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # --- Camera Setup ---
        try:
            self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(CAM_INDEX)

        if not self.cap.isOpened():
            self.status_label.config(text="Cannot open camera.")
            return

        self.running = True
        self.prev_pixel_dist = None
        self.prev_volume = get_mic_volume_percent()

        # Buffers (aspect ratio removed)
        self.timestamps = deque(maxlen=150)
        self.norm_dists = deque(maxlen=150)
        self.pixel_dists = deque(maxlen=150)
        self.volumes = deque(maxlen=150)
        self.pinches = deque(maxlen=150)
        self.start_time = time.time()

        root.bind("<Key>", self._on_keypress)
        root.protocol("WM_DELETE_WINDOW", self.stop_and_close)
        self._update_frame()

    def _on_keypress(self, event):
        if hasattr(event, "char") and event.char and event.char.lower() == "q":
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
        # keep process_frame unchanged (it still returns aspect_ratio as before), but we will ignore it
        frame, norm_dist, pixel_dist, pinch, self.prev_volume, _aspect = process_frame(
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
        self.volumes.append(self.prev_volume if self.prev_volume is not None else np.nan)
        self.pinches.append(1.0 if pinch else 0.0)

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

        # --- Graph 2: Dual-axis pixel / volume / pinch ---
        try:
            vol_arr = np.array(self.volumes)
            pinch_arr = np.array(self.pinches)
            self.line_pix2.set_data(t_arr, pix_arr)
            self.ax2.set_xlim(max(0, elapsed - 15), elapsed + 0.1)
            if np.any(np.isfinite(pix_arr)):
                pmin, pmax = np.nanmin(pix_arr[np.isfinite(pix_arr)]), np.nanmax(pix_arr[np.isfinite(pix_arr)])
                if np.isfinite(pmin) and np.isfinite(pmax):
                    pad = max(1.0, (pmax - pmin) * 0.1)
                    self.ax2.set_ylim(max(0.0, pmin - pad), pmax + pad)
            # volume plotted in pink
            self.line_vol.set_data(t_arr, vol_arr)
            self.ax2b.set_xlim(max(0, elapsed - 15), elapsed + 0.1)
            pinch_scaled = pinch_arr * 100.0
            self.line_pinch_markers.set_data(t_arr, pinch_scaled)
            self.ax2b.set_ylim(-5, 105)
            self.canvas2.draw_idle()
        except Exception:
            pass

        # --- Status Update ---
        now = time.strftime("%H:%M:%S")
        norm_str = f"{norm_dist:.4f}" if norm_dist is not None else "0.0000"
        pix_str = f"{pixel_dist:.1f}" if pixel_dist is not None else "0.0"
        self.status_label.config(
            text=f"[{now}] | Norm={norm_str} | Pixel={pix_str}px | Volume={self.prev_volume}%"
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
        # cleanup child and exit
        try: mc.close()
        except: pass
        self.root.after(50, self.root.destroy)


# ---------------- Run ----------------
# Provide start_hotkeys and keyboard_listener_thread so main() can spawn them
def start_hotkeys():
    """
    Register hotkeys and print current mic volume. Uses the hot_inc/hot_dec/hot_toggle_mute/quit_all functions defined above.
    """
    try:
        # Re-registering hotkeys here is harmless (keyboard will ensure single registration per exact handler)
        keyboard.add_hotkey('ctrl+alt+up', hot_inc)
        keyboard.add_hotkey('ctrl+alt+down', hot_dec)
        keyboard.add_hotkey('ctrl+alt+m', hot_toggle_mute)
        keyboard.add_hotkey('ctrl+alt+q', quit_all)
    except Exception as e:
        print("start_hotkeys: failed to register hotkeys:", e)
    try:
        cur = get_mic_volume_percent()
        print(f"Initial mic volume: {cur}%")
    except Exception:
        pass
    print("Hotkeys active. Ctrl+Alt+Up/Down/M/Q")

def keyboard_listener_thread():
    # Keep the keyboard module active for global events
    try:
        keyboard.hook(lambda e: None)
        keyboard.wait()
    except Exception:
        # if hook/wait fail, just return and let the app run
        pass

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
