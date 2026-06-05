# 👁️ NeuroCursor

> **Real-time, hardware-free gaze-controlled mouse using computer vision — no eye-tracking hardware required.**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?style=flat-square&logo=opencv)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

**NeuroCursor** is a real-time, software-only gaze-controlled mouse that maps a user's eye gaze to screen coordinates using a standard laptop webcam — no specialized eye-tracking hardware needed.

The system processes live facial landmarks at **30 fps**, applies a **25-point RBF Thin Plate Spline calibration grid** for spatial accuracy, and runs a **6-state adaptive Kalman filter** to eliminate cursor jitter — achieving smooth, responsive hands-free PC control.

**Blink detection** via Eye Aspect Ratio (EAR) simulates left and right mouse clicks, making the system fully operable without any physical input device.

> Built for accessibility. Designed for real-world usability.

---

## ✨ Features

- 🎥 **Hardware-free** — works on any standard laptop or USB webcam
- ⚡ **Real-time** — stable 30 fps gaze tracking
- 📐 **25-point RBF Calibration** — Thin Plate Spline mapping for precise gaze-to-screen coordinate conversion
- 🔄 **6-State Kalman Filter** — adaptive noise filtering for smooth cursor movement (60%+ jitter reduction)
- 👁️ **EAR Blink Detection** — left and right click simulation via eye blink geometry
- 🖥️ **Cross-platform** — runs on Windows, macOS, and Linux
- 🧩 **Zero external hardware dependencies** — pure Python, standard webcam

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Face & Landmark Detection | MediaPipe FaceMesh (468 landmarks) |
| Computer Vision Pipeline | OpenCV |
| Gaze-to-Screen Mapping | SciPy — RBF Thin Plate Spline |
| Noise Filtering | Custom 6-State Kalman Filter |
| Mouse Control | PyAutoGUI |
| Language | Python 3.8+ |

---

## 📁 Repository Structure

```
NeuroCursor/
│
├── eye_mouse_control.py    # Main application — run this
├── Requirements.txt        # All dependencies
├── Neuro-Cursor_Paper.pdf   # Research paper
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/AryaBadugu/NeuroCursor.git
cd NeuroCursor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the application
```bash
python neurocursor.py
```

### Calibration
1. When the calibration screen appears, **look at each dot** as it appears on screen
2. The system collects gaze data across **25 calibration points**
3. Once calibration is complete, cursor control activates automatically

### Controls
| Action | How |
|---|---|
| Move cursor | Look at the target location on screen |
| Left click | Blink your **left eye** |
| Right click | Blink your **right eye** |
| Exit | Press `ESC` |

---

## 🔬 How It Works

### Pipeline Overview

```
Webcam Feed
    ↓
MediaPipe FaceMesh (468 facial landmarks @ 30fps)
    ↓
Iris Landmark Extraction (gaze vector computation)
    ↓
25-Point RBF Calibration Grid (Thin Plate Spline mapping)
    ↓
6-State Adaptive Kalman Filter (jitter suppression)
    ↓
PyAutoGUI (cursor movement)
    ↓
EAR Blink Detection (click simulation)
```

### Key Technical Components

#### 1. Gaze Estimation
MediaPipe's FaceMesh model detects **468 facial landmarks** in real time. Iris landmark coordinates are extracted and used to compute a gaze vector relative to the eye socket geometry.

#### 2. Calibration — RBF Thin Plate Spline
Raw gaze coordinates are noisy and non-linear in their mapping to screen space. A **25-point calibration grid** collects gaze samples at known screen positions. A **Radial Basis Function (RBF) Thin Plate Spline** interpolation (via `scipy.interpolate.RBFInterpolator`) learns the non-linear mapping from gaze space to screen space for the user's specific head position and webcam setup.

#### 3. Kalman Filter — Noise Suppression
A **6-state adaptive Kalman filter** (position x/y, velocity x/y, acceleration x/y) smooths the raw gaze output over time. The filter balances two competing objectives:
- **Too aggressive** → cursor lags and feels unresponsive
- **Too loose** → jitter returns

Tuning the **process noise covariance (Q)** and **measurement noise covariance (R)** matrices is what makes the cursor feel natural. The result is **60%+ jitter reduction** compared to raw landmark output.

#### 4. Blink Detection — EAR
The **Eye Aspect Ratio (EAR)** is computed from 6 eye landmark coordinates:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

When EAR drops below a threshold for a sustained number of frames, a blink is detected and the corresponding mouse click is triggered. Left and right eyes are tracked independently for left/right click differentiation.

---

## 📄 Research Paper

The full technical paper describing the system architecture, methodology, and evaluation is included in the repository:

📎 [`Neuro-Cursor_Paper.pdf`](./Neuro-Cursor_Paper.pdf)

---

## 📋 Requirements

```
mediapipe
opencv-python
pyautogui
scipy
numpy
```

Install all with:
```bash
pip install -r requirements.txt
```

> **Note:** Python 3.8–3.11 recommended. MediaPipe may have compatibility issues with Python 3.12+.

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| Webcam not detected | Check camera permissions; try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` |
| Cursor too jittery | Improve lighting conditions; ensure face is centered in frame |
| Calibration inaccurate | Redo calibration; keep head still during calibration |
| Low FPS | Close background apps; reduce frame resolution in config |
| `mediapipe` install error | Use Python 3.8–3.11; try `pip install mediapipe==0.10.7` |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Tracking Speed | 30 fps (standard webcam) |
| Jitter Reduction | 60%+ (vs. raw landmark output) |
| Calibration Points | 25-point grid |
| Hardware Required | Standard laptop webcam |
| External Dependencies | None |

---

## 🌟 Star this repo if you found it useful!

```
If you use NeuroCursor in your research or project, please cite the paper included in this repository.
```

---

<p align="center">
  Built with 👁️ by <a href="https://github.com/AryaBadugu">Arya Badugu</a> | SIES GST, Navi Mumbai
</p>
