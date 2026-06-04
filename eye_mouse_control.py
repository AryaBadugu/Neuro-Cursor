"""
NEURO-CURSOR 
=========================================================
Features:
  1. FIXED vertical gaze normalization (ny / eye_width)
  2. Polynomial regression calibration (proven stable)
  3. Weighted dual-eye fusion
  4. Wink-based click detection (individual eye tracking)
  5. Gentle head movement for coarse screen positioning
  6. Edge-gaze scrolling (look at top/bottom of screen)
  7. Adaptive Kalman + Median + Deadzone filtering
  8. Eye region extraction with landmark display
  9. Cursor freeze during all actions

Controls:
  Left eye wink   = Left click
  Right eye wink  = Right click
  Long blink      = Double click (hold eyes closed ~0.5s)
  Look at top     = Scroll up
  Look at bottom  = Scroll down
  Slight head turn= Coarse cursor positioning
  'c' key         = Recalibrate
  'q' key / ESC   = Quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque
import threading


try:
    import uiautomation as auto
    HAS_UIAUTOMATION = True
    print("[INIT] UI Automation loaded — snap-to-element available ('n' to toggle)")
except ImportError:
    HAS_UIAUTOMATION = False
    print("[INIT] uiautomation not installed — snap-to-element disabled")

# ================================================================
#  CONFIGURATION
# ================================================================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
print(f"[INIT] Screen: {SCREEN_W}x{SCREEN_H}")

CAM_W, CAM_H = 640, 480

# ---- Calibration ----
CAL_SETTLE_SEC = 1.0
MIN_SAMPLES = 40
RANSAC_ITERATIONS = 50
RANSAC_THRESHOLD = 120
IQR_FACTOR = 1.5
CONSISTENCY_THRESH = 0.025     # More lenient — fewer point rejections

# ---- Wink Detection ----
WINK_CLOSED = 0.19             # EAR below this = eye is closed
WINK_OPEN = 0.25               # EAR above this = eye is open
BLINK_BOTH_CLOSED = 0.22       # Both eyes below this = blink
WINK_MIN_DUR = 0.10            # Quicker response for winks
WINK_MAX_DUR = 0.8             # More forgiving max duration
LONG_BLINK_THRESH = 0.40       # Blink >400ms = double click (natural!)
CLICK_COOLDOWN = 0.35          # Faster click recovery
PAUSE_BLINK_THRESH = 2.0       # Close both eyes 2s = toggle pause
WINK_CONFIRM_MS = 0.08         # 80ms delay to confirm wink (vs natural blink)

# ---- Edge-Gaze Scrolling ----
SCROLL_ZONE = 0.25             # Top/bottom 25% of screen (easier to trigger)
SCROLL_DWELL = 0.4             # Faster scroll activation
SCROLL_MIN = 2
SCROLL_MAX = 20
SCROLL_INTERVAL = 0.04          # Faster scroll ticks

# ---- Head Movement Assist ----
HEAD_DEADZONE = 0.012          # Ignore natural postural sway (normalized)
HEAD_WEIGHT = 0.30             # 30% head, 70% gaze — precision assist
HEAD_SENS_X = SCREEN_W * 1.2   # Sensitivity past deadzone
HEAD_SENS_Y = SCREEN_H * 1.2

# ---- Adaptive Kalman ----
KALMAN_Q_SLOW = 0.02           # Smoother during fixation
KALMAN_Q_FAST = 0.08
KALMAN_R = 12.0                # Lowered to trust gaze more at edges
VELOCITY_THRESHOLD = 200

# ---- Filtering ----
MEDIAN_WINDOW = 5
DEADZONE_PX = 1.5              # Restored to smooth v6/v7 value
EMA_ALPHA_SLOW = 0.22          # Slightly faster to reach corners
EMA_ALPHA_FAST = 0.55          # Responsive during saccades
EMA_VEL_THRESH = 40            # Pixel velocity threshold

# ---- Snap-to-Element ----
SNAP_QUERY_INTERVAL = 0.20     # Query UI elements every 200ms
SNAP_DWELL_THRESH = 0.15       # 150ms dwell before snapping
SNAP_RELEASE_MARGIN = 0.5      # Release when gaze is 50% beyond element bounds
SNAP_MAX_FRAC = 0.40           # Ignore elements wider/taller than 40% of screen
SNAP_MIN_SIZE = 10             # Ignore elements smaller than 10px

# ---- MediaPipe Landmarks ----
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE_LR = (33, 133)       # outer, inner
RIGHT_EYE_LR = (362, 263)
LEFT_EYE_TB = (159, 145)
RIGHT_EYE_TB = (386, 374)
NOSE_TIP = 1
L_BROW = [66, 105, 63, 70, 46]
R_BROW = [296, 334, 293, 300, 276]
L_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                 159, 160, 161, 246]
R_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                 386, 385, 384, 398]


# ================================================================
#  ADAPTIVE KALMAN 2D
# ================================================================
class AdaptiveKalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 500
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * KALMAN_R
        self.last_time = None
        self.last_pos = None

    def step(self, measurement):
        t = time.time()
        if self.last_time is None:
            self.last_time = t
            self.last_pos = measurement
            self.x[0, 0], self.x[1, 0] = measurement
            return measurement
        dt = max(t - self.last_time, 0.016)
        if self.last_pos:
            dx = measurement[0] - self.last_pos[0]
            dy = measurement[1] - self.last_pos[1]
            vel = math.sqrt(dx * dx + dy * dy) / dt
        else:
            vel = 0
        Q_val = KALMAN_Q_FAST if vel > VELOCITY_THRESHOLD else KALMAN_Q_SLOW
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.eye(4) * Q_val
        z = np.array(measurement).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_time = t
        self.last_pos = measurement
        return float(self.x[0, 0]), float(self.x[1, 0])


# ================================================================
#  SNAP-TO-ELEMENT ENGINE (Windows UI Automation)
# ================================================================
class SnapEngine:
    """Snaps the cursor to UI elements (icons, buttons, files, links, etc.)
    when the user's gaze lands nearby. Uses Windows UI Automation API."""

    def __init__(self):
        self.enabled = True
        self.snap_pos = None         # (cx, cy) of snapped element
        self.snap_rect = None        # (l, t, r, b) of snapped element
        self.last_query_time = 0
        self.pending_rect = None     # Element being dwelled on
        self.dwell_start = None

    def query(self, gaze_x, gaze_y):
        """Returns (snap_x, snap_y) or None."""
        if not self.enabled or not HAS_UIAUTOMATION:
            return None

        now = time.time()

        # If currently snapped, check if gaze still near the element
        if self.snap_pos and self.snap_rect:
            l, t, r, b = self.snap_rect
            ew, eh = r - l, b - t
            margin_x = max(ew * SNAP_RELEASE_MARGIN, 30)
            margin_y = max(eh * SNAP_RELEASE_MARGIN, 30)
            if (l - margin_x <= gaze_x <= r + margin_x and
                    t - margin_y <= gaze_y <= b + margin_y):
                return self.snap_pos  # Stay snapped
            else:
                # Gaze moved away — release
                self.snap_pos = None
                self.snap_rect = None
                self.pending_rect = None
                self.dwell_start = None

        # Throttle UI queries
        if now - self.last_query_time < SNAP_QUERY_INTERVAL:
            return self.snap_pos
        self.last_query_time = now

        try:
            element = auto.ControlFromPoint(int(gaze_x), int(gaze_y))
            if element:
                rect = element.BoundingRectangle
                w = rect.right - rect.left
                h = rect.bottom - rect.top

                # Filter: ignore oversized (desktop/window) and tiny elements
                if (SNAP_MIN_SIZE < w < SCREEN_W * SNAP_MAX_FRAC and
                        SNAP_MIN_SIZE < h < SCREEN_H * SNAP_MAX_FRAC):
                    new_rect = (rect.left, rect.top, rect.right, rect.bottom)

                    if new_rect != self.pending_rect:
                        # New element — start dwell timer
                        self.pending_rect = new_rect
                        self.dwell_start = now
                    elif self.dwell_start and now - self.dwell_start >= SNAP_DWELL_THRESH:
                        # Dwell confirmed — snap!
                        cx = (rect.left + rect.right) // 2
                        cy = (rect.top + rect.bottom) // 2
                        self.snap_pos = (cx, cy)
                        self.snap_rect = new_rect
                        return self.snap_pos
                else:
                    # Element too large or too small — clear pending
                    self.pending_rect = None
                    self.dwell_start = None
        except Exception:
            pass

        return self.snap_pos


# ================================================================
#  NEURO-CURSOR CORE
# ================================================================
class NeuroCursor:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        # Calibration
        self.poly_coeffs = None
        self.kalman = AdaptiveKalman2D()
        self.prev_output = None
        self.med_x = deque(maxlen=MEDIAN_WINDOW)
        self.med_y = deque(maxlen=MEDIAN_WINDOW)
        self.ema_pos = None             # EMA smoothing state

        # Wink/blink state
        self.wink_state = None        # 'left_wink', 'right_wink', 'blink', None
        self.wink_start = 0
        self.last_click_t = 0
        self.frozen_gaze = None
        self.is_action = False        # True when any eye is closed

        # Pause mode
        self.paused = False
        self.pause_blink_start = None

        # Scroll
        self.scroll_zone_enter_t = None
        self.scroll_zone = None
        self.last_scroll_t = 0
        self.is_scrolling = False

        # Head tracking
        self.nose_ref = None          # Set during calibration

        # Cursor position
        self.last_cx = SCREEN_W // 2
        self.last_cy = SCREEN_H // 2

        # Display
        self.last_lm = None
        self.action_text = ""
        self.action_time = 0

        print("[INIT] NeuroCursor v8.1 ready")

    def preprocess(self, frame):
        """CLAHE low-light enhancement."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a, b = cv2.split(lab)
        l_ch = self.clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a, b]), cv2.COLOR_LAB2BGR)

    def _pt(self, lm, idx):
        return np.array([lm[idx].x, lm[idx].y])

    def _px(self, lm, idx, w, h):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    # ---- GAZE (FIXED vertical normalization) ----
    def compute_gaze(self, lm):
        """
        Gaze computation with CORRECTED scaling.
        Both nx and ny are divided by eye_width, giving similar numeric
        ranges. This is CRITICAL for polynomial regression accuracy.
        """
        L_l = self._pt(lm, LEFT_EYE_LR[0])
        L_r = self._pt(lm, LEFT_EYE_LR[1])
        L_p = self._pt(lm, LEFT_PUPIL)
        R_l = self._pt(lm, RIGHT_EYE_LR[0])
        R_r = self._pt(lm, RIGHT_EYE_LR[1])
        R_p = self._pt(lm, RIGHT_PUPIL)

        def norm_eye(p_l, p_r, p_p):
            eye_vec = p_r - p_l
            width_sq = np.dot(eye_vec, eye_vec)
            if width_sq < 1e-8:
                return None, 0
            eye_width = math.sqrt(width_sq)
            pupil_vec = p_p - p_l

            # Horizontal: fraction along eye axis [~0.3 to 0.7]
            nx = np.dot(pupil_vec, eye_vec) / width_sq

            # Vertical: perpendicular distance / eye_width [~-0.15 to 0.15]
            # CRITICAL FIX: dividing by eye_width gives similar scale to nx
            perp = np.array([-eye_vec[1], eye_vec[0]])
            pn = np.linalg.norm(perp)
            if pn > 1e-8:
                perp /= pn
            ny = np.dot(pupil_vec, perp) / eye_width

            return np.array([nx, ny]), eye_width

        L, wL = norm_eye(L_l, L_r, L_p)
        R, wR = norm_eye(R_l, R_r, R_p)

        if L is not None and R is not None:
            if np.linalg.norm(L - R) > 0.3:
                return None
        if L is None and R is None:
            return None
        if L is None:
            return R
        if R is None:
            return L
        total = wL + wR
        if total < 1e-8:
            return (L + R) / 2
        return (L * wL + R * wR) / total

    # ---- EAR per eye ----
    def _ear_single(self, lm, tb, lr):
        t, b = self._pt(lm, tb[0]), self._pt(lm, tb[1])
        l, r = self._pt(lm, lr[0]), self._pt(lm, lr[1])
        return np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6)

    # ---- HEAD DELTA ----
    def compute_head_delta(self, lm):
        """Compute head movement with deadzone to ignore natural sway."""
        nose = self._pt(lm, NOSE_TIP)
        if self.nose_ref is None:
            return 0.0, 0.0
        # Negate X because cv2.flip mirrors the frame
        dx = -(nose[0] - self.nose_ref[0])
        dy = nose[1] - self.nose_ref[1]
        
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < HEAD_DEADZONE:
            return 0.0, 0.0
            
        # Smoothly scale past the deadzone so it doesn't jump
        scale = (dist - HEAD_DEADZONE) / dist
        return dx * scale, dy * scale

    # ---- EYE DISPLAY ----
    def extract_eye_display(self, frame, lm):
        h, w = frame.shape[:2]

        def get_eye_roi(contour, iris, brow, pupil):
            all_pts = contour + brow
            xs = [int(lm[i].x * w) for i in all_pts]
            ys = [int(lm[i].y * h) for i in all_pts]
            pad = 15
            x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
            x2, y2 = min(w, max(xs)+pad), min(h, max(ys)+pad)
            if x2-x1 < 10 or y2-y1 < 10:
                return None
            roi = frame[y1:y2, x1:x2].copy()
            cpts = [(int(lm[i].x*w)-x1, int(lm[i].y*h)-y1) for i in contour]
            for i in range(len(cpts)):
                cv2.line(roi, cpts[i], cpts[(i+1)%len(cpts)], (0,180,0), 1)
                cv2.circle(roi, cpts[i], 1, (0,255,0), -1)
            for idx in iris:
                cv2.circle(roi, (int(lm[idx].x*w)-x1, int(lm[idx].y*h)-y1), 2, (255,255,0), -1)
            cv2.circle(roi, (int(lm[pupil].x*w)-x1, int(lm[pupil].y*h)-y1), 2, (0,0,255), -1)
            bpts = [(int(lm[i].x*w)-x1, int(lm[i].y*h)-y1) for i in brow]
            for i in range(len(bpts)-1):
                cv2.line(roi, bpts[i], bpts[i+1], (0,255,255), 1)
            roi = cv2.resize(roi, (roi.shape[1]*3, roi.shape[0]*3), interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_ch, a, b_ch = cv2.split(lab)
            l_ch = self.clahe.apply(l_ch)
            return cv2.cvtColor(cv2.merge([l_ch, a, b_ch]), cv2.COLOR_LAB2BGR)

        le = get_eye_roi(L_EYE_CONTOUR, LEFT_IRIS, L_BROW, LEFT_PUPIL)
        re = get_eye_roi(R_EYE_CONTOUR, RIGHT_IRIS, R_BROW, RIGHT_PUPIL)
        if le is not None and re is not None:
            mh = max(le.shape[0], re.shape[0])
            if le.shape[0] < mh:
                le = cv2.copyMakeBorder(le, 0, mh-le.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            if re.shape[0] < mh:
                re = cv2.copyMakeBorder(re, 0, mh-re.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            return np.hstack([le, np.zeros((mh, 10, 3), dtype=np.uint8), re])
        return le or re

    # ---- PROCESS FRAME ----
    def process_frame(self, frame):
        """Returns (gaze, click_type, is_action, eye_display).
        click_type: None, 'left', 'right', 'double'
        is_action: True when an eye is closed (freeze cursor)
        """
        # CLAHE low-light enhancement
        enhanced = self.preprocess(frame)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        gaze = None
        click_type = None
        eye_display = None

        if not res.multi_face_landmarks:
            return gaze, click_type, self.is_action, eye_display

        lm = res.multi_face_landmarks[0].landmark
        self.last_lm = lm
        eye_display = self.extract_eye_display(frame, lm)

        # Gaze
        raw_gaze = self.compute_gaze(lm)

        # Individual eye EAR
        ear_l = self._ear_single(lm, LEFT_EYE_TB, LEFT_EYE_LR)
        ear_r = self._ear_single(lm, RIGHT_EYE_TB, RIGHT_EYE_LR)
        now = time.time()

        # ---- WINK / BLINK STATE MACHINE ----
        # KEY FIX: During natural blinks, one eye closes ~20-60ms before
        # the other. We use WINK_CONFIRM_MS to wait and check if the
        # other eye also closes (= blink) before confirming a wink.

        both_closed = ear_l < BLINK_BOTH_CLOSED and ear_r < BLINK_BOTH_CLOSED
        left_only = ear_l < WINK_CLOSED and ear_r > WINK_OPEN
        right_only = ear_r < WINK_CLOSED and ear_l > WINK_OPEN

        if both_closed:
            # Both eyes closed — always override to 'blink'
            # This catches natural blinks even if wink was detected first
            if self.wink_state != 'blink':
                self.wink_state = 'blink'
                self.wink_start = now
            self.is_action = True

            # ---- PAUSE TOGGLE: 2-second eyes-closed ----
            if self.pause_blink_start is None:
                self.pause_blink_start = now
            elif now - self.pause_blink_start >= PAUSE_BLINK_THRESH:
                self.paused = not self.paused
                state = "PAUSED" if self.paused else "RESUMED"
                print(f"[PAUSE] Tracking {state} (eyes-closed toggle)")
                self.action_text = f"TRACKING {state}"
                self.action_time = now
                self.pause_blink_start = None  # Reset so it doesn't re-toggle

        elif left_only:
            if self.wink_state != 'left_wink':
                self.wink_state = 'left_wink'
                self.wink_start = now
            self.is_action = True

        elif right_only:
            if self.wink_state != 'right_wink':
                self.wink_state = 'right_wink'
                self.wink_start = now
            self.is_action = True

        else:
            self.pause_blink_start = None  # Eyes opened — reset pause timer
            # Eyes are open — check what action to fire
            if self.wink_state is not None:
                dur = now - self.wink_start

                if self.wink_state == 'left_wink':
                    # Only fire wink if it lasted > confirm delay
                    # (natural blinks get reclassified to 'blink' within 80ms)
                    if (dur > WINK_CONFIRM_MS and WINK_MIN_DUR < dur < WINK_MAX_DUR
                            and now - self.last_click_t > CLICK_COOLDOWN):
                        click_type = 'left'
                        self.action_text = "LEFT CLICK"
                        self.action_time = now
                        self.last_click_t = now

                elif self.wink_state == 'right_wink':
                    if (dur > WINK_CONFIRM_MS and WINK_MIN_DUR < dur < WINK_MAX_DUR
                            and now - self.last_click_t > CLICK_COOLDOWN):
                        click_type = 'right'
                        self.action_text = "RIGHT CLICK"
                        self.action_time = now
                        self.last_click_t = now

                elif self.wink_state == 'blink':
                    # Long blink (>400ms) = DOUBLE CLICK
                    # Short blink (<400ms) = ignored (natural reflex)
                    if dur >= LONG_BLINK_THRESH and dur < WINK_MAX_DUR:
                        if now - self.last_click_t > CLICK_COOLDOWN:
                            click_type = 'double'
                            self.action_text = "DOUBLE CLICK"
                            self.action_time = now
                            self.last_click_t = now

                self.wink_state = None

            # Update gaze only when eyes are open
            self.is_action = False
            if raw_gaze is not None:
                gaze = raw_gaze
                self.frozen_gaze = raw_gaze

        # During action (eyes actually closed), use frozen gaze
        if self.is_action:
            gaze = self.frozen_gaze

        return gaze, click_type, self.is_action, eye_display

    # ---- SCROLL ZONE ----
    def check_scroll_zone(self, cursor_y):
        now = time.time()
        top_line = SCREEN_H * SCROLL_ZONE
        bot_line = SCREEN_H * (1 - SCROLL_ZONE)

        zone = None
        if cursor_y < top_line:
            zone = 'up'
        elif cursor_y > bot_line:
            zone = 'down'

        if zone is None:
            self.scroll_zone_enter_t = None
            self.scroll_zone = None
            self.is_scrolling = False
            return 0

        if zone != self.scroll_zone:
            self.scroll_zone = zone
            self.scroll_zone_enter_t = now
            self.is_scrolling = False
            return 0

        if now - self.scroll_zone_enter_t < SCROLL_DWELL:
            return 0
        if now - self.last_scroll_t < SCROLL_INTERVAL:
            return 0

        self.is_scrolling = True
        self.last_scroll_t = now

        if zone == 'up':
            depth = 1.0 - cursor_y / top_line
        else:
            depth = (cursor_y - bot_line) / (SCREEN_H - bot_line)
        depth = max(0, min(1, depth))
        amount = int(SCROLL_MIN + depth * (SCROLL_MAX - SCROLL_MIN))
        return amount if zone == 'up' else -amount

    # ---- CALIBRATION ----
    def _build_features(self, raw):
        N = len(raw)
        G = np.zeros((N, 6))
        G[:, 0] = raw[:, 0]
        G[:, 1] = raw[:, 1]
        G[:, 2] = raw[:, 0] ** 2
        G[:, 3] = raw[:, 1] ** 2
        G[:, 4] = raw[:, 0] * raw[:, 1]
        G[:, 5] = 1
        return G

    def calibrate(self, raw_data, screen_data):
        raw = np.array(raw_data)
        screen = np.array(screen_data)
        N = len(raw)
        if N < 6:
            G = self._build_features(raw)
            Cx = np.linalg.lstsq(G, screen[:, 0], rcond=None)[0]
            Cy = np.linalg.lstsq(G, screen[:, 1], rcond=None)[0]
            self.poly_coeffs = (Cx, Cy)
            self._reset_filters()
            return True
        best_inliers = None
        best_score = 0
        for _ in range(RANSAC_ITERATIONS):
            idx = np.random.choice(N, min(6, N), replace=False)
            G_s = self._build_features(raw[idx])
            try:
                Cx_s = np.linalg.lstsq(G_s, screen[idx, 0], rcond=None)[0]
                Cy_s = np.linalg.lstsq(G_s, screen[idx, 1], rcond=None)[0]
            except Exception:
                continue
            G_all = self._build_features(raw)
            errs = np.sqrt((G_all @ Cx_s - screen[:, 0])**2 +
                           (G_all @ Cy_s - screen[:, 1])**2)
            mask = errs < RANSAC_THRESHOLD
            s = np.sum(mask)
            if s > best_score:
                best_score = s
                best_inliers = mask
        if best_inliers is not None and np.sum(best_inliers) >= 6:
            raw_c, scr_c = raw[best_inliers], screen[best_inliers]
        else:
            raw_c, scr_c = raw, screen
        G = self._build_features(raw_c)
        Cx = np.linalg.lstsq(G, scr_c[:, 0], rcond=None)[0]
        Cy = np.linalg.lstsq(G, scr_c[:, 1], rcond=None)[0]
        self.poly_coeffs = (Cx, Cy)
        self._reset_filters()
        print(f"[CAL] Polynomial fit on {len(raw_c)}/{N} inliers")
        return True

    def _reset_filters(self):
        self.kalman = AdaptiveKalman2D()
        self.prev_output = None
        self.med_x.clear()
        self.med_y.clear()
        self.ema_pos = None

    def map_to_screen(self, gaze):
        if self.poly_coeffs is None:
            return None
        gx, gy = gaze
        f = np.array([gx, gy, gx**2, gy**2, gx*gy, 1])
        Cx, Cy = self.poly_coeffs
        sx, sy = float(np.dot(Cx, f)), float(np.dot(Cy, f))
        return self.stretch_to_edges(sx, sy)

    def stretch_to_edges(self, sx, sy):
        """Stretch mapped coordinates toward screen edges to fix corner under-reach."""
        nx = sx / SCREEN_W
        ny = sy / SCREEN_H
        STRETCH = 1.3
        nx = 0.5 + (nx - 0.5) * min(STRETCH, 1.0 + abs(nx - 0.5) * 0.6)
        ny = 0.5 + (ny - 0.5) * min(STRETCH, 1.0 + abs(ny - 0.5) * 0.6)
        return max(0, min(SCREEN_W, nx * SCREEN_W)), max(0, min(SCREEN_H, ny * SCREEN_H))

    def apply_filtering(self, sx, sy):
        # 1. Adaptive EMA — smooth during fixation, responsive during saccades
        vel = 0
        if self.ema_pos is not None:
            dx, dy = sx - self.ema_pos[0], sy - self.ema_pos[1]
            vel = math.sqrt(dx * dx + dy * dy)
        alpha = EMA_ALPHA_FAST if vel > EMA_VEL_THRESH else EMA_ALPHA_SLOW
        if self.ema_pos is None:
            self.ema_pos = (sx, sy)
        else:
            self.ema_pos = (self.ema_pos[0] * (1 - alpha) + sx * alpha,
                           self.ema_pos[1] * (1 - alpha) + sy * alpha)
        sx, sy = self.ema_pos

        # 2. Kalman
        sx, sy = self.kalman.step((sx, sy))

        # 3. Median filter
        self.med_x.append(sx)
        self.med_y.append(sy)
        if len(self.med_x) >= 3:
            sx = float(np.median(list(self.med_x)))
            sy = float(np.median(list(self.med_y)))

        # 4. Deadzone
        if self.prev_output is not None:
            dx, dy = sx - self.prev_output[0], sy - self.prev_output[1]
            if math.sqrt(dx * dx + dy * dy) < DEADZONE_PX:
                return self.prev_output
        self.prev_output = (sx, sy)
        return sx, sy


# ================================================================
#  IQR
# ================================================================
def reject_outliers_iqr(samples):
    if len(samples) < 5:
        return np.array(samples)
    arr = np.array(samples)
    mask = np.ones(len(arr), dtype=bool)
    for d in range(arr.shape[1]):
        q1, q3 = np.percentile(arr[:, d], 25), np.percentile(arr[:, d], 75)
        iqr = q3 - q1
        mask &= (arr[:, d] >= q1 - IQR_FACTOR*iqr) & (arr[:, d] <= q3 + IQR_FACTOR*iqr)
    c = arr[mask]
    return c if len(c) >= 3 else arr


# ================================================================
#  16-POINT CALIBRATION
# ================================================================
def run_calibration(cap, tracker):
    points = [
        (0.08, 0.08), (0.37, 0.08), (0.63, 0.08), (0.92, 0.08),
        (0.08, 0.33), (0.37, 0.33), (0.63, 0.33), (0.92, 0.33),
        (0.08, 0.67), (0.37, 0.67), (0.63, 0.67), (0.92, 0.67),
        (0.08, 0.92), (0.37, 0.92), (0.63, 0.92), (0.92, 0.92),
    ]
    NUM_PTS = len(points)

    print("\n" + "=" * 70)
    print("         NEURO-CURSOR v8.0 CALIBRATION")
    print("=" * 70)
    print("  * Look at center of each dot")
    print("  * Keep head STILL")
    print("  * 16 points for maximum accuracy")
    print("=" * 70)
    input("\nPress ENTER to begin...")

    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    raw_pts, scr_pts = [], []
    nose_positions = []

    for idx, (gx, gy) in enumerate(points):
        sx, sy = int(gx * SCREEN_W), int(gy * SCREEN_H)
        samples = []
        t0 = time.time()
        collecting = False

        while len(samples) < MIN_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            gaze, _, _, _ = tracker.process_frame(frame)
            elapsed = time.time() - t0

            if elapsed > CAL_SETTLE_SEC and not collecting:
                collecting = True
            if collecting and gaze is not None:
                samples.append(gaze.copy())
                # Record nose position for reference
                if tracker.last_lm:
                    nose_positions.append(tracker._pt(tracker.last_lm, NOSE_TIP).copy())

            bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.putText(bg, "Look at each dot - keep head still", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            if collecting and len(samples) > 0:
                prog = len(samples) / MIN_SAMPLES
                cv2.ellipse(bg, (sx, sy), (45, 45), -90, 0, int(360*prog),
                            (0, 255, 0), 7)
                cv2.circle(bg, (sx, sy), 25, (0, 255, 0), -1)
                cv2.putText(bg, f"{len(samples)}/{MIN_SAMPLES}",
                            (sx-25, sy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            else:
                cv2.circle(bg, (sx, sy), 25, (0, 255, 255), -1)
            cv2.circle(bg, (sx, sy), 30, (255, 255, 255), 2)
            cv2.circle(bg, (sx, sy), 4, (255, 255, 255), -1)
            cv2.putText(bg, f"Point {idx+1}/{NUM_PTS}",
                        (SCREEN_W//2-70, SCREEN_H-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.imshow(win, bg)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(win)
                return False
            if elapsed > 8:
                break

        if len(samples) >= MIN_SAMPLES // 3:
            cleaned = reject_outliers_iqr(samples)
            std = np.std(cleaned, axis=0).mean()
            if std > CONSISTENCY_THRESH:
                print(f"  [{idx+1}/{NUM_PTS}] UNSTEADY (std={std:.4f}) — skipped")
            else:
                med = np.median(cleaned, axis=0)
                raw_pts.append(tuple(med))
                scr_pts.append((sx, sy))
                print(f"  [{idx+1}/{NUM_PTS}] OK ({len(samples)} samples, std={std:.4f})")
        else:
            print(f"  [{idx+1}/{NUM_PTS}] FAIL")
        time.sleep(0.12)

    cv2.destroyWindow(win)

    # Set nose reference as average during calibration
    if nose_positions:
        tracker.nose_ref = np.mean(nose_positions, axis=0)
        print(f"[CAL] Nose reference set: {tracker.nose_ref}")

    if len(raw_pts) < 8:
        print(f"\n[CAL] FAILED — only {len(raw_pts)}/{NUM_PTS} accepted")
        return False

    # ---- PHASE 2: Smooth pursuit (moving dots center → corners) ----
    pursuit_paths = [
        ((0.5, 0.5), (0.08, 0.08)),
        ((0.5, 0.5), (0.92, 0.08)),
        ((0.5, 0.5), (0.08, 0.92)),
        ((0.5, 0.5), (0.92, 0.92)),
    ]
    PURSUIT_DUR = 3.0
    PURSUIT_SETTLE = 0.5

    print("\n  [Phase 2] Follow the moving dot with your eyes")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for pidx, (start, end) in enumerate(pursuit_paths):
        t0 = time.time()
        pg, ps = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gaze, _, _, _ = tracker.process_frame(frame)
            elapsed = time.time() - t0
            if elapsed > PURSUIT_SETTLE + PURSUIT_DUR:
                break
            tf = 0.0 if elapsed < PURSUIT_SETTLE else \
                min((elapsed - PURSUIT_SETTLE) / PURSUIT_DUR, 1.0)
            dx = start[0] + tf * (end[0] - start[0])
            dy = start[1] + tf * (end[1] - start[1])
            dsx, dsy = int(dx * SCREEN_W), int(dy * SCREEN_H)
            if 0.1 < tf < 0.9 and elapsed > PURSUIT_SETTLE and gaze is not None:
                pg.append(gaze.copy())
                ps.append((dsx, dsy))
            bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.putText(bg, f"Follow the dot ({pidx+1}/4)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            for tt in np.linspace(max(0, tf-0.12), tf, 4):
                tx = start[0] + max(0, tt) * (end[0] - start[0])
                ty = start[1] + max(0, tt) * (end[1] - start[1])
                cv2.circle(bg, (int(tx*SCREEN_W), int(ty*SCREEN_H)), 8, (0,80,0), -1)
            cv2.circle(bg, (dsx, dsy), 22, (0, 255, 100), -1)
            cv2.circle(bg, (dsx, dsy), 27, (255, 255, 255), 2)
            cv2.circle(bg, (dsx, dsy), 4, (255, 255, 255), -1)
            cv2.imshow(win, bg)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(win)
                return False
        for i in range(0, len(pg), 10):
            raw_pts.append(tuple(pg[i]))
            scr_pts.append(ps[i])
        print(f"  [P{pidx+1}] {len(pg)} samples → {len(pg)//5} used")
        time.sleep(0.15)

    cv2.destroyWindow(win)
    print(f"\n[CAL] Total: {len(raw_pts)} data points")

    tracker.calibrate(raw_pts, scr_pts)
    return _run_verification(cap, tracker)


# ================================================================
#  VERIFICATION
# ================================================================
def _run_verification(cap, tracker):
    test_pts = [(0.20, 0.20), (0.80, 0.20), (0.50, 0.50),
                (0.20, 0.80), (0.80, 0.80)]
    win = "Verification"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    errors = []
    for i, (gx, gy) in enumerate(test_pts):
        sx, sy = int(gx * SCREEN_W), int(gy * SCREEN_H)
        samples = []
        t0 = time.time()
        while len(samples) < 25:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gaze, _, _, _ = tracker.process_frame(frame)
            if time.time() - t0 > 1.0 and gaze is not None:
                m = tracker.map_to_screen(gaze)
                if m:
                    # Pure gaze accuracy test — do not add head offset here!
                    # Adding head offset here inflates the error artificially.
                    samples.append(m)
            bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(bg, (sx, sy), 25, (0, 200, 255), -1)
            cv2.circle(bg, (sx, sy), 30, (255, 255, 255), 3)
            cv2.putText(bg, f"Verify {i+1}/5 - look at the dot", (SCREEN_W//2-200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow(win, bg)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(win)
                return False
            if time.time() - t0 > 5:
                break
        if len(samples) >= 10:
            med = np.median(np.array(samples), axis=0)
            err = math.sqrt((med[0]-sx)**2 + (med[1]-sy)**2)
            errors.append(err)
            print(f"  [V{i+1}] error = {err:.0f} px")
        time.sleep(0.15)
    if errors:
        avg = sum(errors) / len(errors)
        bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(bg, f"Accuracy: {avg:.0f} px",
                    (SCREEN_W//2-250, SCREEN_H//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(bg, "ENTER = accept, ESC = recalibrate",
                    (SCREEN_W//2-250, SCREEN_H//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.imshow(win, bg)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                cv2.destroyWindow(win)
                print(f"[CAL] Accuracy: {avg:.0f} px")
                return True
            if key == 27:
                cv2.destroyWindow(win)
                return False
    cv2.destroyWindow(win)
    return True


# ================================================================
#  MAIN
# ================================================================
def main():
    print("\n" + "=" * 70)
    print("         NEURO-CURSOR v8.1")
    print("    Fully Accurate Eye-Controlled Mouse")
    print("=" * 70)
    print("  L-Wink=Left Click | R-Wink=Right Click | Dbl-Blink=Dbl Click")
    print("  Look at top/bottom edge = Scroll | Slight head turn = Coarse move")
    print("=" * 70 + "\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # Enable auto-exposure for low-light
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)     # Boost brightness for dim rooms

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        input("Press ENTER to exit...")
        return

    tracker = NeuroCursor()

    while True:
        if run_calibration(cap, tracker):
            break
        print("\n[INFO] Retrying calibration...\n")

    print("\n[TRACKING] Active!")
    print("  Left wink  → Left Click")
    print("  Right wink → Right Click")
    print("  Double blink → Double Click")
    print("  Look at screen edges → Scroll")
    print("  'p' = Pause/Resume | 'c' = Recalibrate | 'n' = Toggle Snap | 'q' = Quit")
    print("  Close both eyes 2s = Pause/Resume (hands-free)\n")

    snap_engine = SnapEngine() if HAS_UIAUTOMATION else None
    if snap_engine:
        print("[SNAP] Snap-to-element ACTIVE — press 'n' to toggle")

    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Tracking", 480, 360)
    cv2.namedWindow("Eye Detail", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Detail", 500, 150)

    fps_t = time.time()
    fps_count = 0
    fps_display = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)

            gaze, click_type, is_action, eye_display = tracker.process_frame(frame)

            # ---- CURSOR MOVEMENT ----
            # COMPLETELY FROZEN during any eye action (wink/blink) or pause
            if tracker.paused:
                # Skip all cursor/click/scroll when paused
                pass
            elif not is_action and gaze is not None:
                mapped = tracker.map_to_screen(gaze)
                if mapped is not None:
                    fx, fy = mapped

                    # Blend head movement for precision assist (30% head, 70% gaze)
                    if tracker.last_lm is not None:
                        hdx, hdy = tracker.compute_head_delta(tracker.last_lm)
                        head_x = fx + hdx * HEAD_SENS_X
                        head_y = fy + hdy * HEAD_SENS_Y
                        fx = fx * (1 - HEAD_WEIGHT) + head_x * HEAD_WEIGHT
                        fy = fy * (1 - HEAD_WEIGHT) + head_y * HEAD_WEIGHT

                    fx, fy = tracker.apply_filtering(fx, fy)
                    fx = max(0, min(SCREEN_W - 1, fx))
                    fy = max(0, min(SCREEN_H - 1, fy))

                    # Snap-to-element: lock onto nearby UI elements
                    if snap_engine and snap_engine.enabled:
                        snap = snap_engine.query(fx, fy)
                        if snap:
                            fx, fy = snap

                    tracker.last_cx = fx
                    tracker.last_cy = fy
                    try:
                        pyautogui.moveTo(int(fx), int(fy), _pause=False)
                    except Exception:
                        pass

            # ---- CLICKS (skip if paused) ----
            if tracker.paused:
                click_type = None
            if click_type == 'left':
                try:
                    pyautogui.click(_pause=False)
                    print("[ACTION] Left Click")
                except Exception:
                    pass
            elif click_type == 'right':
                try:
                    pyautogui.rightClick(_pause=False)
                    print("[ACTION] Right Click")
                except Exception:
                    pass
            elif click_type == 'double':
                try:
                    pyautogui.doubleClick(_pause=False)
                    print("[ACTION] Double Click")
                except Exception:
                    pass

            # ---- SCROLL ----
            scroll = 0 if tracker.paused else tracker.check_scroll_zone(tracker.last_cy)
            if scroll != 0:
                try:
                    pyautogui.scroll(scroll, _pause=False)
                except Exception:
                    pass

            # ---- FPS ----
            fps_count += 1
            if time.time() - fps_t >= 1.0:
                fps_display = fps_count
                fps_count = 0
                fps_t = time.time()

            # ---- DRAW ----
            if tracker.last_lm is not None:
                h, w = frame.shape[:2]
                lm = tracker.last_lm
                for idx in L_EYE_CONTOUR + R_EYE_CONTOUR:
                    cv2.circle(frame, tracker._px(lm, idx, w, h), 1, (0,255,0), -1)
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    cv2.circle(frame, tracker._px(lm, idx, w, h), 2, (255,255,0), -1)
                for idx in [LEFT_PUPIL, RIGHT_PUPIL]:
                    cv2.circle(frame, tracker._px(lm, idx, w, h), 3, (0,0,255), -1)

            # Status
            if tracker.paused:
                state_str = "PAUSED (p or 2s blink)"
                state_col = (0, 140, 255)
            elif is_action:
                state_str = "WINK/BLINK"
                state_col = (0, 0, 255)
            elif tracker.is_scrolling:
                d = "UP" if tracker.scroll_zone == 'up' else "DOWN"
                state_str = f"SCROLL {d}"
                state_col = (0, 255, 255)
            elif snap_engine and snap_engine.enabled and snap_engine.snap_pos:
                state_str = "LOCKED ON"
                state_col = (255, 200, 0)
            else:
                state_str = "TRACKING"
                state_col = (0, 255, 0)

            cv2.putText(frame, state_str, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_col, 2)
            cv2.putText(frame, f"{fps_display} fps", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # Action feedback
            now = time.time()
            if now - tracker.action_time < 1.0:
                cv2.putText(frame, tracker.action_text, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Guide
            cv2.putText(frame, "L-Wink=L | R-Wink=R | Dbl-Blink=Dbl",
                        (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

            cv2.imshow("Eye Tracking", frame)
            if eye_display is not None:
                cv2.imshow("Eye Detail", eye_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                print("\n[RECAL] Starting...")
                if run_calibration(cap, tracker):
                    print("[RECAL] Done\n")
                else:
                    print("[RECAL] Failed\n")
            elif key == ord("n") and snap_engine:
                snap_engine.enabled = not snap_engine.enabled
                s = "ON" if snap_engine.enabled else "OFF"
                print(f"[SNAP] Snap-to-element: {s}")
            elif key == ord("p"):
                tracker.paused = not tracker.paused
                s = "PAUSED" if tracker.paused else "RESUMED"
                print(f"[PAUSE] Tracking {s}")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
        print("\n[SHUTDOWN] Complete")
        input("Press ENTER to exit...")


if __name__ == "__main__":
    main()