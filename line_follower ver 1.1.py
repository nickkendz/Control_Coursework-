import cv2
import numpy as np
import serial
import threading
import time
from collections import deque
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  TUNING
# ══════════════════════════════════════════════════════════════

# ── BLACK DETECTION (absolute HSV) ───────────────────────────
#
#  Genuine black tape:          V ≈ 20–55,  S ≈ 0–25
#  Robot shadow on white floor: V ≈ 60–120, S ≈ 0–20
#  Carpet (brown/warm):         V ≈ 80–140, S ≈ 35–90
#
#  KEY CHANGE from previous version:
#    V_HARD_MAX is now 58 (was 72).
#    Camera shadows cast by the robot chassis land at V ≈ 60–100.
#    Dropping the ceiling to 58 cleanly rejects them while tape
#    (V ≈ 20–55) is still detected.  If your tape reads above 55
#    under your lighting, raise this by 3–4 at a time.
#
#  Additional shadow guard — edge sharpness:
#    Black tape has a hard, crisp edge against a light floor.
#    The robot's own shadow fades gradually (soft gradient).
#    SHADOW_EDGE_MIN_GRAD is the minimum mean Sobel magnitude
#    measured at the blob boundary.  Blobs with soft edges are
#    classified as shadows and ignored.
#    Raise toward 18 if shadows still slip through.
#    Lower toward 6  if tape is missed in dim lighting.
#
V_RATIO             = 0.42   # tape darker than 42% of floor brightness
V_HARD_MAX          = 80     # raised: tape can read V≈55-70 under bright light
V_HARD_MIN          = 18     # lowered: allow detection in very bright scenes
S_MAX_BLACK         = 30     # achromatic ceiling; carpet/warm shadow > 35
SHADOW_EDGE_MIN_GRAD = 6.0   # lowered: real tape edges comfortably exceed 6

# ── ROI ──────────────────────────────────────────────────────
ROI_TOP_FRAC    = 0.20
ROI_BOTTOM_FRAC = 0.75

# ── SHADOW EXCLUSION STRIP (bottom of ROI) ───────────────────
SHADOW_STRIP = 0.25

# ── DETECTION QUALITY THRESHOLDS ─────────────────────────────
OFF_TRACK_RATIO     = 0.18
MIN_LINE_RATIO      = 0.004
TAPE_MAX_WIDTH_FRAC = 0.45
TAPE_MIN_WIDTH_PX   = 3

# ── WIDTH LEARNING ────────────────────────────────────────────
#
#  The physical tape width is constant.  Once the robot has seen
#  the tape a few times it learns the expected pixel-width and
#  rejects detections that deviate too much.  This filters out:
#    • robot wires (very narrow, 1–3 px)
#    • wide shadow blobs that pass the V/S filters
#
#  TAPE_WIDTH_LEARN_SIZE : rolling window of recent valid widths
#  TAPE_WIDTH_TOLERANCE  : accept ± this fraction of the learned
#                          median width.  0.45 = ±45 %.
#  TAPE_WIDTH_MIN_SAMPLES: do not apply the learned constraint
#                          until this many samples are collected
#                          (avoids rejecting everything on startup)
#
TAPE_WIDTH_LEARN_SIZE   = 30    # NEW
TAPE_WIDTH_TOLERANCE    = 0.45  # NEW  (±45 % of median)
TAPE_WIDTH_MIN_SAMPLES  = 8     # NEW  (apply after 8 good readings)

# ── FINISH LINE DETECTION ─────────────────────────────────────
#
#  A finish line is a strip of tape perpendicular to the robot's
#  direction of travel.  In the camera frame it appears as a
#  near-horizontal band of dark pixels spanning most of the
#  frame width, rather than a narrow vertical stripe.
#
#  Detection method:
#    For each row in the usable ROI, count what fraction of the
#    row's pixels are white in the clean mask.
#    If FINISH_MIN_ROWS rows exceed FINISH_ROW_COVERAGE,
#    the robot has reached the finish line.
#
#  FINISH_ROW_COVERAGE : fraction of row width that must be dark
#                        (0.35 = 35 % of 320 px ≈ 112 px wide band)
#  FINISH_MIN_ROWS     : consecutive/total rows that meet the above
#
FINISH_ROW_COVERAGE = 0.35   # NEW
FINISH_MIN_ROWS     = 6      # NEW

# ── MORPHOLOGY ───────────────────────────────────────────────
MORPH_OPEN_K  = 3
MORPH_CLOSE_K = 9

# ── SPEEDS (encoder ticks / 100ms) ───────────────────────────
CRAWL_TICKS   = 45
TURN_TICKS    = 35
ROTATE_TICKS  = 35
REVERSE_TICKS = 35
MANUAL_TICKS  = 45

# ── OFF-TRACK RECOVERY ────────────────────────────────────────
REVERSE_TIME   = 1.2
RECOVER_PAUSE  = 0.5

# ── SEARCH BEHAVIOUR ─────────────────────────────────────────
SEARCH_FLIP_TIME = 2.5

# ── STEERING SMOOTHING ────────────────────────────────────────
ERROR_SMOOTH = 0.70

# ── CAMERA ───────────────────────────────────────────────────
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240

# ══════════════════════════════════════════════════════════════
#  SERIAL PROTOCOL  "<mode>,<ticks_l>,<ticks_r>\n"
#  0=STOP  1=FORWARD  2=ROTATE_LEFT  3=ROTATE_RIGHT  4=BACKWARD
# ══════════════════════════════════════════════════════════════

output_frame   = None
lock           = threading.Lock()

last_error       = 0
last_line_time   = 0.0
search_direction = 1
search_flip_ts   = 0.0
smoothed_error   = 0.0

# FSM states: SEARCHING | TRACKING | OFF_TRACK | RECOVERING | FINISHED
auto_state    = "SEARCHING"
state_enter_t = 0.0

robot_mode = "auto"
mode_lock  = threading.Lock()

keys_held = {"w": False, "s": False, "a": False, "d": False}
keys_lock = threading.Lock()

# ── Width-learning ring buffer ────────────────────────────────
tape_width_history = deque(maxlen=TAPE_WIDTH_LEARN_SIZE)   # NEW

try:
    arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)
    time.sleep(2)
    print("[INFO] Arduino connected on /dev/ttyUSB0")
except Exception as e:
    arduino = None
    print(f"[WARN] Arduino not found: {e}")


def send(mode, ticks_l=0, ticks_r=0):
    ticks_l = int(max(0, ticks_l))
    ticks_r = int(max(0, ticks_r))
    if arduino is not None:
        arduino.write(f"{mode},{ticks_l},{ticks_r}\n".encode())


def keys_to_command():
    with keys_lock:
        w = keys_held["w"]; s = keys_held["s"]
        a = keys_held["a"]; d = keys_held["d"]
    if w and not s:
        if a and not d: return (1, int(MANUAL_TICKS * 0.45), MANUAL_TICKS)
        if d and not a: return (1, MANUAL_TICKS, int(MANUAL_TICKS * 0.45))
        return (1, MANUAL_TICKS, MANUAL_TICKS)
    if s and not w:
        if a and not d: return (4, MANUAL_TICKS, int(MANUAL_TICKS * 0.45))
        if d and not a: return (4, int(MANUAL_TICKS * 0.45), MANUAL_TICKS)
        return (4, MANUAL_TICKS, MANUAL_TICKS)
    if a and not d: return (2, MANUAL_TICKS, MANUAL_TICKS)
    if d and not a: return (3, MANUAL_TICKS, MANUAL_TICKS)
    return (0, 0, 0)


# ══════════════════════════════════════════════════════════════
#  HELPER – edge sharpness check
# ══════════════════════════════════════════════════════════════
def blob_edge_sharpness(v_channel, mask, left, right):
    """
    Return mean Sobel gradient magnitude at the left & right edges
    of the detected tape blob.

    v_channel : 2D array (usable_h × w), dtype uint8  — used for Sobel
    mask      : 2D binary mask (usable_h × w) — the actual clean detection
                mask; used to locate the blob boundary.

    BUG FIX: previous version used (v_channel < 200) as the blob mask which
    captured almost all pixels so edge_mask was always ~0, causing the
    function to always return 0.0 and reject everything as a shadow.
    """
    l = max(0, left - 2)
    r = min(v_channel.shape[1] - 1, right + 2)
    region = v_channel[:, l:r + 1].astype(np.float32)

    sobel_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    magnitude = np.abs(sobel_x)

    # Use the actual clean detection mask for the blob boundary
    blob_strip = mask[:, l:r + 1]
    dilated    = cv2.dilate(blob_strip, np.ones((3, 3), np.uint8))
    edge_mask  = cv2.subtract(dilated, blob_strip)

    n = np.sum(edge_mask > 0)
    if n == 0:
        return 0.0
    return float(np.sum(magnitude[edge_mask > 0])) / n


# ══════════════════════════════════════════════════════════════
#  CAMERA + DECISION LOOP
# ══════════════════════════════════════════════════════════════
def process_camera():
    global output_frame, last_error, last_line_time
    global search_direction, search_flip_ts, smoothed_error
    global auto_state, state_enter_t, tape_width_history

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        cx_frame = w // 2

        # ── 1. ROI ────────────────────────────────────────────
        roi_top     = int(h * ROI_TOP_FRAC)
        roi_bottom  = int(h * ROI_BOTTOM_FRAC)
        roi_h       = roi_bottom - roi_top
        strip_start = int(roi_h * (1.0 - SHADOW_STRIP))
        usable_h    = strip_start
        usable_area = w * usable_h

        roi = frame[roi_top:roi_bottom, :]

        # ── 2. HSV + blur ─────────────────────────────────────
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        v_ch = blurred[:usable_h, :, 2]
        s_ch = blurred[:usable_h, :, 1]

        # ── 3. Adaptive V threshold ───────────────────────────
        floor_v80 = float(np.percentile(v_ch, 80))
        v_thresh  = int(floor_v80 * V_RATIO)
        v_thresh  = max(V_HARD_MIN, min(v_thresh, V_HARD_MAX))

        # ── 4. Black mask (V + S filters) ─────────────────────
        v_mask     = cv2.inRange(v_ch, 0, v_thresh)
        s_mask     = cv2.inRange(s_ch, 0, S_MAX_BLACK)
        black_mask = cv2.bitwise_and(v_mask, s_mask)

        k_o   = np.ones((MORPH_OPEN_K,  MORPH_OPEN_K),  np.uint8)
        k_c   = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
        clean = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,  k_o)
        clean = cv2.morphologyEx(clean,      cv2.MORPH_CLOSE, k_c)

        thresh = np.zeros((roi_h, w), dtype=np.uint8)
        thresh[:usable_h, :] = clean

        # ── 5. Density classification ─────────────────────────
        white_px    = int(np.sum(clean > 0))
        white_ratio = white_px / max(usable_area, 1)

        is_off_track   = (white_ratio > OFF_TRACK_RATIO)
        has_candidates = (not is_off_track) and (white_ratio >= MIN_LINE_RATIO)

        # ── 5b. FINISH LINE detection ──────────────────────────
        #
        #  A perpendicular (finish) line appears as a wide horizontal
        #  dark band.  We count how many rows have ≥ FINISH_ROW_COVERAGE
        #  of their width occupied by dark pixels.
        #  If enough such rows exist, declare FINISHED.
        #
        #  This check runs on the raw clean mask so it is independent
        #  of the narrow-line peak detection below.
        #
        finish_line_detected = False
        if has_candidates:
            row_coverage   = np.sum(clean > 0, axis=1) / w   # shape (usable_h,)
            wide_rows      = int(np.sum(row_coverage > FINISH_ROW_COVERAGE))
            finish_line_detected = (wide_rows >= FINISH_MIN_ROWS)

        # ── 6. Column histogram + peak + width filter ─────────
        line_found    = False
        current_error = 0
        cx_line       = cx_frame
        tape_width_px = 0
        col_smooth    = np.zeros(w, dtype=np.float32)
        edge_sharpness = 0.0
        reject_reason  = "NONE"

        if has_candidates and not finish_line_detected:
            col_sum    = np.sum(clean, axis=0).astype(np.float32)
            col_smooth = cv2.GaussianBlur(
                col_sum.reshape(1, -1), (1, 21), 0
            ).flatten()

            peak_col = int(np.argmax(col_smooth))
            peak_val = float(col_smooth[peak_col])

            if peak_val > 0:
                half  = peak_val / 2.0
                left  = peak_col
                right = peak_col
                while left > 0 and col_smooth[left] > half:
                    left -= 1
                while right < w - 1 and col_smooth[right] > half:
                    right += 1
                tape_width_px = right - left
                max_w = int(w * TAPE_MAX_WIDTH_FRAC)

                # ── Fixed-range width gate ─────────────────────
                width_ok = TAPE_MIN_WIDTH_PX <= tape_width_px <= max_w

                # ── Learned width gate (NEW) ───────────────────
                #
                #  Once enough samples are collected, the robot knows
                #  the expected pixel-width of its specific tape.
                #  Reject anything outside ±TAPE_WIDTH_TOLERANCE of
                #  the rolling median — catches wires (too narrow)
                #  and partial shadows (too wide).
                #
                learned_ok = True
                if width_ok and len(tape_width_history) >= TAPE_WIDTH_MIN_SAMPLES:
                    median_w   = float(np.median(tape_width_history))
                    lo = median_w * (1.0 - TAPE_WIDTH_TOLERANCE)
                    hi = median_w * (1.0 + TAPE_WIDTH_TOLERANCE)
                    if not (lo <= tape_width_px <= hi):
                        learned_ok   = False
                        reject_reason = f"WIDTH({tape_width_px}px≠{int(median_w)}±{int(TAPE_WIDTH_TOLERANCE*100)}%)"

                # ── Edge sharpness gate (NEW) ──────────────────
                #
                #  Shadow blobs have soft, gradual edges.
                #  Black tape on a light floor has a sharp step.
                #  Reject any blob whose boundary Sobel magnitude
                #  is below SHADOW_EDGE_MIN_GRAD.
                #
                sharp_ok = True
                if width_ok and learned_ok:
                    edge_sharpness = blob_edge_sharpness(v_ch, clean, left, right)
                    if edge_sharpness < SHADOW_EDGE_MIN_GRAD:
                        sharp_ok      = False
                        reject_reason = f"SHADOW(grad={edge_sharpness:.1f})"

                if width_ok and learned_ok and sharp_ok:
                    cx_line        = peak_col
                    current_error  = cx_line - cx_frame
                    line_found     = True
                    last_error     = current_error
                    last_line_time = time.time()
                    tape_width_history.append(tape_width_px)   # update learner
                elif width_ok and not learned_ok:
                    pass   # reject_reason already set
                elif not width_ok:
                    reject_reason = f"WIDTH_OOB({tape_width_px}px)"

        # ── 7. Error smoothing ────────────────────────────────
        if line_found:
            smoothed_error = (ERROR_SMOOTH * smoothed_error
                              + (1.0 - ERROR_SMOOTH) * current_error)

        # ── 8. Mode dispatch ──────────────────────────────────
        with mode_lock:
            current_mode = robot_mode

        now    = time.time()
        status = ""

        # ══════════════════════════════════════════════════════
        #  MANUAL MODE
        # ══════════════════════════════════════════════════════
        if current_mode == "manual":
            cmd = keys_to_command()
            send(*cmd)
            status = "MANUAL"
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 60, 180), -1)
            cv2.putText(frame, "[ MANUAL CONTROL ]",
                        (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)
            with keys_lock:
                held = [k.upper() for k, v in keys_held.items() if v]
            cv2.putText(frame, "Keys: " + (" ".join(held) if held else "none"),
                        (6, 46), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 220, 255), 1)

        # ══════════════════════════════════════════════════════
        #  AUTO MODE  —  5-state FSM
        #
        #   TRACKING   : line found → steer forward
        #   SEARCHING  : no line   → rotate in-place
        #   OFF_TRACK  : too much noise → reverse
        #   RECOVERING : pause after reverse → then search
        #   FINISHED   : perpendicular line seen → stop forever
        #               (reset by switching to manual then back to auto)
        # ══════════════════════════════════════════════════════
        else:
            # ── Finish line overrides everything ──────────────
            if finish_line_detected and auto_state != "FINISHED":
                auto_state    = "FINISHED"
                state_enter_t = now
                send(0)

            # ── Off-track override (not during FINISHED) ──────
            elif (is_off_track
                  and auto_state not in ("OFF_TRACK", "RECOVERING", "FINISHED")):
                auto_state    = "OFF_TRACK"
                state_enter_t = now
                send(0)

            # ── State: FINISHED ────────────────────────────────
            if auto_state == "FINISHED":
                send(0)
                status = "★ FINISH LINE — STOPPED ★"

            # ── State: TRACKING ────────────────────────────────
            elif auto_state == "TRACKING":
                if line_found:
                    severity = min(abs(smoothed_error) / float(cx_frame), 1.0)
                    inner    = max(int(TURN_TICKS * (1.0 - severity * 0.35)), 25)
                    outer    = CRAWL_TICKS
                    if smoothed_error > 0:
                        tl, tr = outer, inner
                    elif smoothed_error < 0:
                        tl, tr = inner, outer
                    else:
                        tl = tr = CRAWL_TICKS
                    send(1, tl, tr)
                    n_samples = len(tape_width_history)
                    status = (f"TRACKING err:{int(smoothed_error):+d} "
                              f"w:{tape_width_px}px "
                              f"[{n_samples}/{TAPE_WIDTH_LEARN_SIZE}lrn]")
                else:
                    auto_state    = "SEARCHING"
                    state_enter_t = now
                    status = "TRACKING → SEARCHING"

            # ── State: SEARCHING ───────────────────────────────
            elif auto_state == "SEARCHING":
                if line_found:
                    auto_state    = "TRACKING"
                    state_enter_t = now
                    status = "SEARCHING → TRACKING"
                else:
                    if last_error > 0:   search_direction = 1
                    elif last_error < 0: search_direction = -1
                    if now - search_flip_ts > SEARCH_FLIP_TIME:
                        search_direction *= -1
                        search_flip_ts    = now
                    if search_direction == 1:
                        send(3, ROTATE_TICKS, ROTATE_TICKS)
                        status = "SEARCHING → spin RIGHT"
                    else:
                        send(2, ROTATE_TICKS, ROTATE_TICKS)
                        status = "SEARCHING → spin LEFT"

            # ── State: OFF_TRACK ───────────────────────────────
            elif auto_state == "OFF_TRACK":
                elapsed   = now - state_enter_t
                remaining = max(0.0, REVERSE_TIME - elapsed)
                if elapsed < REVERSE_TIME:
                    send(4, REVERSE_TICKS, REVERSE_TICKS)
                    status = f"!! OFF TRACK !! reversing {remaining:.1f}s"
                else:
                    send(0)
                    auto_state    = "RECOVERING"
                    state_enter_t = now
                    status = "OFF_TRACK → RECOVERING"

            # ── State: RECOVERING ──────────────────────────────
            elif auto_state == "RECOVERING":
                send(0)
                elapsed = now - state_enter_t
                if elapsed > RECOVER_PAUSE:
                    auto_state     = "SEARCHING"
                    state_enter_t  = now
                    smoothed_error = 0.0
                    status = "RECOVERING → SEARCHING"
                else:
                    status = f"RECOVERING {RECOVER_PAUSE - elapsed:.1f}s"

            # ── Auto HUD banner ────────────────────────────────
            state_colours = {
                "TRACKING":   (0, 130, 0),
                "SEARCHING":  (0, 100, 180),
                "OFF_TRACK":  (0, 0, 200),
                "RECOVERING": (0, 130, 130),
                "FINISHED":   (160, 0, 160),
            }
            banner_col = state_colours.get(auto_state, (60, 60, 60))
            cv2.rectangle(frame, (0, 0), (w, 28), banner_col, -1)
            cv2.putText(frame, f"[ {auto_state} ]",
                        (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)
            txt_col = (0, 255, 80) if line_found else (0, 180, 255)
            cv2.putText(frame, status,
                        (6, 46), cv2.FONT_HERSHEY_PLAIN, 0.9, txt_col, 1)

        # ── 9. Overlay ────────────────────────────────────────
        cv2.rectangle(frame, (0, roi_top), (w, roi_bottom), (0, 255, 0), 2)

        if finish_line_detected:
            # Draw a magenta horizontal bar to show finish detection
            bar_y = roi_top + usable_h // 2
            cv2.line(frame, (0, bar_y), (w, bar_y), (255, 0, 255), 3)
            cv2.putText(frame, "FINISH", (w // 2 - 30, bar_y - 6),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 2)

        if line_found:
            cv2.line(frame, (cx_line, roi_bottom), (cx_line, roi_top),
                     (0, 255, 255), 2)
            cv2.circle(frame, (cx_line, roi_top + usable_h // 2), 7,
                       (0, 0, 255), -1)
            cv2.arrowedLine(frame,
                            (cx_frame, roi_bottom - 4),
                            (cx_line,  roi_bottom - 4),
                            (0, 255, 100), 2, tipLength=0.3)

        # ── 10. Mask panel ────────────────────────────────────
        mask_bgr  = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        mask_full = np.zeros_like(frame)
        mask_full[roi_top:roi_bottom, :] = mask_bgr

        cv2.line(mask_full,
                 (0, roi_top + strip_start),
                 (w, roi_top + strip_start), (0, 0, 180), 1)

        if has_candidates and col_smooth.max() > 0:
            norm = col_smooth / col_smooth.max()
            for x in range(0, w, 2):
                bar_h = int(norm[x] * 35)
                if bar_h > 0:
                    cv2.line(mask_full,
                             (x, roi_top + strip_start - 1),
                             (x, roi_top + strip_start - 1 - bar_h),
                             (0, 180, 80), 1)

        cv2.putText(mask_full, "HSV BLACK MASK",
                    (4, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 200, 255), 2)
        cv2.putText(mask_full,
                    f"V<{v_thresh} S<{S_MAX_BLACK} den:{white_ratio:.3f}",
                    (4, 36), cv2.FONT_HERSHEY_PLAIN, 0.9, (160, 160, 255), 1)

        # Show detection result: LINE, FINISH, rejected reason, or NONE
        if finish_line_detected:
            det_str = "FINISH"
            det_col = (255, 0, 255)
        elif line_found:
            det_str = f"LINE w:{tape_width_px}px g:{edge_sharpness:.0f}"
            det_col = (0, 255, 80)
        elif is_off_track:
            det_str = "NOISE/CARPET"
            det_col = (0, 80, 255)
        else:
            det_str = reject_reason if reject_reason != "NONE" else "NONE"
            det_col = (80, 80, 255)

        cv2.putText(mask_full, det_str,
                    (4, 54), cv2.FONT_HERSHEY_PLAIN, 0.9, det_col, 1)

        # Show learned width range in mask panel
        if len(tape_width_history) >= TAPE_WIDTH_MIN_SAMPLES:
            med_w = float(np.median(tape_width_history))
            cv2.putText(mask_full,
                        f"lrn_w:{int(med_w)}±{int(TAPE_WIDTH_TOLERANCE*100)}%",
                        (4, 70), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 80), 1)

        with lock:
            output_frame = cv2.hconcat([frame, mask_full]).copy()

        time.sleep(0.04)


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.02)
                continue
            ok, buf = cv2.imencode('.jpg', output_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 72])
            if not ok:
                continue
            data = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.04)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/key', methods=['POST'])
def key_event():
    data  = request.get_json(force=True)
    key   = data.get("key", "").lower()
    state = data.get("state", "up")
    arrow_map = {"arrowup": "w", "arrowdown": "s",
                 "arrowleft": "a", "arrowright": "d"}
    key = arrow_map.get(key, key)
    if key in keys_held:
        with keys_lock:
            keys_held[key] = (state == "down")
    return jsonify(ok=True)


@app.route('/mode', methods=['POST'])
def set_mode():
    global robot_mode, smoothed_error, search_flip_ts, auto_state, state_enter_t
    global tape_width_history
    data     = request.get_json(force=True)
    new_mode = data.get("mode", "auto")
    with mode_lock:
        robot_mode = new_mode
    if new_mode == "auto":
        smoothed_error     = 0.0
        search_flip_ts     = time.time()
        auto_state         = "SEARCHING"
        state_enter_t      = time.time()
        # NOTE: tape_width_history is intentionally NOT cleared here.
        # The robot keeps its learned width across manual/auto switches.
        # To reset the learned width, restart the process.
        with keys_lock:
            for k in keys_held:
                keys_held[k] = False
        send(0)
    return jsonify(mode=new_mode)


@app.route('/status')
def status():
    with mode_lock:
        m = robot_mode
    with keys_lock:
        held = [k for k, v in keys_held.items() if v]
    learned_w = (float(np.median(tape_width_history))
                 if len(tape_width_history) >= TAPE_WIDTH_MIN_SAMPLES else None)
    return jsonify(mode=m, keys=held, auto_state=auto_state,
                   learned_width=learned_w,
                   width_samples=len(tape_width_history))


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f14; color: #e0e0e0;
    font-family: "Courier New", monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px; gap: 18px;
  }
  h1 { font-size: 1.2rem; letter-spacing: 0.15em; color: #7ecfff; text-transform: uppercase; }
  #feed-wrap { border: 2px solid #2a2a40; border-radius: 6px; overflow: hidden; box-shadow: 0 0 24px #00000080; }
  #feed-wrap img { display: block; width: 640px; max-width: 98vw; }
  #mode-row { display: flex; align-items: center; gap: 14px; }
  #mode-label { font-size: 0.85rem; color: #888; letter-spacing: 0.1em; }
  #mode-btn { padding: 8px 28px; font-family: inherit; font-size: 1rem; font-weight: bold;
    letter-spacing: 0.12em; border: none; border-radius: 4px; cursor: pointer;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s; }
  #mode-btn.auto   { background: #1a6b2e; color: #88ffaa; box-shadow: 0 0 10px #1a6b2e88; }
  #mode-btn.manual { background: #6b1a1a; color: #ffaaaa; box-shadow: 0 0 10px #6b1a1a88; }
  .key-grid { display: grid; grid-template-columns: repeat(3, 52px); grid-template-rows: repeat(2, 52px); gap: 6px; }
  .key { display: flex; align-items: center; justify-content: center;
    border: 2px solid #2a2a40; border-radius: 5px; font-size: 1.1rem; font-weight: bold;
    background: #1a1a24; color: #555; transition: background 0.08s, color 0.08s, border-color 0.08s; user-select: none; }
  .key.active { background: #1a3a6b; border-color: #4a9eff; color: #7ecfff; box-shadow: 0 0 8px #4a9eff66; }
  .key.w { grid-column: 2; grid-row: 1; } .key.a { grid-column: 1; grid-row: 2; }
  .key.s { grid-column: 2; grid-row: 2; } .key.d { grid-column: 3; grid-row: 2; }
  #info { font-size: 0.78rem; color: #556; text-align: center; line-height: 1.7; max-width: 640px; }
  #info span { color: #7ecfff; }
  #toast { position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%);
    background: #1a2a3a; border: 1px solid #4a9eff; color: #7ecfff;
    padding: 8px 22px; border-radius: 20px; font-size: 0.9rem;
    opacity: 0; transition: opacity 0.3s; pointer-events: none; }
</style>
</head>
<body>
<h1>&#x1F916; Robot Control Panel</h1>
<div id="feed-wrap"><img src="/video_feed" alt="Camera feed"></div>
<div id="mode-row">
  <span id="mode-label">MODE:</span>
  <button id="mode-btn" class="auto" onclick="toggleMode()">AUTO</button>
</div>
<div class="key-grid">
  <div class="key w" id="key-w">W &#x2191;</div>
  <div class="key a" id="key-a">&#x2190; A</div>
  <div class="key s" id="key-s">S &#x2193;</div>
  <div class="key d" id="key-d">D &#x2192;</div>
</div>
<div id="info">
  <b>Keyboard shortcuts</b><br>
  <span>W / &uarr;</span> Forward &nbsp;|&nbsp; <span>S / &darr;</span> Backward &nbsp;|&nbsp;
  <span>A / &larr;</span> Rotate left &nbsp;|&nbsp; <span>D / &rarr;</span> Rotate right<br>
  <span>M</span> Toggle AUTO / MANUAL &nbsp;|&nbsp; Click page first to capture keyboard focus.
</div>
<div id="toast"></div>
<script>
  let currentMode = "auto";
  function toggleMode() {
    const next = currentMode === "auto" ? "manual" : "auto";
    fetch("/mode", { method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({mode:next}) }).then(r=>r.json()).then(data=>{
      currentMode = data.mode; updateModeUI();
      showToast(currentMode==="auto" ? "Switched to AUTO mode" : "Switched to MANUAL — use WASD / arrows");
    });
  }
  function updateModeUI() {
    const btn = document.getElementById("mode-btn");
    btn.textContent = currentMode === "auto" ? "AUTO" : "MANUAL";
    btn.className   = currentMode === "auto" ? "auto" : "manual";
  }
  const KEY_MAP = {"w":"w","arrowup":"w","s":"s","arrowdown":"s","a":"a","arrowleft":"a","d":"d","arrowright":"d"};
  function sendKey(key, state) {
    fetch("/key", { method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({key, state}) });
  }
  function highlightKey(k, active) {
    const el = document.getElementById("key-"+k);
    if (el) el.classList.toggle("active", active);
  }
  document.addEventListener("keydown", e => {
    if (e.key.toLowerCase()==="m") { toggleMode(); return; }
    if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.key)) e.preventDefault();
    const k = KEY_MAP[e.key.toLowerCase()];
    if (!k || e.repeat) return;
    if (currentMode !== "manual") return;
    highlightKey(k, true); sendKey(k, "down");
  });
  document.addEventListener("keyup", e => {
    const k = KEY_MAP[e.key.toLowerCase()];
    if (!k) return;
    highlightKey(k, false); sendKey(k, "up");
  });
  window.addEventListener("blur", () => {
    ["w","a","s","d"].forEach(k => { highlightKey(k,false); sendKey(k,"up"); });
  });
  let toastTimer = null;
  function showToast(msg) {
    const t = document.getElementById("toast");
    t.textContent = msg; t.style.opacity = "1";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.style.opacity="0", 2200);
  }
</script>
</body>
</html>'''


if __name__ == '__main__':
    threading.Thread(target=process_camera, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
