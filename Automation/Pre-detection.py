"""
Run command: python -m streamlit run Pre-detection.py
"""
import cv2
import numpy as np
import streamlit as st
from collections import deque
import time
import os


st.set_page_config(page_title="Automated Interface Detection", layout="wide")
st.title("Automated Liquid Interface Detection")

# =========================
# 1. Camera initialization
# =========================
@st.cache_resource
def get_camera():
    cap = cv2.VideoCapture(1)
    return cap

cap = get_camera()

# =========================
# 2. Hardcoded core parameters
# =========================
X_MIN, X_MAX = 302, 418
Y_MIN, Y_MAX = 320, 355
T1, T2 = 15, 35

# =========================
# 3. Main UI layout
# =========================
col1, col2 = st.columns([1.5, 1])
with col1:
    st.subheader("Live Tracking Feed")
    video_placeholder = st.empty()
    st.markdown("---")
with col2:
    st.subheader("Algorithm Vision (ROI)")
    canny_placeholder = st.empty()
    st.subheader("Detection Status")
    status_placeholder = st.empty()

def render_status(title, subtitle, bg_color, border_color, text_color):
    html = f"""
    <div style="background-color: {bg_color}; padding: 20px; border-left: 8px solid {border_color}; border-radius: 5px;">
        <h1 style="color: {text_color}; margin-top: 0; font-family: sans-serif; font-weight: 800;">{title}</h1>
        <h3 style="color: {text_color}; margin-bottom: 0; font-family: sans-serif; opacity: 0.9;">{subtitle}</h3>
    </div>
    """
    return html

# =========================
# 4. File-based signal setup
# =========================
SIGNAL_FILE = 'pre_detect_signal.txt'
if not os.path.exists(SIGNAL_FILE):
    with open(SIGNAL_FILE, 'w') as f:
        f.write("WAITING")

evaluating = False

# =========================
# 5. Core detection loop
# =========================
history_buffer = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    try:
        with open(SIGNAL_FILE, 'r') as f:
            cmd = f.read().strip()
    except Exception:
        cmd = "WAITING"

    if cmd == "CHECK_NOW" and not evaluating:
        evaluating = True
        history_buffer.clear()

    display_frame = frame.copy()

    H, W = frame.shape[:2]
    act_x1, act_x2 = max(0, min(X_MIN, W-1)), max(0, min(X_MAX, W))
    act_y1, act_y2 = max(0, min(Y_MIN, H-1)), max(0, min(Y_MAX, H))

    if act_x2 <= act_x1 or act_y2 <= act_y1:
        continue

    cv2.rectangle(display_frame, (act_x1, act_y1), (act_x2, act_y2), (0, 0, 150), 2)
    cv2.putText(display_frame, "ROI", (act_x1, act_y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)

    roi = frame[act_y1:act_y2, act_x1:act_x2]
    if roi.size == 0:
        continue

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred_roi, T1, T2)

    margin = 10
    edges[:, :margin] = 0
    edges[:, -margin:] = 0

    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_interface = False
    valid_lines = []
    roi_width = act_x2 - act_x1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > roi_width * 0.6:
            cnt_roi = edges_dilated[y:y+h, x:x+w]
            if np.count_nonzero(cnt_roi) > w * 0.8:
                pts = cnt[:, 0, :]
                pts_sorted = pts[np.argsort(pts[:, 0])]
                n_pts = len(pts_sorted)

                if n_pts > 10:
                    chunk = n_pts // 5
                    left_y = np.mean(pts_sorted[:chunk, 1])
                    right_y = np.mean(pts_sorted[-chunk:, 1])
                    mid_y = np.mean(pts_sorted[2*chunk:3*chunk, 1])

                    edge_y_avg = (left_y + right_y) / 2.0

                    if mid_y < edge_y_avg - 1:
                        valid_lines.append((x, y, w, h, w))

    if len(valid_lines) > 0:
        valid_lines.sort(key=lambda item: item[4], reverse=True)
        best_line = valid_lines[0]
        x, y, w, h, _ = best_line

        found_interface = True

        cv2.rectangle(display_frame, (act_x1 + x, act_y1 + y), (act_x1 + x + w, act_y1 + y + h), (0, 255, 0), 3)
        center_x, center_y = act_x1 + x + w// 2, act_y1 + y + h//2
        cv2.circle(display_frame, (center_x, center_y), 4, (0, 255, 0), -1)
        cv2.putText(display_frame, "LOCKED", (act_x1+x, act_y1+y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    history_buffer.append(1 if found_interface else 0)

    if len(history_buffer) == 10:
        confidence = sum(history_buffer)

        if evaluating:
            if confidence >= 4:
                with open(SIGNAL_FILE, 'w') as f:
                    f.write("RESULT_TRUE")
            else:
                with open(SIGNAL_FILE, 'w') as f:
                    f.write("RESULT_FALSE")
            evaluating = False

        if confidence >= 4:
            html = render_status("DETECTED", "Incompatible oil & surfactant detected.", "#d1e7dd", "#0f5132", "#0f5132")
            status_placeholder.markdown(html, unsafe_allow_html=True)
        elif confidence <= 3:
            html = render_status("UNDETECTED", "Continuous interface not found.", "#f8d7da", "#842029", "#842029")
            status_placeholder.markdown(html, unsafe_allow_html=True)
        else:
            html = render_status("PENDING", "Awaiting stable signal...", "#fff3cd", "#997404", "#856404")
            status_placeholder.markdown(html, unsafe_allow_html=True)
    else:
        if evaluating:
            html = render_status("ANALYZING", "Hardware requested check. Analyzing new frames...", "#fff3cd", "#997404", "#856404")
        else:
            html = render_status("STANDBY", "Camera running. Awaiting hardware trigger...", "#cff4fc", "#055160", "#055160")
        status_placeholder.markdown(html, unsafe_allow_html=True)

    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(display_frame_rgb, channels="RGB", width="stretch")
    canny_placeholder.image(edges, channels="GRAY", width="stretch")
