import cv2
import numpy as np
import mss
import win32gui
import math
import ctypes
import time
import os

# ==========================================
# 0. Windows DPI setup
# ==========================================
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# ==========================================
# 1. Target window and crop settings
# ==========================================
WINDOW_TITLE = "LDPlayer"

CROP_TOP = 40
CROP_BOTTOM = 0
CROP_LEFT = 0
CROP_RIGHT = 0

def get_window_rect(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd: return None
    rect = win32gui.GetWindowRect(hwnd)
    return {"left": rect[0] + 2, "top": rect[1] + 30, "width": rect[2] - rect[0] - 4, "height": rect[3] - rect[1] - 32}

monitor = get_window_rect(WINDOW_TITLE)
sct = mss.mss()

# ==========================================
# 2. Droplet scoring algorithm
# ==========================================
def run_cv_algorithm(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    roi_size = 450
    half_l = roi_size // 2
    y1, y2 = max(0, cy - half_l), min(h, cy + half_l)
    x1, x2 = max(0, cx - half_l), min(w, cx + half_l)

    roi_frame = frame[y1:y2, x1:x2].copy()
    display_frame = frame.copy()

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, binary = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    binary = cv2.dilate(binary, kernel_small, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    corrected_gray = gray.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    region_info = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 400:
            x_, y_, w_, h_ = stats[label, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
            mask_single = (labels == label).astype(np.uint8)
            mask_patch = mask_single[y_:y_+h_, x_:x_+w_]
            roi_gray_patch = gray[y_:y_+h_, x_:x_+w_].copy()

            white_threshold = 50
            white_mask = ((roi_gray_patch > white_threshold) & (mask_patch > 0)).astype(np.uint8)
            white_coords = np.column_stack(np.where(white_mask > 0))
            inside_black_region = True
            for wy, wx in white_coords:
                if mask_patch[wy, wx] == 0:
                    inside_black_region = False
                    break
            if inside_black_region and white_coords.shape[0] > 0:
                non_white_pixels = roi_gray_patch[(roi_gray_patch <= white_threshold) & (mask_patch > 0)]
                if non_white_pixels.size > 0:
                    mean_val = int(np.mean(non_white_pixels))
                    roi_gray_patch[white_mask > 0] = mean_val
                    corrected_gray[y_:y_+h_, x_:x_+w_][white_mask > 0] = mean_val

            roi_patch = cv2.GaussianBlur(roi_gray_patch, (7, 7), 0)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded_mask_patch = cv2.erode(mask_patch, kernel_erode, iterations=1)

            true_pixels = roi_patch[eroded_mask_patch > 0]
            if true_pixels.size == 0: continue
            true_mean_gray = np.mean(true_pixels)

            roi_patch_modified = roi_patch.copy()
            edge_mask = (mask_patch == 1) & (eroded_mask_patch == 0)
            roi_patch_modified[edge_mask] = true_mean_gray

            masked_pixels = roi_patch_modified[mask_patch > 0]
            region_gray_variance = np.var(masked_pixels) if masked_pixels.size > 0 else 1

            contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(max_contour)
                original_area = cv2.contourArea(max_contour)
                convex_hull_area = cv2.contourArea(hull)
                area_ratio = original_area / convex_hull_area if convex_hull_area > 0 else 0

                raw_reward = (area * area_ratio ** 5) / region_gray_variance if region_gray_variance > 0 else 0
                new_reward = math.sqrt(raw_reward)

                max_contour[:, 0, 0] += x1
                max_contour[:, 0, 1] += y1
                hull[:, 0, 0] += x1
                hull[:, 0, 1] += y1

                region_info.append({
                    'reward': new_reward, 'ratio': area_ratio, 'variance': region_gray_variance,
                    'area': area, 'contour': max_contour, 'hull': hull
                })

    if region_info:
        region_info.sort(key=lambda item: item['reward'], reverse=True)
        best = region_info[0]
        return display_frame, best
    return display_frame, None


# ==========================================
# 3. File-based command loop
# ==========================================
SIGNAL_FILE = 'chara_signal.txt'


if not os.path.exists(SIGNAL_FILE):
    with open(SIGNAL_FILE, 'w') as f:
        f.write("WAITING")

print(f"\n[{time.strftime('%H:%M:%S')}] AutoChara background mode started. Monitoring window [{WINDOW_TITLE}] ...")
print("Waiting for EVALUATE command from main program...")

while True:
    try:
        with open(SIGNAL_FILE, 'r') as f:
            cmd = f.read().strip()
    except Exception:
        cmd = "WAITING"

    if cmd == "EVALUATE":
        print(f"\n[{time.strftime('%H:%M:%S')}] Received evaluation command. Capturing screen for scoring...")
        with open(SIGNAL_FILE, 'w') as f:
            f.write("PROCESSING")

        try:

            monitor = get_window_rect(WINDOW_TITLE)

            if monitor:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                h, w = frame.shape[:2]
                if h > CROP_TOP + CROP_BOTTOM and w > CROP_LEFT + CROP_RIGHT:
                    frame = frame[CROP_TOP : h-CROP_BOTTOM, CROP_LEFT : w-CROP_RIGHT]

                    _, data = run_cv_algorithm(frame)

                    if data:
                        reward = data['reward']
                        print(f"[{time.strftime('%H:%M:%S')}] Scoring successful. Reward = {reward:.4f}")
                        with open(SIGNAL_FILE, 'w') as f:
                            f.write(f"RESULT_{reward:.4f}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] No valid droplet features extracted; returning FAIL.")
                        with open(SIGNAL_FILE, 'w') as f:
                            f.write("RESULT_FAIL")
                else:
                    print("Screen size is abnormal.")
                    with open(SIGNAL_FILE, 'w') as f:
                        f.write("RESULT_FAIL")
            else:
                print("Target window not found. Confirm it is running and not minimized.")
                with open(SIGNAL_FILE, 'w') as f:
                    f.write("RESULT_FAIL")

        except Exception as e:
            print(f"Core analysis logic crashed: {e}")
            with open(SIGNAL_FILE, 'w') as f:
                f.write("RESULT_FAIL")

    time.sleep(0.1)
