import argparse
import json
import cv2
import numpy as np
import os
import glob
from typing import List, Optional
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# 匯入 UKF 相關套件
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ==========================================
# 1. 全域參數設定
# ==========================================
INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'output_videos'
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.MTS']
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# 影像處理參數 (根據你的環境微調)
LOWER_YELLOW = np.array([20, 50, 50])
UPPER_YELLOW = np.array([50, 255, 255])
MIN_AREA = 5
MAX_AREA = 1000

# 追蹤進階參數
BASE_MAX_DISTANCE = 60      # 基礎匹配像素距離
DIST_EXPAND_RATE = 10       # 熄滅時每幀擴張的搜尋半徑
MAX_SKIPPED_FRAMES = 50     # 容許熄滅的最大幀數
MAX_ANGLE_DIFF = np.radians(90) # 運動角度偏差門檻

# Toggle UKF prediction logging
PRINT_PREDICTIONS = True

# ==========================================
# 2. 運動模型與數學工具
# ==========================================
def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi: x -= 2 * np.pi
    return x

def residual_x(a, b):
    y = a - b
    y[3] = normalize_angle(y[3])
    return y

def fx_ctrv(x, dt):
    px, py, v, theta, omega = x
    if abs(omega) < 0.001:
        new_x = px + v * np.cos(theta) * dt
        new_y = py + v * np.sin(theta) * dt
    else:
        new_x = px + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        new_y = py + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
    
    new_v = v * 0.98 # 模擬空氣阻力或動量衰減
    new_omega = omega * 0.95
    new_theta = normalize_angle(theta + omega * dt)
    return np.array([new_x, new_y, new_v, new_theta, new_omega])

def hx_ctrv(x):
    return np.array([x[0], x[1]])

# ==========================================
# 3. 核心追蹤類別
# ==========================================
class FireflyTrack:
    def __init__(self, track_id, center, box, color_data, fps):
        self.track_id = track_id
        self.box = box
        self.fps = fps
        self.skipped_frames = 0
        self.total_active_frames = 1 
        self.state = 'Tentative'
        self.path = [(int(center[0]), int(center[1]))]

        # 儲存色彩數據: (BGR, HSV)
        self.last_bgr = color_data[0]
        self.last_hsv = color_data[1]

        # 初始化 UKF
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0.)
        self.ukf = UKF(dim_x=5, dim_z=2, fx=fx_ctrv, hx=hx_ctrv, 
                       dt=1.0/fps, points=points, residual_x=residual_x)
        self.ukf.x = np.array([float(center[0]), float(center[1]), 0.0, 0.0, 0.0])
        self.ukf.P = np.diag([10., 10., 50., np.pi, 2.])
        self.ukf.Q = np.diag([0.1, 0.1, 5.0, 0.5, 0.5])
        self.ukf.R = np.diag([3.0, 3.0])

    def predict(self):
        self.ukf.predict()
        pred = self.ukf.x.copy()
        if PRINT_PREDICTIONS:
            track_label = f"{self.track_id:3d}" if self.track_id is not None else "TENT"
            print(
                f"[Pred] ID:{track_label} x={pred[0]:7.2f} y={pred[1]:7.2f} "
                f"v={pred[2]:6.2f} theta={pred[3]:6.2f} omega={pred[4]:6.2f}"
            )
        return pred[:2]

    def update(self, center, box, color_data):
        self.ukf.update(np.array([float(center[0]), float(center[1])]))

        # --- 新增：即時記錄與印出座標 ---
        curr_pos = (int(center[0]), int(center[1]))
        self.path.append(curr_pos)
        if self.track_id:
            print(f"[Live] ID:{self.track_id:3d} 座標: {curr_pos}")
        # ----------------------------

        self.box = box
        self.last_bgr = color_data[0]
        self.last_hsv = color_data[1]
        self.skipped_frames = 0
        self.total_active_frames += 1
        if self.total_active_frames >= 2:
            self.state = 'Confirmed'

class Tracker:
    def __init__(self, fps):
        self.tracks = []
        self.next_id = 1
        self.fps = fps
        self.summaries: List[dict] = []
        self._summarized_ids = set()

    def _summarize_track(self, track) -> None:
        if track.track_id is None or track.track_id in self._summarized_ids:
            return
        duration = track.total_active_frames / self.fps
        path = [[int(x), int(y)] for x, y in track.path]
        brightest = None
        if track.last_bgr is not None:
            brightest = [int(v) for v in track.last_bgr]
        self.summaries.append(
            {
                "track_id": int(track.track_id),
                "duration_s": float(duration),
                "path": path,
                "brightest_bgr": brightest,
            }
        )
        self._summarized_ids.add(track.track_id)

    def update(self, detections):
        for t in self.tracks: t.predict()

        if detections:
            track_centers = [t.ukf.x[:2] for t in self.tracks]
            det_centers = [d[:2] for d in detections]
            
            if not track_centers:
                for det in detections: self._add_track(det)
            else:
                cost = distance.cdist(track_centers, det_centers)
                row, col = linear_sum_assignment(cost)
                assigned_t, assigned_d = set(), set()

                for r, c in zip(row, col):
                    track = self.tracks[r]
                    det_pos = det_centers[c]
                    
                    # 計算夾角檢查
                    dx, dy = det_pos[0] - track.ukf.x[0], det_pos[1] - track.ukf.x[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # 邏輯優化：位移大於 10 像素才檢查角度，避免變暗時的質心震盪導致斷追蹤
                    angle_ok = True
                    if dist > 10:
                        move_angle = np.arctan2(dy, dx)
                        angle_err = abs(normalize_angle(move_angle - track.ukf.x[3]))
                        if angle_err > MAX_ANGLE_DIFF: angle_ok = False

                    dynamic_dist = BASE_MAX_DISTANCE + (track.skipped_frames * DIST_EXPAND_RATE)
                    
                    if cost[r, c] < dynamic_dist and angle_ok:
                        track.update(det_pos, detections[c][2], detections[c][3])
                        if track.state == 'Confirmed' and track.track_id is None:
                            track.track_id = self.next_id
                            self.next_id += 1
                        assigned_t.add(r)
                        assigned_d.add(c)

                for i, t in enumerate(self.tracks):
                    if i not in assigned_t: t.skipped_frames += 1
                for i, det in enumerate(detections):
                    if i not in assigned_d: self._add_track(det)
        else:
            for t in self.tracks: t.skipped_frames += 1

        # 在移除死亡 ID 前，印出其完整數據
        for t in self.tracks:
            if (t.state == 'Confirmed' and t.skipped_frames > MAX_SKIPPED_FRAMES) or \
               (t.state == 'Tentative' and t.skipped_frames > 2):
                self._summarize_track(t)
                if t.track_id:
                    duration = t.total_active_frames / self.fps
                    print("-" * 30)
                    print(f"[Track Summary] ID:{t.track_id:3d}")
                    print(f"時長: {duration:5.2f}s")
                    print(f"移動軌跡 (x, y): {t.path}") # 印出整段座標清單
                    print(f"最亮 BGR: {t.last_bgr}")
                    print("-" * 30)

        self.tracks = [t for t in self.tracks if not (
            (t.state == 'Confirmed' and t.skipped_frames > MAX_SKIPPED_FRAMES) or 
            (t.state == 'Tentative' and t.skipped_frames > 2))]

    def finalize(self) -> None:
        for track in self.tracks:
            if track.state == 'Confirmed':
                self._summarize_track(track)

    def _add_track(self, det):
        self.tracks.append(FireflyTrack(None, det[:2], det[2], det[3], self.fps))

# ==========================================
# 4. 影像處理與主程式
# ==========================================
def get_detections_with_color(frame):
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.GaussianBlur(hsv_full, (3, 3), 0), LOWER_YELLOW, UPPER_YELLOW)
    
    # 形態學閉運算：防止變暗時光點碎裂
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            # 提取 ROI 最亮點數據
            roi_hsv = hsv_full[y:y+h, x:x+w]
            roi_bgr = frame[y:y+h, x:x+w]
            _, _, _, max_loc = cv2.minMaxLoc(roi_hsv[:, :, 2])
            brightest_hsv = roi_hsv[max_loc[1], max_loc[0]]
            brightest_bgr = roi_bgr[max_loc[1], max_loc[0]]
            
            dets.append((x + w//2, y + h//2, (x, y, w, h), (brightest_bgr, brightest_hsv)))
    return dets

def _write_firefly_json(path: str, records: List[dict]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def process_video(input_path: str, output_path: Optional[str], json_path: Optional[str]) -> int:
    if not os.path.isfile(input_path):
        print(f"找不到影片: {input_path}")
        return 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {input_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f'RESULT_{os.path.basename(input_path)}')
    else:
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    out = cv2.VideoWriter(output_path, FOURCC, fps, (w, h))
    tracker = Tracker(fps)
    f_idx = 0
    print(f"\n--- 開始處理影片: {os.path.basename(input_path)} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1

        dets = get_detections_with_color(frame)
        tracker.update(dets)

        for t in tracker.tracks:
            if t.state == 'Confirmed':
                cx, cy = int(t.ukf.x[0]), int(t.ukf.x[1])
                bx, by, bw, bh = t.box
                color = (0, 255, 0) if t.skipped_frames == 0 else (0, 165, 255)
                cv2.rectangle(frame, (cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2), color, 2)
                cv2.putText(frame, f"ID:{t.track_id}", (cx-bw//2, cy-bh//2-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        if f_idx % 60 == 0:
            print(f"進度: Frame {f_idx}")

    tracker.finalize()
    if json_path:
        _write_firefly_json(json_path, tracker.summaries)

    cap.release()
    out.release()
    print(f"--- 影片處理完成: {os.path.basename(input_path)} ---")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Firefly tracking")
    parser.add_argument("-i", "--input", dest="input_path", help="input video path")
    parser.add_argument("-o", "--output", dest="output_path", help="output video path")
    parser.add_argument("-j", "--json", dest="json_path", help="output json path")
    args = parser.parse_args()

    if args.input_path:
        raise SystemExit(process_video(args.input_path, args.output_path, args.json_path))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    for path in video_files:
        output_path = os.path.join(OUTPUT_DIR, f'RESULT_{os.path.basename(path)}')
        process_video(path, output_path, None)

if __name__ == "__main__":
    main()