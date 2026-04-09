import cv2
import numpy as np
import os
import glob
import time
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# 匯入 UKF 相關套件
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ==========================================
# 1. 全域參數與路徑設定
# ==========================================
INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'output_videos'
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

FPS = 30.0
DT = 1.0 / FPS  # 時間間隔

LOWER_YELLOW = np.array([0, 75, 75])
UPPER_YELLOW = np.array([50, 255, 255])
MIN_AREA = 1
MAX_AREA = 1000

# 追蹤器進階參數
BASE_MAX_DISTANCE = 60      # 基礎匹配距離
DIST_EXPAND_RATE = 15       # 每熄滅一幀，搜尋半徑擴大 15 像素
MIN_HITS = 2            
MAX_SKIPPED_FRAMES = 50 

# ==========================================
# 2. CTRV 運動模型定義 (給 UKF 使用)
# ==========================================
def normalize_angle(x):
    """將角度限制在 -pi 到 pi 之間"""
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x

def residual_x(a, b):
    """計算狀態殘差，特別處理角度相減的問題"""
    y = a - b
    y[3] = normalize_angle(y[3])
    return y

def fx_ctrv(x, dt):
    """
    CTRV 模型狀態轉移函數
    狀態向量 x = [x, y, v, theta, omega]
    """
    px, py, v, theta, omega = x
    
    # 避免除以零 (當角速度極小時，視為直線運動)
    if abs(omega) < 0.001:
        new_x = px + v * np.cos(theta) * dt
        new_y = py + v * np.sin(theta) * dt
    else:
        new_x = px + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        new_y = py + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
    
    # 稍微衰減速度與角速度，模擬螢火蟲熄滅時的「減速慣性」
    new_v = v * 0.95 
    new_omega = omega * 0.95
    new_theta = normalize_angle(theta + omega * dt)
    
    return np.array([new_x, new_y, new_v, new_theta, new_omega])

def hx_ctrv(x):
    """測量函數：我們只能從影像中觀測到 (x, y)"""
    return np.array([x[0], x[1]])

# ==========================================
# 3. UKF 追蹤類別
# ==========================================
class FireflyTrack:
    def __init__(self, track_id, center, box):
        self.track_id = track_id
        self.box = box           # (x, y, w, h)
        self.skipped_frames = 0  
        self.hits = 1            
        self.state = 'Tentative'
        self.history = [center]

        # 1. 建立 Sigma Points
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0.)

        # 2. 初始化 UKF (狀態 5 維，觀測 2 維)
        self.ukf = UKF(dim_x=5, dim_z=2, fx=fx_ctrv, hx=hx_ctrv, 
                       dt=DT, points=points, residual_x=residual_x)
        
        # 初始狀態: [x, y, v=0, theta=0, omega=0]
        self.ukf.x = np.array([float(center[0]), float(center[1]), 0.0, 0.0, 0.0])
        
        # 初始不確定性 (P矩陣)：給予 v, theta, omega 較大的初始誤差容忍度
        self.ukf.P = np.diag([10., 10., 50., np.pi, 2.])
        
        # 過程雜訊 (Q矩陣)：允許速度和角速度在模型預測中產生變化
        self.ukf.Q = np.diag([0.1, 0.1, 5.0, 0.5, 0.5])
        
        # 測量雜訊 (R矩陣)：觀測值的可信度
        self.ukf.R = np.diag([3.0, 3.0])

    def predict(self):
        # UKF 的強項：無論是否熄滅，都交給模型去推算，P矩陣會自動放大
        self.ukf.predict()
        return self.ukf.x[:2]

    def update(self, center, box):
        current_pos = np.array([float(center[0]), float(center[1])])
        
        # 獲得新觀測值，進行狀態更新
        self.ukf.update(current_pos)
        
        self.box = box
        self.skipped_frames = 0
        self.hits += 1
        if self.hits >= MIN_HITS:
            self.state = 'Confirmed'
            
        self.history.append((int(current_pos[0]), int(current_pos[1])))
        if len(self.history) > 30: 
            self.history.pop(0)

    def is_dead(self):
        if self.state == 'Tentative' and self.skipped_frames > 1: return True
        if self.state == 'Confirmed' and self.skipped_frames > MAX_SKIPPED_FRAMES: return True
        return False

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 1
        self.total_unique_count = 0 

    def update(self, detections):
        # 1. 預測所有軌跡
        for t in self.tracks:
            t.predict()

        if not detections:
            for t in self.tracks: t.skipped_frames += 1
        else:
            # 2. 匹配偵測點
            track_centers = [t.ukf.x[:2] for t in self.tracks]
            det_centers = [d[:2] for d in detections]
            
            if not track_centers:
                for det in detections: self._add_track(det)
            else:
                cost = distance.cdist(track_centers, det_centers)
                row, col = linear_sum_assignment(cost)

                assigned_t, assigned_d = set(), set()
                for r, c in zip(row, col):
                    # 動態距離門檻：熄滅越久，找回來的範圍越大
                    dynamic_max_dist = BASE_MAX_DISTANCE + (self.tracks[r].skipped_frames * DIST_EXPAND_RATE)
                    
                    if cost[r, c] < dynamic_max_dist:
                        self.tracks[r].update(det_centers[c], detections[c][2])
                        if self.tracks[r].state == 'Confirmed' and self.tracks[r].track_id is None:
                            self.tracks[r].track_id = self.next_track_id
                            self.next_track_id += 1
                            self.total_unique_count += 1
                        assigned_t.add(r)
                        assigned_d.add(c)

                for i, t in enumerate(self.tracks):
                    if i not in assigned_t: t.skipped_frames += 1
                for i, det in enumerate(detections):
                    if i not in assigned_d: self._add_track(det)

        self.tracks = [t for t in self.tracks if not t.is_dead()]

    def _add_track(self, det):
        self.tracks.append(FireflyTrack(None, det[:2], det[2]))

# ==========================================
# 4. 影像處理與主程式 (保持與原版大致相同)
# ==========================================
def get_detections_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (3, 3), 0)
    mask = cv2.inRange(blurred, LOWER_YELLOW, UPPER_YELLOW)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            dets.append((x + w//2, y + h//2, (x, y, w, h)))
    return dets

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_paths = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))

    for path in video_paths:
        filename = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # 防呆機制，若讀不到 fps 預設給 30
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'UKF_TRACK_{filename}'), FOURCC, fps, (w, h))
        
        tracker = Tracker()
        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            f_idx += 1

            dets = get_detections_hsv(frame)
            tracker.update(dets)

            output_frame = frame.copy()
            for t in tracker.tracks:
                if t.state == 'Confirmed':
                    # 依據 UKF 預測狀態繪製框
                    curr_x, curr_y = int(t.ukf.x[0]), int(t.ukf.x[1])
                    bw, bh = t.box[2], t.box[3]
                    
                    color = (0, 255, 0) if t.skipped_frames == 0 else (0, 165, 255) # 熄滅時畫橘色
                    
                    cv2.rectangle(output_frame, (curr_x - bw//2, curr_y - bh//2), 
                                  (curr_x + bw//2, curr_y + bh//2), color, 2)
                    cv2.putText(output_frame, f"ID:{t.track_id}", (curr_x - bw//2, curr_y - bh//2 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # (可選) 畫出歷史軌跡
                    for i in range(1, len(t.history)):
                        cv2.line(output_frame, t.history[i-1], t.history[i], color, 1)

            cv2.putText(output_frame, f"Fireflies: {tracker.total_unique_count}", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(output_frame)
            if f_idx % 30 == 0: print(f"Processing {filename}: Frame {f_idx}")

        cap.release()
        out.release()
        print(f"Done: {filename} | Total Unique Fireflies: {tracker.total_unique_count}")

if __name__ == "__main__":
    main()