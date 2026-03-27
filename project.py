import cv2
import numpy as np
import os
import glob
import time
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. 全域參數與路徑設定
# ==========================================
INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'output_videos'
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# --- HSV 顏色範圍 (根據你的調整：黃色) ---
LOWER_YELLOW = np.array([0, 75, 75])
UPPER_YELLOW = np.array([50, 255, 255])

# --- 偵測參數 ---
MIN_AREA = 1
MAX_AREA = 1000

# --- 追蹤器進階參數 ---
MAX_DISTANCE = 60       # 匹配的最大像素距離 (適度放寬以利捕捉預測點)
MIN_HITS = 2            # 較快進入確認狀態
MAX_SKIPPED_FRAMES = 50 # 容許熄滅約 0.6 秒 (以 30fps 計)

# ==========================================
# 2. 慣性強化版追蹤類別
# ==========================================
class FireflyTrack:
    def __init__(self, track_id, center, box):
        self.track_id = track_id
        self.box = box           # (x, y, w, h)
        self.skipped_frames = 0  
        self.hits = 1            
        self.state = 'Tentative'
        self.history = [center]

        # 慣性預測關鍵變數
        self.last_seen_pos = np.array([float(center[0]), float(center[1])])
        self.velocity = np.array([0.0, 0.0]) # 像素/每幀 (vx, vy)

        # 初始化 Kalman Filter (作為平滑器)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # F 矩陣: x = x + vx
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R *= 5.0   # 降低觀測雜訊權重
        self.kf.P *= 100.0
        self.kf.x[:2] = self.last_seen_pos.reshape(2, 1)

    def predict(self):
        if self.skipped_frames > 0:
            # 讓速度每一幀衰減 (例如 0.9)，讓它不要飛太遠，增加轉彎後被抓回的機率
            self.velocity *= 0.9 
            self.kf.x[0] += self.velocity[0]
            self.kf.x[1] += self.velocity[1]
        else:
            self.kf.predict()
        return self.kf.x[:2].reshape(-1)

    def update(self, center, box):
        """當亮起時，更新位置並重新計算速度向量"""
        current_pos = np.array([float(center[0]), float(center[1])])
        
        # 計算這一幀的瞬時位移
        instant_v = current_pos - self.last_seen_pos
        
        # 使用動量平滑 (Momentum): 80% 舊速度 + 20% 新位移
        # 這樣可以避免螢火蟲閃爍時的微小抖動導致預測方向大歪
        if self.hits > 1:
            self.velocity = self.velocity * 0.8 + instant_v * 0.2
        else:
            self.velocity = instant_v

        self.last_seen_pos = current_pos
        self.kf.update(current_pos.reshape(2, 1))
        
        self.box = box
        self.skipped_frames = 0
        self.hits += 1
        if self.hits >= MIN_HITS:
            self.state = 'Confirmed'
            
        self.history.append((int(current_pos[0]), int(current_pos[1])))
        if len(self.history) > 30: self.history.pop(0)

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
            track_centers = [t.kf.x[:2].reshape(-1) for t in self.tracks]
            det_centers = [d[:2] for d in detections]
            
            if not track_centers:
                for det in detections: self._add_track(det)
            else:
                cost = distance.cdist(track_centers, det_centers)
                row, col = linear_sum_assignment(cost)

                assigned_t, assigned_d = set(), set()
                for r, c in zip(row, col):
                    if cost[r, c] < MAX_DISTANCE:
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
# 3. 影像處理與主程式
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'VEC_TRACK_{filename}'), FOURCC, fps, (w, h))
        
        tracker = Tracker()
        start_t = time.time()
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
                    # 依據 KF 預測狀態繪製框 (中心點需轉回左上角)
                    curr_x, curr_y = int(t.kf.x[0].item()), int(t.kf.x[1].item())
                    bw, bh = t.box[2], t.box[3]
                    
                    color = (0, 255, 0) if t.skipped_frames == 0 else (0, 0, 255)
                    cv2.rectangle(output_frame, (curr_x - bw//2, curr_y - bh//2), 
                                  (curr_x + bw//2, curr_y + bh//2), color, 2)
                    cv2.putText(output_frame, f"ID:{t.track_id}", (curr_x - bw//2, curr_y - bh//2 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(output_frame, f"Fireflies Counted: {tracker.total_unique_count}", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(output_frame)
            if f_idx % 30 == 0: print(f"Processing {filename}: Frame {f_idx}")

        cap.release()
        out.release()
        print(f"Done: {filename} | Total: {tracker.total_unique_count}")

if __name__ == "__main__":
    main()