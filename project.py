import cv2
import numpy as np
import os
import glob
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
        return self.ukf.x[:2]

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

    def update(self, detections):
        """
        處理每一幀的追蹤邏輯
        detections: 當前幀 OpenCV 偵測到的所有螢火蟲資料 (dict 列表)
        """
        # 1. 預測階段：所有現有軌跡先根據物理模型往後推算一幀
        for t in self.tracks:
            t.predict()

        # 2. 關聯階段：計算預測位置與實際偵測點的距離矩陣
        num_tracks = len(self.tracks)
        num_detections = len(detections)
        
        # 建立距離矩陣 (Cost Matrix)
        cost_matrix = np.zeros((num_tracks, num_detections))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                dist = np.linalg.norm(t.ukf.x[:2] - d['center'])
                cost_matrix[i, j] = dist

        # 使用匈牙利演算法進行最優匹配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_track_indices = set()
        matched_det_indices = set()

        # 3. 更新階段
        # A. 處理成功匹配的軌跡 (螢火蟲亮起中)
        for t_idx, d_idx in zip(row_ind, col_ind):
            dist = cost_matrix[t_idx, d_idx]
            
            # 設定門檻：如果距離太遠，就不視為同一隻
            if dist < (60 + self.tracks[t_idx].skipped_frames * 10):
                t = self.tracks[t_idx]
                d = detections[d_idx]
                
                # 執行 UKF Update (修正模型)
                t.update(d['center'], d['box'], (d['brightest_bgr'], d['hsv']))
                
                # 印出實測座標 (Live 部分已寫在 FireflyTrack.update 內)
                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)

        # B. 處理未匹配到的軌跡 (螢火蟲熄滅中)
        for i, t in enumerate(self.tracks):
            if i not in matched_track_indices:
                t.skipped_frames += 1
                
                # 只有 Confirmed (確定是螢火蟲) 的才印出預測，避免雜訊洗板
                if t.state == 'Confirmed' and t.track_id:
                    pred_x = int(t.ukf.x[0])
                    pred_y = int(t.ukf.x[1])
                    
                    # 記錄預測點，並標註為 Predicted
                    t.path.append((pred_x, pred_y, "Predicted"))
                    
                    print(f"[Pred] ID:{t.track_id:3d} 預測座標: ({pred_x:4d}, {pred_y:4d}) (連續缺失 {t.skipped_frames:2d} 幀)")

        # C. 處理未被分配的偵測點 (新出現的螢火蟲)
        for j, d in enumerate(detections):
            if j not in matched_det_indices:
                new_track = FireflyTrack(self.next_id, d['center'], d['box'], 
                                        (d['brightest_bgr'], d['hsv']), self.fps)
                self.tracks.append(new_track)
                self.next_id += 1

        # 4. 清理階段：移除消失太久的軌跡
        # 在移除前，印出 Summary (這部分的 print 邏輯建議放在這裡)
        for t in self.tracks:
            if (t.state == 'Confirmed' and t.skipped_frames > MAX_SKIPPED_FRAMES) or \
               (t.state == 'Tentative' and t.skipped_frames > 2):
                if t.track_id:
                    print("-" * 40)
                    print(f"[Track Summary] ID: {t.track_id}")
                    print(f"總路徑點數: {len(t.path)}")
                    # 分離實測點與預測點來觀察
                    live_points = [p for p in t.path if len(p) == 2]
                    print(f"實測座標總數: {len(live_points)}")
                    print(f"完整路徑: {t.path}")
                    print("-" * 40)

        # 執行過濾移除
        self.tracks = [t for t in self.tracks if not (
            (t.state == 'Confirmed' and t.skipped_frames > MAX_SKIPPED_FRAMES) or 
            (t.state == 'Tentative' and t.skipped_frames > 2))]

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    for path in video_files:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'RESULT_{os.path.basename(path)}'), 
                              FOURCC, fps, (w, h))
        
        tracker = Tracker(fps)
        f_idx = 0
        print(f"\n--- 開始處理影片: {os.path.basename(path)} ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
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
            if f_idx % 60 == 0: print(f"進度: Frame {f_idx}")

        cap.release()
        out.release()
        print(f"--- 影片處理完成: {os.path.basename(path)} ---")

if __name__ == "__main__":
    main()