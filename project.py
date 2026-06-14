import cv2
import numpy as np
import os
import glob
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# 匯入 UKF 相關套件
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'output_videos'
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.MTS']
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# ================= 參數設定區 =================
# 1. 螢火蟲偵測參數 (HSV 黃/綠色螢光)
LOWER_YELLOW = np.array([20, 50, 50])
UPPER_YELLOW = np.array([50, 255, 255])
MIN_AREA = 5       # 最小物件面積
MAX_AREA = 1000    # 最大物件面積

# 2. 靜態干擾綠斑偵測參數 (暗綠色/微光雜訊)
NOISE_LOWER_GREEN = np.array([35, 20, 20])
NOISE_UPPER_GREEN = np.array([85, 255, 200])
NOISE_MIN_AREA = 1600

# 3. 追蹤器參數
BASE_MAX_DISTANCE = 60      # 基礎匹配像素距離
DIST_EXPAND_RATE = 10       # 熄滅時每幀擴張的搜尋半徑
MAX_SKIPPED_FRAMES = 50     # 容許熄滅的最大幀數
MAX_ANGLE_DIFF = np.radians(90) # 運動角度偏差門檻
# ==============================================


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
    
    new_v = v 
    new_omega = omega
    new_theta = normalize_angle(theta + omega * dt)
    return np.array([new_x, new_y, new_v, new_theta, new_omega])

def hx_ctrv(x):
    return np.array([x[0], x[1]])


class FireflyTrack:
    def __init__(self, track_id, center, box, color_data, fps):
        self.track_id = track_id
        self.box = box # (x, y, w, h)
        self.fps = fps
        self.skipped_frames = 0
        self.total_active_frames = 1 
        self.state = 'Tentative'
        
        # 軌跡路徑：記錄 (x, y, 是否為預測點)
        self.path = [(int(center[0]), int(center[1]), False)]

        # 初始化 UKF (CTRV 模型)
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0.)
        self.ukf = UKF(dim_x=5, dim_z=2, fx=fx_ctrv, hx=hx_ctrv, 
                       dt=1.0/fps, points=points, residual_x=residual_x)
        # 狀態區向量 x: [px, py, v, theta, omega]
        self.ukf.x = np.array([float(center[0]), float(center[1]), 0.0, 0.0, 0.0])
        
        # 協方差矩陣 P, Q, R 設定
        self.ukf.P = np.diag([5., 5., 100., np.pi/4, 0.5]) 
        self.ukf.Q = np.diag([0.5, 0.5, 5.0, 0.01, 0.01])   
        self.ukf.R = np.diag([3.0, 3.0])

    def predict(self):
        self.ukf.predict()
        return self.ukf.x.copy()[:2]

    def update(self, center, box, color_data):
        self.ukf.update(np.array([float(center[0]), float(center[1])]))

        # 速度約束：v 必須為正
        if self.ukf.x[2] < 0:  
            self.ukf.x[2] = -self.ukf.x[2]  
            self.ukf.x[3] = self.ukf.x[3] + np.pi  
            self.ukf.x[3] = np.arctan2(np.sin(self.ukf.x[3]), np.cos(self.ukf.x[3]))

        curr_pos = (int(center[0]), int(center[1]))
        self.path.append((curr_pos[0], curr_pos[1], False)) # 記錄為真實點

        self.box = box
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
        # 1. 預測
        for t in self.tracks: t.predict()

        if detections:
            track_centers = [t.ukf.x[:2] for t in self.tracks]
            det_centers = [d[:2] for d in detections]
            
            if not track_centers:
                for det in detections: self._add_track(det)
            else:
                # 2. 匈牙利算法配對
                cost = distance.cdist(track_centers, det_centers)
                row, col = linear_sum_assignment(cost)
                assigned_t, assigned_d = set(), set()

                for r, c in zip(row, col):
                    track = self.tracks[r]
                    det_pos = det_centers[c]
                    
                    # 計算運動角度偏差
                    dx, dy = det_pos[0] - track.ukf.x[0], det_pos[1] - track.ukf.x[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    angle_ok = True
                    if dist > 10:
                        move_angle = np.arctan2(dy, dx)
                        angle_err = abs(normalize_angle(move_angle - track.ukf.x[3]))
                        if angle_err > MAX_ANGLE_DIFF: angle_ok = False

                    # 動態搜尋半徑
                    dynamic_dist = BASE_MAX_DISTANCE + (track.skipped_frames * DIST_EXPAND_RATE)
                    
                    # 門檻驗證（距離 + 角度）
                    if cost[r, c] < dynamic_dist and angle_ok:
                        track.update(det_pos, detections[c][2], detections[c][3])
                        if track.state == 'Confirmed' and track.track_id is None:
                            track.track_id = self.next_id
                            self.next_id += 1
                        assigned_t.add(r)
                        assigned_d.add(c)

                # 3. 處理未配對的 Track (進入盲推)
                for i, t in enumerate(self.tracks):
                    if i not in assigned_t: 
                        t.skipped_frames += 1
                        # 軌跡記錄預測位置
                        t.path.append((int(t.ukf.x[0]), int(t.ukf.x[1]), True)) 
                        
                # 4. 處理未配對的 Detection (新增 Track)
                for i, det in enumerate(detections):
                    if i not in assigned_d: self._add_track(det)
        else:
            # 無偵測點，所有 Track 進入盲推
            for t in self.tracks: 
                t.skipped_frames += 1
                t.path.append((int(t.ukf.x[0]), int(t.ukf.x[1]), True))

        # 5. 移除過期或無效的 Track
        self.tracks = [t for t in self.tracks if not (
            (t.state == 'Confirmed' and t.skipped_frames > MAX_SKIPPED_FRAMES) or 
            (t.state == 'Tentative' and t.skipped_frames > 2))]

    def _add_track(self, det):
        self.tracks.append(FireflyTrack(None, det[:2], det[2], det[3], self.fps))


# ================= 靜態遮罩自動生成模組 =================
def auto_generate_hsv_static_mask(cap, num_frames=30):
    """
    透過多幀平均與 HSV 顏色過濾，自動偵測並生成靜態干擾區的遮罩
    """
    print(f"正在分析前 {num_frames} 幀以自動建立靜態干擾遮罩...")
    
    # 確保影片回到開頭
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        
    if not frames: return None

    # 1. 疊加計算平均值，過濾掉移動中的螢火蟲
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
    # 2. 轉換至 HSV 並抓取指定顏色的綠斑
    hsv_frame = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_frame, NOISE_LOWER_GREEN, NOISE_UPPER_GREEN)
    
    # 3. 形態學膨脹讓區塊完整
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(color_mask, kernel, iterations=2)
    
    # 4. 尋找輪廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 修正維度解包問題，只取前兩個值 (h, w)
    h, w = avg_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_created = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > NOISE_MIN_AREA:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            # 向外擴展 15 像素緩衝區
            pad = 15
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + box_w + pad), min(h, y + box_h + pad)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            print(f"[*] 自動偵測到靜態綠斑! 座標:({x1}, {y1}) 面積:{int(area)}")
            mask_created = True
            
    if not mask_created:
        print("[*] 畫面乾淨，未偵測到靜態綠斑干擾。")

    # 分析完畢，將影片倒帶回第 0 幀，供主迴圈使用
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return mask if mask_created else None
# ======================================================


def get_detections_with_color(frame):
    # 影像處理獲取偵測點
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 高斯模糊減少雜訊
    blurred_hsv = cv2.GaussianBlur(hsv_full, (3, 3), 0)
    mask = cv2.inRange(blurred_hsv, LOWER_YELLOW, UPPER_YELLOW)
    
    # 形態學閉運算連接斷點
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            dets.append((x + w//2, y + h//2, (x, y, w, h), (None, None)))
    return dets


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not video_files:
        print(f"找不到影片檔案於: {INPUT_DIR}")
        return

    for path in video_files:
        cap = cv2.VideoCapture(path)
        
        # 🌟 【新加入】在開始處理前，全自動生成遮罩 🌟
        interference_mask = auto_generate_hsv_static_mask(cap, num_frames=30)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 取得影片寬高 (使用標準屬性代碼 3 和 4)
        w, h = int(cap.get(3)), int(cap.get(4))
        
        output_path = os.path.join(OUTPUT_DIR, f'RESULT_{os.path.basename(path)}')
        out = cv2.VideoWriter(output_path, FOURCC, fps, (w, h))
        
        tracker = Tracker(fps)
        f_idx = 0
        print(f"\n--- 開始處理影片: {os.path.basename(path)} ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            f_idx += 1

            # 🌟 【新加入】如果這部影片有生成遮罩，就在偵測前把它塗黑 🌟
            if interference_mask is not None:
                # 遮罩為白色的地方 (255)，在原圖上被替換成黑色 (0, 0, 0)
                frame[interference_mask == 255] = (0, 0, 0)

            dets = get_detections_with_color(frame)
            tracker.update(dets)

            # 視覺化繪製：軌跡、外框與 ID
            for t in tracker.tracks:
                if t.state == 'Confirmed':
                    # A. 繪製歷史軌跡線條
                    for i in range(1, len(t.path)):
                        pt1 = t.path[i-1][:2]
                        pt2 = t.path[i][:2]
                        is_pred = t.path[i][2]
                        
                        # 真實偵測為綠線，盲推預測為橘線
                        line_color = (0, 165, 255) if is_pred else (0, 255, 0)
                        cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

                    # B. 繪製當前外框與 ID
                    cx, cy = int(t.ukf.x[0]), int(t.ukf.x[1])
                    _, _, bw, bh = t.box
                    is_predicting = t.skipped_frames > 0
                    
                    if is_predicting:
                        color = (0, 165, 255)
                        text = f"ID:{t.track_id} (Loss)"
                    else:
                        color = (0, 255, 0)
                        text = f"ID:{t.track_id}"
                    
                    cv2.rectangle(frame, (cx - bw//2, cy - bh//2), 
                                  (cx + bw//2, cy + bh//2), color, 2)
                    cv2.putText(frame, text, (cx - bw//2, cy - bh//2 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            if f_idx % 60 == 0: print(f"進度: Frame {f_idx}")

        cap.release()
        out.release()
        print(f"--- 影片處理完成: {os.path.basename(path)} ---")

if __name__ == "__main__":
    main()