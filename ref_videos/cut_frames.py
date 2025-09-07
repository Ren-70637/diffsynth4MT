import os
import cv2

video_path = "/home/Wind645/code/diffsynth4MT/ref_videos/293408_small.mp4"
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频文件：{video_path}")

# 获取原视频参数
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter
fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
out_path = os.path.join(output_dir, "cardesert41.mp4")
writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

for idx in range(225):
    ret, frame = cap.read()
    if not ret:
        break
    if idx >= 184:
        writer.write(frame)

cap.release()
writer.release()
print(f"已将前 {idx+1} 帧保存为视频：{out_path}")