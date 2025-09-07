import cv2

video_path = "bmx-trees.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频文件：{video_path}")

# 获取总帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频总帧数：{frame_count}")

cap.release()
