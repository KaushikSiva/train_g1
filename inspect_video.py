import cv2
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("video:", video_path)
print("frame_count:", frame_count)
print("fps:", fps)
print("size:", width, "x", height)

cap.release()