import cv2
import os
import sys

video_path = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("video:", video_path)
print("frame_count:", frame_count)

idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
    cv2.imwrite(out_path, frame)

    if idx % 100 == 0:
        print("saved", idx)

    idx += 1

cap.release()
print("done, saved:", idx)