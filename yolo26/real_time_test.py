import time
import cv2
from ultralytics import YOLO

model = YOLO("weights/best.pt")

#opencv for video stats
video_path = "videos/best_case_RH.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"Could not open video file {video_path}. Check the path and file permissions.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Total Frames: {total_frames}")

times = []          # seconds per frame
labels = []         # top detected class per frame ("none" if nothing found)
confs = {}          # class -> list of confidences

frame_index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    start = time.perf_counter()
    result = model.predict(frame, conf=0.25, verbose=False)[0]
    times.append(time.perf_counter() - start)

    if len(result.boxes) > 0:
        best = int(result.boxes.conf.argmax())          # most confident box
        name = result.names[int(result.boxes.cls[best])]
        confs.setdefault(name, []).append(float(result.boxes.conf[best]))
        labels.append(name)
    else:
        labels.append("none")

    frame_index += 1

cap.release()
print(f"Processed {frame_index} frames from the video.")

# --- core statistics ---
avg_ms = 1000 * sum(times) / len(times)
detected = sum(1 for l in labels if l != "none")
switches = sum(1 for a, b in zip(labels, labels[1:]) if a != "none" and b != "none" and a != b)

print(f"Average latency: {avg_ms:.1f} ms/frame ({1000 / avg_ms:.1f} FPS vs {fps:.1f} FPS source)")
print(f"Frames with a detection: {detected}/{frame_index} ({100 * detected / frame_index:.1f}%)")
print(f"Label switches between consecutive detections: {switches}")
for name, c in confs.items():
    print(f"  {name}: {len(c)} frames, mean confidence {sum(c) / len(c):.2f}")