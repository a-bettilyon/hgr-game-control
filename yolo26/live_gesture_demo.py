"""Live webcam hand-gesture detection with a YOLO26 model trained on HaGRID."""
import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="weights/yolo26n_hagrid9_best.pt", help="path to trained .pt file")
    p.add_argument("--cam", type=int, default=0, help="webcam index")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--device", default=None, help="cpu, mps, 0, ... (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.cam}. Try --cam 1 or check camera permissions.")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {float(box.conf[0]):.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        cv2.putText(frame, f"{1 / (now - prev):.0f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        prev = now

        cv2.imshow("gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
