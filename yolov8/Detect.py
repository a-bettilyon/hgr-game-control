import cv2
from ultralytics import YOLO
#load weights

def detect_frames():
    model = YOLO(r"C:\Users\chris\Desktop\Weights_TimTest\best_openvino_model")

    #open and config webcam 0 for default cam and will use actual resolution if not sup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    #print if cannot find webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")
    printed_resolution = False
    while True:
        success, frame = cap.read()
        if not success:
            print("no frame")
            break
        if not printed_resolution:
            height, width = frame.shape[:2]
            print(f"Actual camera resolution: {width} x {height}")
            printed_resolution = True
        result = model.predict(
            frame,
            conf=0.25,
            imgsz=736,
            device="xpu",
            verbose=False,
        )[0]

        cv2.imshow("YOLO Detection - press Q to quit", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
        yield result
