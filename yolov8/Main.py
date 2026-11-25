import pyautogui
import time
import pydirectinput
from Detect import detect_frames
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

GAME_DIR = Path(
    r"C:\Users\chris\PycharmProjects\YOLOGameController\Mario-Level-1-master"
)


def run_vision(stop_event):
    armed = True

    for result in detect_frames():
        if stop_event.is_set():
            break

        detected_classes = {
            result.names[int(class_id)]
            for class_id in result.boxes.cls
        }

        print("Detected:", detected_classes, flush=True)

        if "fist" in detected_classes:
            armed = True
        elif "ok" in detected_classes and armed:
            pyautogui.press("space")
            armed = False
        elif "peace" in detected_classes:
            pydirectinput.keyDown("a")

            pydirectinput.keyUp("a")
        elif "like" in detected_classes:
            pydirectinput.keyDown("right")

            pydirectinput.keyUp("right")
        elif "dislike" in detected_classes:
            pydirectinput.keyDown("left")

            pydirectinput.keyUp("left")
        elif "stop" in detected_classes:
            pydirectinput.keyDown("s")

            pydirectinput.keyUp("s")


def main():
    stop_event = Event()


    game = subprocess.Popen(
        [sys.executable, str(GAME_DIR / "mario_level_1.py")],
        cwd=GAME_DIR,
    )
    vision_thread = Thread(
        target=run_vision,
        args=(stop_event,),
        daemon=True,
    )
    vision_thread.start()

    try:
        game.wait()
    except KeyboardInterrupt:
         game.terminate()
    finally:
        stop_event.set()
        vision_thread.join(timeout=2)

        if game.poll() is None:
            game.terminate()
            game.wait()


if __name__ == "__main__":
    main()
