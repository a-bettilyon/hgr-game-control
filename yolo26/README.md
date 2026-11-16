# YOLO26 Hand Gesture Recognition for Game Control (⚠️ Under Construction)

Real-time webcam detection of hand gestures with a YOLO26 (nano) model fine-tuned on a subset of [HaGRIDv2](https://github.com/hukenovs/hagrid). The long-term goal is controlling a video game with gestures.

**Gestures:** `dislike`, `fist`, `like`, `stop`, `one`, plus `no_gesture` so a resting hand isn't read as a command.

## How to use

### Step 1: Train in Colab
Make a copy of `train_gestures.ipynb` and open it in Google Colab (set the runtime to a GPU). The notebook:
- downloads a small sample of HaGRID images (360 per class) and converts the boxes to YOLO format,
- fine-tunes `yolo26n.pt` for 50 epochs at 640 px,
- saves results and `best.pt` to your Google Drive.

Download `best.pt` from Drive and place it in `weights/`. A trained copy is already included if you just want to run inference.

### Step 2: Run inference locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
conda env create -f environment.yml
conda activate gesture-control
python live_gesture.py
```

Click the video window and press `q` or `Esc` to quit.

| Flag | Default | Purpose |
|---|---|---|
| `--weights` | `weights/best.pt` | trained model file |
| `--cam` | `0` | webcam index |
| `--conf` | `0.5` | confidence threshold |
| `--imgsz` | `640` | inference size |
| `--device` | auto | `cpu`, `mps`, `0`, ... |

**Troubleshooting**
- *Window never appears or closes instantly:* try `--cam 1`. If an iPhone is nearby, macOS may treat it as camera 0 (Continuity Camera); disconnect it.
- *`No module named ultralytics`:* activate the `gesture-control` environment in the same terminal first.
- *Black window on macOS:* allow camera access for your terminal in System Settings → Privacy & Security → Camera.

### Step 3: (⚠️ Work in progress)
Game control: pick one detection per frame, require several consecutive frames before acting, add a cooldown, and send key presses.

## Results
Validation on <N> held-out images: mAP50 <x>, mAP50-95 <y>. See `assets/` for the training curves and confusion matrix.

These scores are optimistic. Validation images come from the same dataset and capture style as training, so real webcam performance can differ.

## Acknowledgements and licenses
- Dataset: [HaGRIDv2](https://github.com/hukenovs/hagrid). Please cite the HaGRID paper and check the dataset license before reuse.
- Model and training tooling: [Ultralytics](https://github.com/ultralytics/ultralytics). Check their license terms before choosing one for this repo.
