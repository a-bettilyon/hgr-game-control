# YOLO26 Hand Gesture Recognition for Game Control

Real-time webcam detection of hand gestures with a YOLO26 (nano) model fine-tuned on a subset of [HaGRIDv2](https://github.com/hukenovs/hagrid). The long-term goal is controlling a video game with gestures.

**Gesture classes:** `call`, `dislike`, `fist`, `like`, `ok`, `palm`, `peace`, `rock`, `stop`.

## How to use

### Step 1: Train in Colab
Download release v0.2.0, which contains the 9-class HaGRID dataset. Also make a copy of 02_yolo26n_hagrid_9class_full_export.ipynb` and open it in Google Colab (set the runtime to a GPU). The notebook:
- reads dataset HaGRID images and converts the boxes to YOLO format,
- fine-tunes `yolo26n.pt` for 100 epochs at 640 px,
- saves results and `best.pt` to your Google Drive.

Download `best.pt` from Drive and place it in `weights/`. A trained copy is already included if you only want to run inference.

### Step 2: Run inference locally

```bash
git clone https://github.com/a-bettilyon/hgr-game-control.git
cd hgr-game-control
conda env create -f environment.yml
conda activate gesture-control
python live_gesture.py
```

Click the video window and press `q` or `Esc` to quit.

**Troubleshooting**
- *Window never appears or closes instantly:* try `--cam 1`. If an iPhone is nearby, macOS may treat it as camera 0 (Continuity Camera); disconnect it.
- *`No module named ultralytics`:* activate the `gesture-control` environment in the same terminal first.
- Make sure to allow camera access for your terminal in System Settings.

### Step 3: Game Control (⚠️ Work in progress)

## Results: (⚠️ Work in progress)
These scores are optimistic. Validation images come from the same dataset and capture style as training, so real webcam performance can differ.

## Acknowledgements and licenses
- Dataset: [HaGRID](https://huggingface.co/datasets/miracl/hagrid). Please cite the HaGRID paper and check the dataset license before reuse.
- Model and training tooling: [Ultralytics](https://github.com/ultralytics/ultralytics). Check their license terms before choosing one for this repo.
