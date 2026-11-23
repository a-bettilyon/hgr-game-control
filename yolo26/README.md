# YOLO26 Hand Gesture Recognition for Game Control

Real-time webcam detection of hand gestures with a YOLO26 (nano) model fine-tuned on a subset of [HaGRID](https://github.com/hukenovs/hagrid). The long-term goal is controlling a video game with gestures.


**Gesture classes:** `call`, `dislike`, `fist`, `like`, `ok`, `palm`, `peace`, `rock`, `stop`.

## How to use

### Step 1: Train in Colab
Download release v0.2.0, which contains the 9-class HaGRID dataset. Also make a copy of 02_yolo26n_hagrid_9class_full_export.ipynb` and open it in Google Colab (set the runtime to a GPU). The notebook:
- reads dataset HaGRID images and converts the boxes to YOLO format,
- fine-tunes `yolo26n.pt` (YOLO26 nano) for up to 100 epochs at 640 px with early stopping (our run stopped at epoch 82; best weights from epoch 72),
- saves results and `yolo26n_hagrid9_best.pt` to your Google Drive.

Download `yolo26n_hagrid9_best.pt` from Drive and place it in `weights/`. A trained copy is already included if you only want to run inference.

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
- Make sure to allow camera access for your terminal in settings.

## Results

We evaluate the model two ways: **static image evaluation** on a held-out test set, and **real-time video evaluation** on live or recorded video.

### Static image evaluation (test set)

These scores are optimistic. Test images come from the same dataset and capture style as training, so real webcam performance can differ. Note that Images and Instances are equal because each image contains one labeled hand.

| Class   | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---------|-------:|----------:|----------:|-------:|------:|---------:|
| **all** | 1563 | 1563 | 0.993 | 0.992 | 0.993 | 0.871 |
| call    | 171 | 171 | 0.987 | 1.000 | 0.994 | 0.857 |
| dislike | 169 | 169 | 0.998 | 1.000 | 0.995 | 0.865 |
| fist    | 176 | 176 | 1.000 | 0.997 | 0.995 | 0.839 |
| like    | 192 | 192 | 1.000 | 0.991 | 0.995 | 0.854 |
| ok      | 157 | 157 | 1.000 | 0.997 | 0.995 | 0.901 |
| palm    | 140 | 140 | 0.983 | 0.971 | 0.991 | 0.911 |
| peace   | 203 | 203 | 0.996 | 0.990 | 0.995 | 0.867 |
| rock    | 180 | 180 | 0.995 | 0.989 | 0.995 | 0.866 |
| stop    | 175 | 175 | 0.975 | 0.994 | 0.981 | 0.880 |

### Real-time video evaluation (in progress)
Per-frame latency, FPS, detection stability, and gesture accuracy on video.

## Acknowledgements and licenses
- Dataset: [HaGRID](https://github.com/hukenovs/hagrid) (Kapitanov et al., WACV 2024). Please cite the paper and check the dataset license before reuse.
- Model and training tooling: [Ultralytics](https://github.com/ultralytics/ultralytics), licensed AGPL-3.0 (or under an enterprise license). Consider this when choosing a license for this repo.
