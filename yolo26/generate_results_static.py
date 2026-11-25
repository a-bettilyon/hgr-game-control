from ultralytics import YOLO

model = YOLO("weights/best.pt")     # replace with your best weights path
m = model.val(data="hagrid9_yolo/data.yaml", split="test")

total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

model.info()

print(m.box.mp, m.box.mr, m.box.map50, m.box.map)  # precision, recall, mAP50, mAP50-95
print(m.box.maps)                                   # per-class mAP50-95
