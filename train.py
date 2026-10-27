import os
from ultralytics import YOLO

# 1. Establish project directory paths
base_dir = os.getcwd()
merged_dir = os.path.join(base_dir, "merged")
merged_dir_posix = merged_dir.replace('\\', '/')

# 2. Automatically create the train/valid/test folder structure
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(merged_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(merged_dir, split, 'labels'), exist_ok=True)

# 3. Create the configuration file (data.yaml) for YOLO
yaml_content = f"""train: {merged_dir_posix}/train/images
val: {merged_dir_posix}/valid/images
test: {merged_dir_posix}/test/images

nc: 4
names: ['crack', 'excess_reinforcement', 'porosity', 'spatters']
"""

yaml_path = os.path.join(merged_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"✅ Configuration file created at: {yaml_path}")

# 4. Load the base YOLOv8 small model
model = YOLO('yolov8s.pt')

# 5. Start training
# Set device='cpu' if you do not have an NVIDIA GPU set up.
model.train(
    data=yaml_path.replace('\\', '/'),
    epochs=50,             # Number of training loops
    imgsz=416,             # Image dimensions
    batch=4,               # Batch size per iteration
    workers=0,             # Set to 0 to prevent Windows process errors
    device='cpu',          # Change to 0 if an NVIDIA GPU is available
    project='runs/welding',
    name='v1',
    exist_ok=True
)

print("✅ Training complete!")