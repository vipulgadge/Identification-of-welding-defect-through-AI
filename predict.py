import os
import glob
import cv2
from ultralytics import YOLO

# 1. Load trained model weights
model = YOLO(r'runs\detect\runs\welding\v1\weights\best.pt')

# 2. Find the first image in your test folder automatically
test_images_dir = r'merged\test\images'
image_files = glob.glob(os.path.join(test_images_dir, '*.[jJ][pP][gG]')) + \
              glob.glob(os.path.join(test_images_dir, '*.[pP][nN][gG]'))

if not image_files:
    # Fallback to train folder if test folder is empty
    test_images_dir = r'merged\train\images'
    image_files = glob.glob(os.path.join(test_images_dir, '*.[jJ][pP][gG]'))

if image_files:
    image_path = image_files[0]
    print(f"📷 Testing on image: {image_path}")

    # 3. Run prediction
    results = model.predict(source=image_path, conf=0.25)

    # 4. Save output result image
    for result in results:
        annotated_frame = result.plot()
        cv2.imwrite('output_result.jpg', annotated_frame)

    print("✅ Prediction complete! Open 'output_result.jpg' to view detected defects.")
else:
    print(f"❌ No images found in {test_images_dir}")