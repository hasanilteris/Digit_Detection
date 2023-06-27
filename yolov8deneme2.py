from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("C:\\Users\\hasan\\OneDrive\\Masaüstü\\deneme2\\deneme.pt")

model.predict(source="0", show=True, conf=0.5)
print(model)