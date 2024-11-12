from ultralytics import YOLO
import cv2
from PIL import Image

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, source: int):
        results = self.model.predict(source=source, show=False)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            #result.show()  # display to screen
        return results
    