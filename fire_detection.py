from ultralytics import YOLO
import time
import cv2
from flask import Flask, jsonify
from threading import Timer
app = Flask(__name__)

class FireDetectionServer:
    def __init__(self, model_path, dispatch_time):
        self.model = YOLO(model_path)
        self.dispatch_time = dispatch_time
        self.confidence_threshold = 0.7
        self.camera = cv2.VideoCapture(0)
        self.img_counter = 0
        self.timer = None
        
    def start(self):
        self.timer = Timer(self.dispatch_time, self.dispatch_result)
        self.timer.start()
        
    def stop(self):
        if self.timer is not None:
            self.timer.cancel()
            
    def dispatch_result(self):
        result = self.capture_photo()
        # Dispatch the result to the desired destination
        # For example, you can send it via HTTP request or save it to a database
        print(result)
        
        # Restart the timer for the next dispatch
        self.timer = Timer(self.dispatch_time, self.dispatch_result)
        self.timer.start()
        
    def capture_photo(self):
        ret, frame = self.camera.read()

        if not ret:
            print("failed to grab frame")
            return False

        cv2.imshow("test", frame)
        img_path = "opencv_frame_{}.png".format(self.img_counter)
        cv2.imwrite(img_path, frame)
        self.img_counter += 1

        return self.predict(img_path)
 
    def predict(self, source):
        results = self.model.predict(source=source, imgsz=640, conf=self.confidence_threshold, save=False, show=False)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                return True
        return False

# Example usage
if __name__ == '__main__':
    fire_detection_server = FireDetectionServer('best.pt', dispatch_time=10)
    fire_detection_server.start()
    app.run()
    #print(fire_detection.capture_photo())