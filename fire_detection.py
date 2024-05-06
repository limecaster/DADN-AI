from ultralytics import YOLO
import time
import cv2

class FireDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.dispatch_time = 10
        self.confidence_threshold = 0.7
        self.camera = cv2.VideoCapture(0)
        self.img_counter = 0
        
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
        results = self.model.predict(source=source, imgsz=640, conf=self.confidence_threshold, save=False, show=True)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                return True
        return False
    
# Example usage    
if __name__ == '__main__':
    fire_detection = FireDetection('best.pt')
    print(fire_detection.predict(source='fire_on_house.jpg'))
    print(fire_detection.predict(source='green.png'))
    #print(fire_detection.capture_photo())