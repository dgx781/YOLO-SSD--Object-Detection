import cv2
import glob

from ultralytics import YOLO
model= YOLO('yolov8n.pt')

from screeninfo import get_monitors
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Initialize the webcam
cap = cv2.VideoCapture(0)
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    # Resize the frame to the screen width and height
    frame = cv2.resize(frame, (screen_width, screen_height))
    # Show the image
    result= model.predict(frame, classes=0,iou=0.1)  # Use 'classes' instead of 'clasess'
    
    for box in result[0].boxes:
        class_id = result[0].names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        
        # Extract coordinates
        x_min, y_min, x_max, y_max = cords
        
        # Draw the bounding box on the image
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green color for the box
        text = f'{class_id} ({conf})'
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)   
        cv2.imshow('Webcam with Bounding Box', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()