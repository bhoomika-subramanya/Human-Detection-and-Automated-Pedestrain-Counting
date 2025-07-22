import cv2
import torch
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break 

  
    results = model(frame)

    person_count = 0  

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0]) 
            if model.names[cls] == "person": 
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    cv2.putText(frame, f"Pedestrians: {person_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

  
    cv2.imshow("Pedestrian Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
cv2.destroyAllWindows()