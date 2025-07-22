import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

file_path = input("Enter the full path of the image or video: ")

if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model(frame)

        person_count = 0 

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  
                if model.names[cls] == "person":
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
        cv2.putText(frame, f"Total Pedestrians: {person_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pedestrian Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()
elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    image = cv2.imread(file_path)
    
    if image is None:
        print("❌ Error: Could not open image file.")
        exit()

    results = model(image)  # Run YOLO on image
    person_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, f"Total Pedestrians: {person_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Pedestrian Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("❌ Unsupported file format. Please provide an image (.jpg, .png) or video (.mp4, .avi).")