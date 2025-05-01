from ultralytics import YOLO
import cv2

model = YOLO('yolov11n.pt')  # Try with a generic model or a face-specific one

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO Test', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
