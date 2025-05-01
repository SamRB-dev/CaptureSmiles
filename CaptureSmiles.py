# -*- coding: utf-8 -*-
import cv2
from rich.console import Console

console = Console()

# Camera Feedback
camera = cv2.VideoCapture(0)
def main():
    try:
        while True:
            # Capture frame
            status, frame = camera.read()
            
            # Greyscale the frame
            greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(greyscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
            # Display the frame
            cv2.imshow('Smile Detector', frame)
                
            # Quit Procedure
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        console.print(f":warning: Error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        console.print(":heavy_check_mark: Done")
        
        
if __name__ == "__main__":
    main()