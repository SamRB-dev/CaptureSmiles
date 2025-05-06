# -*- coding: utf-8 -*-
import cv2
from rich.console import Console

console = Console()

# Camera Feedback
camera = cv2.VideoCapture(0)
def main():
    try:
        while True:
            status, frame = camera.read()
            greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            faces = face_cascade.detectMultiScale(greyscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

                # Extract face region of interest (ROI)
                roi_gray = greyscale_frame[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]

                # Detect smile in face ROI
                smiles = smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=5,   # try tuning this to avoid false positives
                    minSize=(25, 25)
                )

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                    # Optional: print("Smile detected!")

            cv2.imshow('Smile Detector', frame)

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