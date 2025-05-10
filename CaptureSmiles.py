# -*- coding: utf-8 -*-
from ultralytics import YOLO
from rich.console import Console
import cv2, datetime

# Rich Console for colored output
console = Console()

# Video Feed
camera = cv2.VideoCapture(0)

# Custom YOLOv11 Model 
model = YOLO("model/Smile_Detector_v2.pt")

def main() -> None:
    try:
        while True:
            status, frame = camera.read()
            results = model(frame)
            cv2.imshow("CaptureSmiles", results[0].plot())
            if len(results[0].boxes.conf) != 0 and results[0].boxes.conf[0] > 0.6:
                console.print(f"✅ [bold green]Smile detected![/bold green]")
                cv2.imwrite(f"smile_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.jpg", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        console.print("[bold red]Exiting...[/bold red]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        
    finally:
        camera.release()
        cv2.destroyAllWindows()
        console.print("✅ [bold green]Done[/bold green]")

if __name__ == "__main__":
    main()