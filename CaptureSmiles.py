# -*- coding: utf-8 -*-
from ultralytics import YOLO
from rich.console import Console
import cv2 

# Rich Console for colored output
console = Console()

# Video Feed
camera = cv2.VideoCapture(0)

# Custom YOLOv11 Model 
model = YOLO("model/best.pt")

def main():
    try:
        while True:
            status, frame = camera.read()
            results = model(frame)
            cv2.imshow("CaptureSmiles", results[0].plot())
            print(results)
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