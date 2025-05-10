# -*- coding: utf-8 -*-
from ultralytics import YOLO
from rich.console import Console

# Rich Console for colored output
console = Console()

def main():
    model = YOLO("model/best.pt")
    result = model.predict("tests/data/01.jpg", save=True)
    result.show()
        
        
if __name__ == "__main__":
    main()