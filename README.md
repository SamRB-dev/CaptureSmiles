# CaptureSmiles

A Computer Vision (CV) program that detects smiles using **YOLOv11** model, trained with custom [Data](https://universe.roboflow.com/selfdriving-captb/musebot/dataset/2) and takes a picture and saves it locally. Written in **Python**.

### Requires More Testing

### Tech Stack
- OpenCV
- YOLOv11

### Usage
##### Create Virtual Environment
```python
uv venv .venv
```

##### Activate 
```Shell
source .venv/bin/activate
```

##### Install Required Libraries
```python
uv pip install -r pyproject.toml
```
##### Run 
```python
uv run CaptureSmiles.py
```
