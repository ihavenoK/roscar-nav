# Model Files

The model weight files (*.pt) are excluded from Git.  Download them before building:

## Required Models

| File | Source | Purpose |
|------|--------|---------|
| `yolov8n.pt` | `pip install ultralytics` (auto-downloaded by YOLO on first run) | Person detection |
| `best_new.pt` | Custom trained model — place in this directory | Traffic light detection |

## Setup

```bash
# The person detection model is automatically downloaded by Ultralytics YOLO
# on first run.  You may also manually download it:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# Place the custom traffic light model here:
# cp /path/to/your/best_new.pt src/multi_nav_traffic/model/
```
