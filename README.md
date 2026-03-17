# 👁️ OPTIC: Real-Time Spatial Tracking Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OPTIC** is a device-agnostic, tracking-by-detection analytics pipeline designed for high-precision spatial monitoring. By merging SOTA object detection with temporal persistence, OPTIC transforms raw video into a "virtual turnstile," providing actionable telemetry without the overhead of expensive hardware sensors.

[**Explore the Live Dashboard**](https://optic-spatial-track.streamlit.app) 

---

## 🎬 System Demo

![OPTIC Demo](assets/demo.gif)
*Real-time entity tracking and ROI intersection logic in action.*

---

## 🧠 Core Architecture & Technical Logic

OPTIC bridges the gap between raw pixel data and spatial intelligence through a four-stage pipeline:

### 1. Detection & Inference
Utilizes **YOLOv8 Nano** for real-time bounding box generation. The model is optimized for low-latency environments while maintaining high mAP (mean Average Precision) for human and object detection.

### 2. Temporal State Management (Memory)
Implements **ByteTrack** with Kalman filtering. This ensures that even during brief occlusions or "noisy" frames, the system retains a persistent ID for every entity, preventing double-counting errors.

### 3. Spatial Logic & ROI Tripwires
The engine utilizes OpenCV’s `pointPolygonTest`. It calculates the relationship between an entity's centroid $C(x, y)$ and a user-defined Polygon $P$:

$$f(P, C) = \text{cv2.pointPolygonTest}(P, C, \text{False})$$

* **Result > 0:** Entity is inside the ROI.
* **Result < 0:** Entity is outside the ROI.

### 4. Hardware-Agnostic Acceleration
The pipeline automatically detects and initializes the most efficient compute provider available on the host machine:
* **NVIDIA:** `CUDA`
* **Apple Silicon:** `MPS` (Metal Performance Shaders)
* **Generic:** `CPU`

---

## 🛠️ Key Features

* ✅ **Real-time Analytics:** Sub-30ms inference on modern hardware.
* ✅ **Dynamic ROI:** Custom "Tripwire" zones definable via the Streamlit UI.
* ✅ **Hardware Optimization:** Native support for M-series chips and NVIDIA GPUs.
* ✅ **Modular UI:** Clean, custom-styled Streamlit dashboard for telemetry visualization.

---

## ⚙️ Quick Start

### 1. Clone & Environment
```bash
git clone [https://github.com/Tushar-Sharma43/optic-spatial-tracking.git](https://github.com/Tushar-Sharma43/optic-spatial-tracking.git)
cd optic-spatial-tracking

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

## Acknowledgments
Code structure, deployment debugging, and optimization logic were developed with assistance from Google Gemini.
