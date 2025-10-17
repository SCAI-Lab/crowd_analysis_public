# Crowdbot Generation Toolkit

This repository integrates three key components for 2D/3D LiDAR-based object **detection**, **tracking**, and **evaluation**. It consolidates and organizes previously separate projects into a unified, modular toolkit.

---

## 🔧 Components

### 1. **LiDAR Detection (2D & 3D)**  
**Located in:** `lidar_det_2D_3D/`  
**Package name:** `lidar_det`

- Implements DR-SPAAM (2D detection) and Person-MinkUNet (3D detection) models.
- Install with:

```bash
pip install -e ./lidar_det_2D_3D
```

This module includes two internal dependencies that must be installed separately:

```bash
pip install -e ./lidar_det_2D_3D/lib/iou3d
pip install -e ./lidar_det_2D_3D/lib/jrdb_det3d_eval
```

---

### 2. **LiDAR-based Tracking (AB3DMOT)**  
**Located in:** `AB3DMOT/`  
**Package name:** `ab3dmot`

- Tracks 2D/3D objects from LiDAR using the AB3DMOT framework.
- Install with:

```bash
pip install -e ./AB3DMOT
```

---

### 3. **Evaluation Tools**  
**Located in:** `evaluation_tools/`  
**Package name:** `qolo`

- Provides evaluation, visualization, and dataset utilities for the Crowdbot dataset.
- Install with:

```bash
pip install -e ./evaluation_tools
```

---

## ✅ Requirements

A full list of required Python packages is provided in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

> **Note:**
> - Compatible with **CUDA 11.8** (required for `torch==2.0.0`, `torchvision==0.15.0`)
> - `torchsparse==2.0.0` is available pre-built — no need to build from source

---

### 📦 `requirements.txt` Contents

#### 🔹 Core (used by main repo)

```text
numpy==1.24.4
pandas==2.0.3
matplotlib==3.7.5
scipy==1.10.1
numba==0.58.1
filterpy==1.4.5
tqdm==4.66.4
statsmodels==0.14.1
matplotlib-inline==0.1.7
plotly==5.23.0
PyYAML==6.0.2
easydict==1.9
llvmlite==0.41.1
glob2==0.6
pillow==10.3.0
opencv-python==4.2.0.32
open3d==0.13.0
numpy-quaternion==2023.0.3
scikit-learn==1.3.2
scikit-image==0.21.0
scikit-video==1.1.11
terminaltables==3.1.10
torch==2.0.0
torchvision==0.15.0
torchsparse==2.0.0b0
```

#### 🔹 ROS & sensor integration (for bag processing)

```text
rospkg
pycryptodomex
python-gnupg
sensor-msgs
tf2-sensor-msgs
```

#### 🔹 Optional / Visualization

```text
moviepy>=1.0.1
matplotlib
seaborn
jupyter
```

---

## 🔄 ROS Integration (optional)

To execute the full pipeline with **rosbag processing**, a working **ROS environment** is required.

We recommend using [**Robostack for Noetic**](https://robostack.github.io/noetic.html), which allows installing ROS packages inside a Conda environment.

This enables:
- `rosbag`, `rospkg`, and `python-gnupg`
- Optional: `pycryptodomex`, etc.

---

## 📁 Project Structure (Simplified)

```
Crowdbot_generation/
├── lidar_det_2D_3D/           # DR-SPAAM + Person-MinkUNet
│   └── lib/
│       ├── iou3d/
│       └── jrdb_det3d_eval/
├── AB3DMOT/                   # LiDAR-based tracking
├── evaluation_tools/          # Evaluation and dataset utilities
├── requirements.txt
└── README.md
```

---

## 📜 License

See each subproject for individual licenses. This repository is intended for research and development use.

---

For issues or contributions, please open an issue or submit a pull request.

---

Maintained by [@Draxran](https://github.com/Draxran) & collaborators.
