# 🚘 Robust Epipolar Geometry Estimation for Autonomous Driving using Foundation Models (DINOv2)
## 🔍 Overview
This project evaluates the robustness of classical feature extraction methods (SIFT) and foundation models (DINOv2) for pose estimation in autonomous driving and 3D mapping. Moreover, it provides a complete pipeline combining theoretical feature matching methods with a real-world 3D perception application, developed as a personal project during my master's studies in Computer Vision and Image Processing at CTU in Prague (ČVUT).

The project consists of two main steps:
* **Classical Methods vs. Foundation Models:** Comparatively analyzing classical feature extraction methods (SIFT) and foundation models (DINOv2).
* **Visual Odometry Pipeline:** Building a complete monocular visual odometry pipeline using the KITTI Vision Benchmark Suite.

## 🤼 Classical Methods vs. Foundation Models (04_evaluation_comparison.ipynb)
A critical component of this project was evaluating the robustness of feature matching for Fundamental Matrix estimation. I compared the classical SIFT algorithm against DINOv2 foundation model.

Evaluation results showed that DINOv2 outperformed SIFT in all metrics:
| Method | Inlier Points | Median Symmetric Distance | Median Sampson Error | Mean Sampson Error | Max Sampson Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT** | 925 | 1.749018e+00 | 4.434024e-01 | 7.522841e-01 | 4.401623e+00 |
| **DINOv2** | 965 | 2.323716e-20 | 5.915444e-21 | 9.867080e-21	 | 6.021459e-20 |

## ⚙️ Visual Odometry Pipeline

```mermaid
flowchart LR
    classDef io fill:#eab296,stroke:#0f172a,stroke-width:2px,color:#0f172a;
    classDef step fill:#0f172a,stroke:#eab296,stroke-width:2px,color:#e2e8f0;

    In[/"KITTI Stereo Frames"/]:::io --> S1["1. Disparity & 3D Reprojection<br/>(StereoSGBM)"]:::step
    In --> S2["2. Feature Tracking<br/>(Shi-Tomasi & LK Flow)"]:::step
    
    In -.->|"Left Image (RGB)"| S5["5. 3D Object Detection<br/>(YOLO26n)"]:::step
    
    S1 -.->|"3D Coordinates"| S3["3. Pose Estimation<br/>(PnP RANSAC)"]:::step
    S2 -.->|"2D Tracked Points"| S3
    
    S3 --> S4["4. Global Point Cloud Transform"]:::step
    
    S1 -.->|"Depth Map (Z-axis)"| S5
    S3 -.->|"Global R, t"| S5
    
    S4 --> S6["6. Export & Filtering<br/>(SOR & Deduplication)"]:::step
    S5 --> S6
    
    Out(((<span style='line-height:1.5; display:block;'>Final Outputs:<br/>• Dense Point Cloud<br/>• Trajectory<br/>• YOLO 3D Boxes</span>))):::io
    S6 --> Out
```

Based on the pipeline diagram, the system executes a modular architecture for real-time spatial perception:

1. **Disparity & 3D Reprojection:** The system computes dense disparity maps using StereoSGBM and reprojects them into 3D physical coordinates.
2. **Feature Tracking:** Simultaneously, robust 2D features are tracked across consecutive frames via Shi-Tomasi and Lucas-Kanade Optical Flow.
3. **Pose Estimation:** The tracked 2D features and their corresponding 3D coordinates are fed into a Perspective-n-Point (PnP) solver with RANSAC to continuously estimate the camera's global trajectory (R, t).
4. **Global Point Cloud Transform:** Local 3D points are transformed into a unified global coordinate system using the estimated camera poses.
5. **3D Object Detection:** A YOLO26n model detects vehicles in the RGB frames. These 2D bounding boxes are projected into 3D space utilizing the depth maps and global camera poses.
6. **Export & Filtering:** The final scene undergoes Statistical Outlier Removal (SOR) to filter noise, and the deduplicated 3D boxes, point clouds, and trajectories are exported as distinct `.ply` layers.

## 🗂️ Project Structure
* `notebooks/`: Step-by-step interactive implementations (01 to 07), covering data setup, feature extraction, and visual odometry.
* `src/`: Core Python modules for geometry, feature extraction, data loading, and exporting.
* `requirements/`: Specific dependency files for each notebook *(automatically installed within the notebook cells)*.
* `data/`: Destination for KITTI sample sequences and calibration files *(Generated after running the data setup notebook)*.
* `outputs/`: Destination for generated `.ply` files and video renders *(Auto-generated during execution)*.

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/canerkaynak/epipolar-geometry-dinov2.git
cd epipolar-geometry-dinov2

```

**2. Run the Notebooks**
All dependencies are handled automatically within the notebooks. Simply open the notebooks in your preferred environment and run them sequentially:

* Start with `notebooks/01_setup_data.ipynb` to download and structure the KITTI dataset.
* Proceed through the pipeline, concluding with `notebooks/07_visual_odometry.ipynb` to generate the final 3D scene.
