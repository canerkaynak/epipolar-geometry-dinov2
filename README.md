# 🚘 Robust Epipolar Geometry Estimation for Autonomous Driving using Foundation Models (DINOv2)
## 🔍 Overview
This project evaluates the robustness of classical feature extraction methods (SIFT) and foundation models (DINOv2) for pose estimation in autonomous driving and 3D mapping. Moreover, it provides a complete pipeline combining theoretical feature matching methods with a real-world 3D perception application, developed as a personal project during my master's studies in Computer Vision and Image Processing at CTU in Prague (ČVUT).

The project consists of two main steps:
* **Classical Methods vs. Foundation Models:** Comparatively analyzing classical feature extraction methods (SIFT) and foundation models (DINOv2).
* **Visual Odometry Pipeline:** Building a complete monocular visual odometry pipeline using the better-performing method on the KITTI Vision Benchmark Suite.

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

    In[/"&nbsp;&nbsp;&nbsp;&nbsp;KITTI Stereo Frames&nbsp;&nbsp;&nbsp;&nbsp;"/]:::io --> S1["<br/>&nbsp;&nbsp;&nbsp;&nbsp;1. Disparity & 3D Reprojection&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;(StereoSGBM)&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    In --> S2["<br/>&nbsp;&nbsp;&nbsp;&nbsp;2. Feature Tracking&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;(Shi-Tomasi & LK Flow)&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    
    In -.->|"Left Image (RGB)"| S5["<br/>&nbsp;&nbsp;&nbsp;&nbsp;5. 3D Object Detection&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;(YOLO26n)&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    
    S1 -.->|"3D Coordinates"| S3["<br/>&nbsp;&nbsp;&nbsp;&nbsp;3. Pose Estimation&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;(PnP RANSAC)&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    S2 -.->|"2D Tracked Points"| S3
    
    S3 --> S4["<br/>&nbsp;&nbsp;&nbsp;&nbsp;4. Global Point Cloud Transform&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    
    S1 -.->|"Depth Map (Z-axis)"| S5
    S3 -.->|"Global R, t"| S5
    
    S4 --> S6["<br/>&nbsp;&nbsp;&nbsp;&nbsp;6. Export & Filtering&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;(SOR & Deduplication)&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"]:::step
    S5 --> S6
    
    Out((("<br/>&nbsp;&nbsp;&nbsp;&nbsp;Final Outputs (.ply):&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;• Dense Point Cloud&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;• Trajectory&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;&nbsp;&nbsp;&nbsp;• YOLO 3D Boxes&nbsp;&nbsp;&nbsp;&nbsp;<br/>&nbsp;"))):::io
    S6 --> Out
```
