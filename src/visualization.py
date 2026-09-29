import random
import cv2 as cv
import numpy as np
import open3d as o3d
from typing import List
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def stereo_viewer(left_img: np.ndarray, right_img: np.ndarray, title: str = "Sequence 00 | First Images", left_title: str = "image_2", right_title: str = "image_3"):
    """
    Displays stereo image pair
    Input image type: BGR
    """

    left_img_rgb = cv.cvtColor(left_img, cv.COLOR_BGR2RGB)
    right_img_rgb = cv.cvtColor(right_img, cv.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle(title)

    axs[0].imshow(left_img_rgb)
    axs[0].set_title(left_title)
    axs[1].imshow(right_img_rgb)
    axs[1].set_title(right_title)

def show_matching(imgs: tuple, keypoints: tuple, matches: list, n: int = 10):
    img_1, img_2 = imgs
    kps_1, kps_2 = keypoints

    num_of_matches = len(matches)
    random_indices = random.sample(range(num_of_matches), n)
    sample_matches = matches[random_indices]

    matchings_visualization = cv.drawMatchesKnn(img_1, kps_1, img_2, kps_2, sample_matches, None, flags=2)

    visualization_rgb = cv.cvtColor(matchings_visualization, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(21,3))
    plt.imshow(visualization_rgb)
    plt.show()

def draw_epilines(img: np.ndarray, lines: np.ndarray, pts: np.ndarray, n: int = 10) -> np.ndarray:
    h, w, _ = img.shape
    pts = pts.astype(np.int_)
    
    num_pts = len(pts)
    n = min(n, num_pts)
    random_indices = random.sample(range(num_pts), n)
    
    sample_lines = lines[random_indices]
    sample_pts = pts[random_indices]

    for line, pt in zip(sample_lines, sample_pts):
        a, b, c = line[0]
        x1 = 0
        y1 = int(-c/b)
        x2 = w-1
        y2 = int(((-a*x2) - c) / b)
        color = tuple(np.random.randint(0,255,3).tolist())
        cv.line(img, (x1, y1), (x2, y2), color, 2)
        cv.circle(img, pt, 8, color, -1)

    return img

def visualize_stereo_reconstruction(points: np.ndarray, colors: np.ndarray) -> go.Figure:
    valid_mask = (points[:, 2] > 0.1) & (points[:, 2] < 80)
    points_clean = points[valid_mask]
    colors_clean = colors[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_clean)
    pcd.colors = o3d.utility.Vector3dVector(colors_clean / 255.0)

    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points_final = np.asarray(down_pcd.points)
    colors_final = np.asarray(down_pcd.colors)

    cam_init_pos = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0.35, y=0, z=0),
        eye=dict(x=0.35, y=0, z=-0.8)
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=points_final[:, 0],
        y=points_final[:, 1],
        z=points_final[:, 2],
        mode='markers',
        marker=dict(size=2, color=colors_final)
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=cam_init_pos
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

def visualize_odometry_scene(final_pcd: o3d.geometry.PointCloud, trajectory: List[np.ndarray], global_3d_boxes: List[np.ndarray]) -> go.Figure:
    downsampled_pcd = final_pcd.voxel_down_sample(voxel_size=0.5)

    points_vis = np.asarray(downsampled_pcd.points)
    colors_vis = np.asarray(downsampled_pcd.colors)
    traj_vis = np.array(trajectory)

    map_trace = go.Scatter3d(
        x=points_vis[:, 0],
        y=points_vis[:, 1],
        z=points_vis[:, 2],
        mode='markers',
        marker=dict(size=2, color=colors_vis)
    )

    traj_trace = go.Scatter3d(
        x=traj_vis[:, 0],
        y=traj_vis[:, 1],
        z=traj_vis[:, 2],
        mode='lines',
        line=dict(color='red', width=4)
    )

    box_x, box_y, box_z = [], [], []
    lines_idx = [
        [0, 1], [1, 2], [2, 3], [3, 0], 
        [4, 5], [5, 6], [6, 7], [7, 4], 
        [0, 4], [1, 5], [2, 6], [3, 7]  
    ]

    for corners in global_3d_boxes:
        for idx in lines_idx:
            box_x.extend([corners[idx[0], 0], corners[idx[1], 0], None])
            box_y.extend([corners[idx[0], 1], corners[idx[1], 1], None])
            box_z.extend([corners[idx[0], 2], corners[idx[1], 2], None])

    box_trace = go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines', line=dict(color='lime', width=3),
        name='3D Vehicles'
    )

    fig = go.Figure(data=[map_trace, traj_trace, box_trace])
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig
