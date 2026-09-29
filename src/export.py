import numpy as np
import open3d as o3d
from typing import List

def export_point_cloud_ascii(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    ply_header = f"""ply
      format ascii 1.0
      element vertex {len(points)}
      property float x
      property float y
      property float z
      property uchar red
      property uchar green
      property uchar blue
      end_header
      """
    try:
        with open(filename, 'w') as f:
            f.write(ply_header)
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        print(f"Successfully saved point cloud to: {filename}")
    except Exception as e:
        print(f"Error: {e}")

def export_all_layers(global_pcd: o3d.geometry.PointCloud,
                      trajectory: List[np.ndarray],
                      global_3d_boxes: List[np.ndarray],
                      base_dir: str) -> None:
    print("Statistical Outlier Removal...")
    clean_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print("Saving Point Cloud...")
    export_pcd = clean_pcd.voxel_down_sample(voxel_size=0.1)
    o3d.io.write_point_cloud(f"{base_dir}/outputs/layer_pointcloud.ply", export_pcd)

    print("Saving trajectory...")
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(np.array(trajectory))
    traj_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.io.write_point_cloud(f"{base_dir}/outputs/layer_trajectory.ply", traj_pcd)

    print("Filtering and saving YOLO 3D bounding boxes...")
    filtered_boxes = []
    min_distance_threshold = 2.5
    for box in global_3d_boxes:
        centroid = np.mean(box, axis=0)
        is_duplicate = any(np.linalg.norm(centroid - np.mean(f_box, axis=0)) < min_distance_threshold for f_box in filtered_boxes)
        if not is_duplicate:
            filtered_boxes.append(box)

    box_lineset = o3d.geometry.LineSet()
    points, lines = [], []
    offset = 0
    edges = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]]
    for box in filtered_boxes:
        points.extend(box)
        lines.extend([[i + offset, j + offset] for i, j in edges])
        offset += 8

    if points:
        box_lineset.points = o3d.utility.Vector3dVector(np.array(points))
        box_lineset.lines = o3d.utility.Vector2iVector(np.array(lines))
        box_lineset.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.io.write_line_set(f"{base_dir}/outputs/layer_yolo_boxes.ply", box_lineset)

    print("All layers have been successfully saved!")
