import os
import kagglehub
import cv2 as cv
import numpy as np
import concurrent.futures

def download_kitti_metadata(base_dir: str, sequence: str = "00"):
    # Downloads poses, calibration, and time files
    kagglehub.dataset_download('hocop1/kitti-odometry', path=f'poses/{sequence}.txt', output_dir=base_dir)
    kagglehub.dataset_download('hocop1/kitti-odometry', path=f'sequences/{sequence}/calib.txt', output_dir=base_dir)
    kagglehub.dataset_download('hocop1/kitti-odometry', path=f'sequences/{sequence}/times.txt', output_dir=base_dir)

def download_kitti_images(base_dir: str, sequence: str = "00", num_frames: int = 150):
    # Parallel downloads stereo images
    
    def download_image(folder_name, file_index):
        file_name = str(file_index).zfill(6)
        kagglehub.dataset_download(
            'hocop1/kitti-odometry',
            path=f'sequences/{sequence}/{folder_name}/{file_name}.png',
            output_dir=base_dir
        )

    # 10 worker
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for idx in range(num_frames):
            executor.submit(download_image, 'image_2', idx)
            executor.submit(download_image, 'image_3', idx)

class KittiDataset:
    def __init__(self, base_dir: str):
        self.left_camera_path = f'{base_dir}/sequences/00/image_2/'
        self.right_camera_path = f'{base_dir}/sequences/00/image_3/'
        self.calib_path = f'{base_dir}/sequences/00/'
        
        self.left_camera_files = sorted(os.listdir(self.left_camera_path))
        self.right_camera_files = sorted(os.listdir(self.right_camera_path))
        self.current_idx = 0
        
        with open(f'{self.calib_path}/calib.txt') as calib:
            calib_lines = calib.readlines()
            
        self.projection_2 = np.array([float(x) for x in calib_lines[2][4:].split()])
        self.projection_2.resize(3, 4)
        
        projection_3 = np.array([float(x) for x in calib_lines[3][4:].split()])
        projection_3.resize(3, 4)
        
        f = self.projection_2[0, 0]
        baseline = abs((projection_3[0, 3] - self.projection_2[0, 3]) / f)
        cx = self.projection_2[0, 2]
        cy = self.projection_2[1, 2]
        
        self.Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1/baseline, 0]
        ])

    def get_projection_2(self):
        return self.projection_2

    def get_idx(self):
        return self.current_idx

    def increment_idx(self):
        self.current_idx += 1

    def get_next_frame(self):
        left_img_name = f'{self.left_camera_path}/{self.left_camera_files[self.get_idx()]}'
        right_img_name = f'{self.right_camera_path}/{self.right_camera_files[self.get_idx()]}'
        next_left_img_name = f'{self.left_camera_path}/{self.left_camera_files[self.get_idx() + 1]}'
        
        left_img = cv.imread(left_img_name)
        right_img = cv.imread(right_img_name)
        next_left_img = cv.imread(next_left_img_name)
        
        self.increment_idx()
        return (left_img, right_img, next_left_img)

    def get_total_frames(self):
        return len(self.left_camera_files)

    def is_done(self):
        return self.current_idx == len(self.left_camera_files) - 1
