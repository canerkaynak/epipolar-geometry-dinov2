import kagglehub
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
