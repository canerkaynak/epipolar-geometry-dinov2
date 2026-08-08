import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
