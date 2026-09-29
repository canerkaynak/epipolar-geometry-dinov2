import numpy as np
import cv2 as cv

def calculate_fundamental_matrix(matchings: np.ndarray, keypoints: tuple) -> tuple:
    kps_1, kps_2 = keypoints
    num_of_matches = len(matchings)
    matched_pts_1 = np.zeros((num_of_matches, 2), dtype=np.float32)
    matched_pts_2 = np.zeros((num_of_matches, 2), dtype=np.float32)

    for i in range(num_of_matches):
        matched_pts_1[i][0] = kps_1[matchings[i][0].queryIdx].pt[0]
        matched_pts_1[i][1] = kps_1[matchings[i][0].queryIdx].pt[1]
        matched_pts_2[i][0] = kps_2[matchings[i][0].trainIdx].pt[0]
        matched_pts_2[i][1] = kps_2[matchings[i][0].trainIdx].pt[1]
    
    F, mask = cv.findFundamentalMat(matched_pts_1, matched_pts_2, cv.FM_RANSAC)
    mask = mask.flatten().astype(bool)

    matched_pts_1 = matched_pts_1[mask]
    matched_pts_2 = matched_pts_2[mask]

    return (F, matched_pts_1, matched_pts_2)

def calculate_fundamental_matrix_from_pts(pts_1: np.ndarray, pts_2: np.ndarray):
    F, mask = cv.findFundamentalMat(pts_1, pts_2, cv.FM_RANSAC)
    mask = mask.flatten().astype(bool)
    pts_1_masked = pts_1[mask]
    pts_2_masked = pts_2[mask]

    return F, (pts_1_masked, pts_2_masked)

def reprojectImageTo3D(left_img: np.ndarray, right_img: np.ndarray, Q: np.ndarray):
    left_img_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right_img_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    window_size = 3
    min_disp = 0
    num_disp = 128

    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11,
        P1=8 * 3 * 11 ** 2,
        P2=32 * 3 * 11 ** 2,
        disp12MaxDiff=5,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity_SGBM = stereo.compute(left_img_gray, right_img_gray)
    disparity_SGBM = disparity_SGBM.astype(np.float32) / 16.0

    h, w, c = left_img.shape
    reconstruction = cv.reprojectImageTo3D(disparity_SGBM, Q)
    points = -reconstruction.copy()

    return (points, disparity_SGBM)
