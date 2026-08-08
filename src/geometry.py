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
