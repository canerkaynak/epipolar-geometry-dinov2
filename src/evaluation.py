import cv2 as cv
import numpy as np

def symmetric_epipolar_distance(pts_1: np.ndarray, pts_2: np.ndarray, F: np.ndarray) -> np.ndarray:
    lines_1 = cv.computeCorrespondEpilines(pts_2, 2, F)
    lines_2 = cv.computeCorrespondEpilines(pts_1, 1, F)

    l1 = lines_1.reshape(-1, 3)
    l2 = lines_2.reshape(-1, 3)

    pts_1_homo = np.hstack((pts_1, np.ones((len(pts_1), 1))))
    pts_2_homo = np.hstack((pts_2, np.ones((len(pts_2), 1))))

    numerator_1 = np.abs(np.sum(pts_1_homo * l1, axis=1))
    numerator_2 = np.abs(np.sum(pts_2_homo * l2, axis=1))

    denominator_1 = np.sqrt(l1[:, 0]**2 + l1[:, 1]**2)
    denominator_2 = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2)

    dist_1 = numerator_1/denominator_1
    dist_2 = numerator_2/denominator_2

    symmetric_distances = (dist_1**2) + (dist_2**2)

    return symmetric_distances

def sampson_error(pts_1: np.ndarray, pts_2: np.ndarray, F: np.ndarray) -> np.ndarray:
    lines_1 = cv.computeCorrespondEpilines(pts_2, 2, F)
    lines_2 = cv.computeCorrespondEpilines(pts_1, 1, F)

    l1 = lines_1.reshape(-1, 3)
    l2 = lines_2.reshape(-1, 3)

    pts_2_homo = np.hstack((pts_2, np.ones((len(pts_2), 1))))

    numerator = np.sum((pts_2_homo * l2), axis=1)**2

    denominator = l1[:, 0]**2 + l1[:, 1]**2 + l2[:, 0]**2 + l2[:, 1]**2

    error = numerator/denominator

    return error
