import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def extract_sift_features(img: np.ndarray) -> tuple:
    """
    Extract SIFT features
    Input image type: BGR
    """

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    return (keypoints, descriptors)

def knn_matcher(descriptor_1: np.ndarray, descriptor_2: np.ndarray, ratio: float = 0.75) -> np.ndarray:
    """
    KNN Matcher
    -Output-
    good: list of good matches
    """
    
    bf_matcher = cv.BFMatcher()
    matches = bf_matcher.knnMatch(descriptor_1, descriptor_2, k=2)

    # David Lowe's Ratio Test
    good=[]
    for best, second in matches:
        if best.distance < (ratio * second.distance):
            good.append([best])

    good = sorted(good, key=lambda x: x[0].distance)
    good = np.array(good)

    return good

def load_dinov2_model() -> tuple:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small', device_map="auto")
    patch_size = model.config.patch_size

    return (processor, model, patch_size)

def extract_dinov2_features_stereo(imgs: tuple, processor, model, patch_size):
    img_1, img_2 = imgs

    new_w = (img_1.shape[1] // patch_size) * patch_size
    new_h = (img_1.shape[0] // patch_size) * patch_size

    resized_img_1 = cv.resize(img_1, (new_w, new_h))
    resized_img_2 = cv.resize(img_2, (new_w, new_h))

    resized_img_rgb_1 = cv.cvtColor(resized_img_1, cv.COLOR_BGR2RGB)
    resized_img_rgb_2 = cv.cvtColor(resized_img_2, cv.COLOR_BGR2RGB)

    inputs = processor(images=[resized_img_rgb_1, resized_img_rgb_2], do_resize=False, do_center_crop=False, return_tensors="pt").to(model.device)

    batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
    num_patches_flat = num_patches_height * num_patches_width

    with torch.no_grad():
        outputs = model(**inputs)

        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_features = outputs.last_hidden_state[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))

        patch_features_vector=torch.flatten(patch_features, start_dim=1, end_dim=2)
        normalized_patch_features_vector = torch.nn.functional.normalize(patch_features_vector, p=2, dim=2)

    return ((resized_img_1, resized_img_2), normalized_patch_features_vector, num_patches_width, num_patches_flat)

def match_dinov2_features(features: tuple, patch_size: int, num_patches_width: int, num_patches_flat: int):
    features_1, features_2 = features
    similarity_matrix = features_1 @ features_2.T
    
    img_1_most_similar = torch.argmax(similarity_matrix, dim=1)
    img_2_most_similar = torch.argmax(similarity_matrix, dim=0)
    indices = torch.arange(num_patches_flat)
    mutual_mask = (indices == img_2_most_similar[img_1_most_similar[indices]])

    num_mutual_pts = len(indices[mutual_mask])
    img_1_pts = np.zeros((num_mutual_pts, 2), dtype = np.int32)
    img_2_pts = np.zeros((num_mutual_pts, 2), dtype = np.int32)

    for i, idx in enumerate(indices[mutual_mask]):
        img_1_idx = int(idx)
        img_1_x = int(((img_1_idx % num_patches_width) + 0.5) * patch_size)
        img_1_y = int(((img_1_idx // num_patches_width) + 0.5) * patch_size)
        
        img_2_idx = img_2_most_similar[img_1_idx]
        img_2_x = int(((img_2_idx % num_patches_width) + 0.5) * patch_size)
        img_2_y = int(((img_2_idx // num_patches_width) + 0.5) * patch_size)
        
        img_1_pts[i] = [img_1_x, img_1_y]
        img_2_pts[i] = [img_2_x, img_2_y]

    return (img_1_pts, img_2_pts)
