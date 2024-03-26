import numpy as np
import cv2


def denoise(color_image_bgr):
    raise NotImplementedError


def bgr_to_lab(color_image_bgr):
    return cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2LAB)


def compute_jnd_map(mono_image):

    # Enhanced Just Noticeable Difference Model with Visual Regularity Consideration
    # Jinjian Wu, Guangming Shi, Weisi Lin, and C.C. Jay Kuo
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471943
    
    # Use Prewitt filters to compute gradients and orientation
    kernel_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3
    gradient_h = cv2.filter2D(src=mono_image, ddepth=-1, kernel=kernel_h)
    gradient_v = cv2.filter2D(src=mono_image, ddepth=-1, kernel=kernel_h.T)
    orientation = np.arctan2(gradient_v, gradient_h)

    # Compute regularity in orientation for each local region (i.e. 3x3 region)
    orientation_padded = np.pad(orientation, pad_width=1, mode="constant", constant_values=np.nan)
    orientation_differences = []
    for offset_x, offset_y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        orientation_differences.append(
            orientation_padded[1:orientation_padded.shape[0]-1, 1:orientation_padded.shape[1]-1] 
            - orientation_padded[1+offset_x:orientation_padded.shape[0]-1+offset_x, 1+offset_y:orientation_padded.shape[1]-1+offset_y])
    orientation_differences = np.stack(orientation_differences, axis=-1)
    orientation_differences = (orientation_differences + np.pi) * 30 / (2 * np.pi)  # 12-degree bins, 30 in total
    orientation_differences = orientation_differences.astype(int)  # nans converted to big negative integer
    orientation_checks = []
    for i in range(0, 30):
        orientation_checks.append(np.any(orientation_differences == i, axis=-1))
    orientation_regularity = np.sum(orientation_checks, axis=0)

    # Compute visual masking component of the metric
    luminance_contrast = np.sqrt((np.square(gradient_h) + np.square(gradient_v)) / 2.0)
    visual_masking = (
        ((1.84 * np.power(luminance_contrast, 2.4)) / (np.square(luminance_contrast) + 676)) 
        * ((0.3 * np.power(orientation_regularity, 2.7)) / (np.square(orientation_regularity) + 1)))

    # Compute luminance adaptation component of the metric
    kernel_m = np.ones((3, 3)) / 9
    mean_luminance = cv2.filter2D(src=mono_image, ddepth=-1, kernel=kernel_m)
    luminance_adaptation_part_1 = 17 * (1 - np.sqrt(mean_luminance / 127))
    luminance_adaptation_part_2 = (3 / 128) * (mean_luminance - 127) + 3
    luminance_adaptation = (
        luminance_adaptation_part_1 * (mean_luminance <= 127) 
        + luminance_adaptation_part_2 * (mean_luminance > 127))
    
    # Compute the final metric
    just_noticable_distance = luminance_adaptation + visual_masking - 0.3 * np.min([luminance_adaptation, visual_masking], axis=0)
    return just_noticable_distance


def sample_and_match_patches(mono_image, color_image_lab, N, S, W_h, W_v):
    probability = N / (S ** 2)
    raise NotImplementedError

if __name__ == "__main__":
    # mono_test = np.ones((16, 16), np.uint8) * 10
    # compute_jnd_map(mono_test)
    pass
 

