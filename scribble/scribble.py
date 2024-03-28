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

    # randomly sample some patch locations from the mono image
    rng = np.random.default_rng()
    mono_patch_bounds = np.argwhere(rng.random((mono_image.shape[0] - S + 1, mono_image.shape[1] - S + 1)) < (N / (S ** 2)))
    
    mono_patches = []
    color_match_bounds = np.zeros_like(mono_patch_bounds)
    color_matches = []
    offset_y = (W_v // 2)
    offset_x = (W_h // 2)
    for i, (y_min, x_min) in enumerate(mono_patch_bounds):

        # cut out the patch from the mono image
        mono_patch = mono_image[y_min:y_min+S, x_min:x_min+S]
        mono_patches.append(mono_patch)

        # cut out the window of all possible patches from the color image's lightness channel
        color_y_min = max(0, y_min - offset_y)
        color_x_min = max(0, x_min - offset_x)
        color_y_max = min(color_image_lab.shape[0], y_min + S + offset_y)
        color_x_max = min(color_image_lab.shape[1], x_min + S + offset_x)
        color_slice = color_image_lab[color_y_min:color_y_max, color_x_min:color_x_max, 0]

        # find the match with the lowest residual energy
        result = cv2.matchTemplate(color_slice, mono_patch, cv2.TM_SQDIFF)
        match_y_min, match_x_min = np.unravel_index(np.argmin(result), result.shape)
        color_match_bounds[i, 0] = match_y_min + color_y_min
        color_match_bounds[i, 1] = match_x_min + color_x_min
        color_matches.append(color_image_lab[match_y_min:match_y_min+S, match_x_min:match_x_min+S, :])

    return mono_patches, mono_patch_bounds, color_matches, color_match_bounds

if __name__ == "__main__":
    # mono_test = np.ones((16, 16), np.uint8) * 10
    # compute_jnd_map(mono_test)
    pass
 

