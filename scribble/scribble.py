import numpy as np
import cv2
import skimage.util
from matplotlib import pyplot as plt


def denoise(color_image_bgr):
    raise NotImplementedError


def bgr_to_lab(color_image_bgr):
    return cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2LAB)


def compute_jnd_map(mono_image):

    # Enhanced Just Noticeable Difference Model with Visual Regularity Consideration
    # Jinjian Wu, Guangming Shi, Weisi Lin, and C.C. Jay Kuo
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471943
    
    # Use Prewitt filters to compute gradients and orientation
    mono_image = mono_image.astype(np.float64)
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
        ((1.84 * np.power(luminance_contrast, 24)) / (np.square(luminance_contrast) + 676))
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
        match_y_min += color_y_min
        match_x_min += color_x_min
        color_match_bounds[i, 0] = match_y_min
        color_match_bounds[i, 1] = match_x_min
        color_matches.append(color_image_lab[match_y_min:match_y_min+S, match_x_min:match_x_min+S, :])

    return mono_patches, mono_patch_bounds, color_matches, color_match_bounds

def patch_sampling(mono_image, S=16, N=4):
    step = int(np.sqrt(S ** 2 / N))
    ps = skimage.util.view_as_windows(mono_image, (S, S), step)
    return ps

def block_matching(p, x, y, qs_lab, W_h, W_v, S=16):
    num_row, num_col = qs_lab.shape[0:2]
    W_x = x - int(W_h / 2)
    W_y = y - int(W_v / 2)
    W_qs_lab = qs_lab[max(0, W_y):min(num_row, W_y + W_v), max(0, W_x):min(num_col, W_x + W_h), :, :, :]
    num_row, num_col = W_qs_lab.shape[0:2]
    W_qs_lab = W_qs_lab.reshape(num_row * num_col, S, S, 3)
    norms = np.array([np.linalg.norm(p - q) for q in W_qs_lab[:, :, :, 0]])
    return W_qs_lab[np.argmin(norms)]

def scribbling(Z, J, ps, color_image_lab, W_h, W_v, S=16, N=4):
    H, W = color_image_lab.shape[0:2]
    L = np.zeros((H, W, N))
    U_a = np.zeros((H, W, N)) + 128
    U_b = np.zeros((H, W, N)) + 128
    num_row, num_col = ps.shape[0:2]
    step = int(np.sqrt(S ** 2 / N))
    qs_lab = skimage.util.view_as_windows(color_image_lab, (S, S, 3))
    qs_lab = qs_lab.reshape(qs_lab.shape[0], qs_lab.shape[1], S, S, 3)
    for row in range(0, num_row):
        for col in range(0, num_col):
            p = ps[row, col]
            p_x = col * step
            p_y = row * step
            q_lab = block_matching(p, p_x, p_y, qs_lab, W_h, W_v)
            q_l = q_lab[:, :, 0]
            q_a = q_lab[:, :, 1]
            q_b = q_lab[:, :, 2]
            print(row/num_row*100)
            for S_y in range(0, S):
                for S_x in range(0, S):
                    x = p_x + S_x
                    y = p_y + S_y
                    n = Z[y, x]
                    if np.abs(int(q_l[S_y, S_x]) - int(p[S_y, S_x])) <= J[y, x] and n < N:
                        n = Z[y, x]
                        L[y, x, n] = q_l[S_y, S_x]
                        U_a[y, x, n] = q_a[S_y, S_x]
                        U_b[y, x, n] = q_b[S_y, S_x]
                        Z[y, x] += 1
    return L, U_a, U_b

if __name__ == "__main__":
    # mono_test = np.ones((16, 16), np.uint8) * 10
    # compute_jnd_map(mono_test)

    m_bgr = cv2.imread("view1.png")
    c_bgr = cv2.imread("view5.png")

    m_lab = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2LAB)
    c_lab = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2LAB)

    m_l = m_lab[:, :, 0]
    m_a = m_lab[:, :, 1]
    m_b = m_lab[:, :, 2]

    c_l = c_lab[:, :, 0]
    c_a = c_lab[:, :, 1]
    c_b = c_lab[:, :, 2]
    ps = patch_sampling(m_l)
    Z = np.zeros(m_lab.shape[0:2], int)
    J = compute_jnd_map(m_l)
    plt.imshow(J, cmap="gray")
    plt.show()
    L, U_a, U_b = scribbling(Z, J, ps, c_lab, 100, 30)

    res_L = (L[:, :, 0] + L[:, :, 1] + L[:, :, 2] + L[:, :, 3]) / 4
    res_A = (U_a[:, :, 0] + U_a[:, :, 1] + U_a[:, :, 2] + U_a[:, :, 3]) / 4
    res_B = (U_b[:, :, 0] + U_b[:, :, 1] + U_b[:, :, 2] + U_b[:, :, 3]) / 4
    res_lab = np.dstack((res_L, res_A, res_B)).astype(np.uint8)
    res_bgr = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite("test.png", res_bgr)


    pass
 

