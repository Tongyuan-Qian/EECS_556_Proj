import numpy as np
import cv2
import skimage.util
import skimage.metrics
from matplotlib import pyplot as plt
from tqdm import tqdm
import poisson_disc
from scipy import sparse


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
            orientation_padded[1:orientation_padded.shape[0] - 1, 1:orientation_padded.shape[1] - 1]
            - orientation_padded[1 + offset_x:orientation_padded.shape[0] - 1 + offset_x,
              1 + offset_y:orientation_padded.shape[1] - 1 + offset_y])
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
    just_noticable_distance = luminance_adaptation + visual_masking - 0.3 * np.min(
        [luminance_adaptation, visual_masking], axis=0)
    return just_noticable_distance


# def sample_and_match_patches(mono_image, color_image_lab, N, S, W_h, W_v):
#     # randomly sample some patch locations from the mono image
#     rng = np.random.default_rng()
#     mono_patch_bounds = np.argwhere(
#         rng.random((mono_image.shape[0] - S + 1, mono_image.shape[1] - S + 1)) < (N / (S ** 2)))

#     mono_patches = []
#     color_match_bounds = np.zeros_like(mono_patch_bounds)
#     color_matches = []
#     offset_y = (W_v // 2)
#     offset_x = (W_h // 2)
#     for i, (y_min, x_min) in enumerate(mono_patch_bounds):
#         # cut out the patch from the mono image
#         mono_patch = mono_image[y_min:y_min + S, x_min:x_min + S]
#         mono_patches.append(mono_patch)

#         # cut out the window of all possible patches from the color image's lightness channel
#         color_y_min = max(0, y_min - offset_y)
#         color_x_min = max(0, x_min - offset_x)
#         color_y_max = min(color_image_lab.shape[0], y_min + S + offset_y)
#         color_x_max = min(color_image_lab.shape[1], x_min + S + offset_x)
#         color_slice = color_image_lab[color_y_min:color_y_max, color_x_min:color_x_max, 0]

#         # find the match with the lowest residual energy
#         result = cv2.matchTemplate(color_slice, mono_patch, cv2.TM_SQDIFF)
#         match_y_min, match_x_min = np.unravel_index(np.argmin(result), result.shape)
#         match_y_min += color_y_min
#         match_x_min += color_x_min
#         color_match_bounds[i, 0] = match_y_min
#         color_match_bounds[i, 1] = match_x_min
#         color_matches.append(color_image_lab[match_y_min:match_y_min + S, match_x_min:match_x_min + S, :])

#     return mono_patches, mono_patch_bounds, color_matches, color_match_bounds


def patch_sampling(mono_image, S, rho, mode=0):
    if mode == 0:  # poisson disk
        p_list = skimage.util.view_as_windows(mono_image, (S, S))
        num_row, num_col = p_list.shape[0:2]
        yx_list = poisson_disc.Bridson_sampling(dims=np.array([num_row, num_col]),
                                                radius=np.sqrt(1.5 / (np.pi * rho)),
                                                k=30).astype(int)
        p_list = np.array([p_list[y, x, :, :] for y, x in yx_list])
        x_list = yx_list[:, 1]
        y_list = yx_list[:, 0]
    else:  # constant stride
        step = int(np.sqrt(1 / rho))

        p_list = skimage.util.view_as_windows(mono_image, (S, S), step)
        num_row, num_col = p_list.shape[0:2]
        x_list, y_list = np.meshgrid(np.arange(num_col) * step, np.arange(num_row) * step)

        p_list = p_list.reshape((num_row * num_col, S, S))
        x_list = x_list.reshape(num_row * num_col)
        y_list = y_list.reshape(num_row * num_col)

    return p_list, x_list, y_list


def block_matching(p, x, y, qs_lab, W_h, W_v, S):
    num_row, num_col = qs_lab.shape[0:2]

    W_x = x - int(W_h / 2)  # window x
    W_y = y - int(W_v / 2)  # window y

    # qs in window
    W_qs_lab = qs_lab[max(0, W_y):min(num_row, W_y + W_v), max(0, W_x):min(num_col, W_x + W_h), :, :, :]
    num_row, num_col = W_qs_lab.shape[0:2]
    W_qs_lab = W_qs_lab.reshape(num_row * num_col, S, S, 3)

    norms = np.array([np.linalg.norm(p - q) for q in W_qs_lab[:, :, :, 0]])
    return W_qs_lab[np.argmin(norms)]


def dense_scribbling(p_list, x_list, y_list, C_lab, W_h, W_v, J, S, N):
    H, W = C_lab.shape[0:2]
    Z = np.zeros((H, W), dtype=int)
    L = np.full((H, W, N), np.nan, dtype=np.float64)  # np.zeros((H, W, N), dtype=np.float64)
    U_a = np.full((H, W, N), np.nan, dtype=np.float64)
    U_b = np.full((H, W, N), np.nan, dtype=np.float64)

    qs_lab = skimage.util.view_as_windows(C_lab, (S, S, 3))
    qs_lab = qs_lab.reshape(qs_lab.shape[0], qs_lab.shape[1], S, S, 3)

    K = p_list.shape[0]
    for k in tqdm(range(0, K)):
        p, p_x, p_y = p_list[k], x_list[k], y_list[k]
        q_lab = block_matching(p, p_x, p_y, qs_lab, W_h, W_v, S)
        q_l, q_a, q_b = q_lab[:, :, 0], q_lab[:, :, 1], q_lab[:, :, 2]

        for S_y in range(0, S):
            for S_x in range(0, S):
                x = p_x + S_x
                y = p_y + S_y
                n = Z[y, x]
                if np.abs(int(q_l[S_y, S_x]) - int(p[S_y, S_x])) <= J[y, x] and n < N:
                    L[y, x, n] = q_l[S_y, S_x]
                    U_a[y, x, n] = q_a[S_y, S_x]
                    U_b[y, x, n] = q_b[S_y, S_x]
                    Z[y, x] += 1
                    
    return Z, L, U_a, U_b


def color_seeds(mono_image_colorized_lab, mono_image_colorization_mask, B=20, tau_l=9, N_p=3, W_N=50, eps=2**-52):

    offset = W_N // 2
    mask_seed_pixels = np.full(mono_image_colorization_mask.shape, False)

    for y_min in range(0, mono_image_colorized_lab.shape[0], B):
        for x_min in range(0, mono_image_colorized_lab.shape[1], B):

            # cut the image into non-overlapping blocks of pixels
            block = mono_image_colorized_lab[y_min:y_min + B, x_min:x_min + B, :]
            block_mask = mono_image_colorization_mask[y_min:y_min + B, x_min:x_min + B]

            # sort the pixels in each block by luminance
            block_coords = np.mgrid[0:block.shape[0], 0:block.shape[1]].reshape((2, -1)).T
            block_colors = block.reshape((-1, 3))
            block_mask_values = block_mask.flatten()
            argsorter = np.argsort(block_colors[:, 0], axis=0)
            block_coords = block_coords[argsorter]
            block_colors = block_colors[argsorter]
            block_mask_values = block_mask_values[argsorter]

            # split the pixels using a level splitter tau_l on the luminance channel
            level_bounds = np.nonzero(block_colors[1:, 0] - block_colors[:-1, 0] > tau_l)[0] + 1
            level_bounds = np.concatenate([[0], level_bounds, [len(block_colors)]])
            for level_idx in range(len(level_bounds) - 1):
                level_mask_values = block_mask_values[level_bounds[level_idx]:level_bounds[level_idx + 1]]

                # if the level has no colorization, generate a seed pixel
                if (not np.any(level_mask_values)):
                    seed_idx = np.random.default_rng().integers(level_bounds[level_idx], level_bounds[level_idx + 1])
                    seed_y = block_coords[seed_idx, 0] + y_min
                    seed_x = block_coords[seed_idx, 1] + x_min
                    window_y_min = max(0, seed_y - offset)
                    window_x_min = max(0, seed_x - offset)
                    window_y_max = min(mono_image_colorized_lab.shape[0], seed_y + offset)
                    window_x_max = min(mono_image_colorized_lab.shape[1], seed_x + offset)
                    candidate_colors = mono_image_colorized_lab[window_y_min:window_y_max, window_x_min:window_x_max].reshape((-1, 3))
                    candidate_mask = mono_image_colorization_mask[window_y_min:window_y_max, window_x_min:window_x_max].flatten()
                    candidate_colors = candidate_colors[candidate_mask]

                    if (len(candidate_colors) > 0):
                        luminance_diffs = np.abs(block_colors[seed_idx, 0] - candidate_colors[:, 0])
                        argsorter = np.argsort(luminance_diffs)
                        candidate_colors = candidate_colors[argsorter][:N_p]
                        luminance_diffs = luminance_diffs[argsorter][:N_p]
                        candidate_weights = 1 / (luminance_diffs + eps)
                        candidate_weights = candidate_weights / np.sum(candidate_weights)
                        seed_color = np.sum(candidate_colors * candidate_weights[:, np.newaxis], axis=0)
                        mono_image_colorized_lab[seed_y, seed_x, 1:3] = seed_color[1:3]
                        mask_seed_pixels[seed_y, seed_x] = True

    return mono_image_colorized_lab, mask_seed_pixels


# def normalization(L, Mono):
#     eps = 2 ** -52
#     rows, cols, N = L.shape
#     W_Prime = np.zeros(L.shape)
#     W = np.zeros(L.shape)
#     # Should be able to vectorize this later if needed
#     for i in range(rows):
#         for j in range(cols):
#             for n in range(N):
#                 W_Prime[i,j,n] = 1 / (np.abs(L[i,j,n] - Mono[i,j]) + eps) # Equation 2    
#     # Calculate all norm_consts along axis 2 first and then sum
#     norm_const = np.sum(W_Prime,axis=2)
#     for i in range(rows):
#         for j in range(cols):
#             #norm_const = np.sum(W_Prime[i,j,:]) # Equation 3
#             for n in range(N):
#                 W[i,j,n] = W_Prime[i,j,n]/norm_const[i,j]
#     return W

# def propagate(M_colorized, mask_combined):
#     M_a, M_b = M_colorized[:, :, 1:3]



def compute_weights(L, M, eps=2**-52):
    W_prime = np.reciprocal(np.abs(L - M[:, :, np.newaxis]) + eps)
    W = np.divide(W_prime, np.nansum(W_prime, axis=-1, keepdims=True))
    return W


def weighted_avg(U, W):
    U_weighted = np.multiply(U, W)
    U_weighted_average = np.nansum(U_weighted, axis=-1)
    return U_weighted_average  # warning: U_weighted_average = 0 for unmatched pixels


def compute_mask_unambiguous(U, tau=5):
    U_sorted = np.sort(U, axis=-1)  # nan values remain at end of each U[i, j]
    differences = U_sorted[:, :, 1:] - U_sorted[:, :, :-1]
    return np.all(differences <= tau, axis=-1)  # comparison against nan values is False

def evaluation_metrics(Original, Out):
    
    psnr = cv2.PSNR(Original, Out)
    ssim = skimage.metrics.structural_similarity(Original, Out, channel_axis=2)
    delta_e = np.mean(skimage.color.deltaE_cie76(Original, Out))
    
    
    

    
    end

# if __name__ == "__main__":
#
#     # read image
#     print("Reading image...")
#     M_bgr = cv2.imread("../view1.png")  # mono image
#     C_bgr = cv2.imread("../view5.png")  # color image
#     H_M, W_M = M_bgr.shape[0:2]
#     H_C, W_C = C_bgr.shape[0:2]
#     assert H_M == H_C and W_M == W_C
#
#     # set param
#     S = 16  # patch size
#     W_h = 100  # horizontal window size
#     W_v = 30  # vertical window size
#     N = 5
#     T = 5
#     eps = 1 / (2 ** 52)
#
#     # 1. pre denoise and convert color
#     print("Denoising...")
#     # TODO pre denoise
#
#     print("Converting color...")
#     M_lab = cv2.cvtColor(M_bgr, cv2.COLOR_BGR2LAB)
#     C_lab = cv2.cvtColor(C_bgr, cv2.COLOR_BGR2LAB)
#     M_l, _, _ = cv2.split(M_lab)
#     C_l, C_a, C_b = cv2.split(C_lab)
#
#     # 2. patch sampling
#     print("Patch sampling...")
#     rho = N / (S ** 2)
#     p_list, x_list, y_list = patch_sampling(M_l, S, rho)
#
#     # 3. JND
#     print("Computing JND...")
#     J = compute_jnd_map(M_l)
#
#     # 4~13. dense scribbling
#     print("Dense scribbling...")
#     Z, L, U_a, U_b = dense_scribbling(p_list, x_list, y_list, C_lab, W_h, W_v, J, S, N)
#
#     # 14. compute weight matrix
#     print("Computing and normalizing weights...")
#     # W = normalization(L, M_l)
#     W = compute_weights(L, M_l)
#
#     # 15~16. weighted average
#     print("Estimating colors...")
#     M_a = weighted_avg(U_a, W)
#     M_b = weighted_avg(U_b, W)
#
#     # 17. valid match mask
#     print("Computing valid match mask...")
#     mask_valid = np.greater_equal(Z, T)
#
#     # 18. color ambiguous mask
#     print("Computing ambiguous color mask..")
#     mask_unambiguous_a = compute_mask_unambiguous(U_a)
#     mask_unambiguous_b = compute_mask_unambiguous(U_b)
#     mask_combined = np.all([mask_valid, mask_unambiguous_a, mask_unambiguous_b], axis=0)
#
#     # 19. seed generation
#     print("Generating seed pixels...")
#     M_colorized = np.dstack((M_l, M_a, M_b))
#     M_colorized, mask_seed_pixels = color_seeds(M_colorized, mask_combined)
#     mask_combined = np.any([mask_combined, mask_seed_pixels], axis=0)
#
#     # 20. color propagation
#     # print("Propagating colors...")
#
#     # 21. return
#     print("Saving output...")
#     M_colorized[np.logical_not(mask_combined)] = [0, 128, 128]
#     M_colorized[:, :, 0] = M_l
#     res_bgr = cv2.cvtColor(M_colorized.astype(np.uint8), cv2.COLOR_LAB2BGR)
#     cv2.imwrite("hint.png", res_bgr)
