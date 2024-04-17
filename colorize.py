from scribble.scribble import *
from propagation.propagator import Propagator, plot_results


# read image
print("Reading image...")
M_bgr = cv2.imread("view1.png")  # mono image
C_bgr = cv2.imread("view5.png")  # color image
H_M, W_M = M_bgr.shape[0:2]
H_C, W_C = C_bgr.shape[0:2]
assert H_M == H_C and W_M == W_C

# set param
S = 16  # patch size
W_h = 100  # horizontal window size
W_v = 30  # vertical window size
N = 5
T = 5
eps = 1 / (2 ** 52)

# 1. pre denoise and convert color
print("Denoising...")
# TODO pre denoise

print("Converting color...")
M_lab = cv2.cvtColor(M_bgr, cv2.COLOR_BGR2LAB)
C_lab = cv2.cvtColor(C_bgr, cv2.COLOR_BGR2LAB)
M_l, _, _ = cv2.split(M_lab)
C_l, C_a, C_b = cv2.split(C_lab)

# 2. patch sampling
print("Patch sampling...")
rho = N / (S ** 2)
p_list, x_list, y_list = patch_sampling(M_l, S, rho)

# 3. JND
print("Computing JND...")
J = compute_jnd_map(M_l)

# 4~13. dense scribbling
print("Dense scribbling...")
Z, L, U_a, U_b = dense_scribbling(p_list, x_list, y_list, C_lab, W_h, W_v, J, S, N)

# 14. compute weight matrix
print("Computing and normalizing weights...")
# W = normalization(L, M_l)
W = compute_weights(L, M_l)

# 15~16. weighted average
print("Estimating colors...")
M_a = weighted_avg(U_a, W)
M_b = weighted_avg(U_b, W)

# 17. valid match mask
print("Computing valid match mask...")
mask_valid = np.greater_equal(Z, T)

# 18. color ambiguous mask
print("Computing ambiguous color mask..")
mask_unambiguous_a = compute_mask_unambiguous(U_a)
mask_unambiguous_b = compute_mask_unambiguous(U_b)
mask_combined = np.all([mask_valid, mask_unambiguous_a, mask_unambiguous_b], axis=0)

# 19. seed generation
print("Generating seed pixels...")
M_colorized = np.dstack((M_l, M_a, M_b))
M_colorized, mask_seed_pixels = color_seeds(M_colorized, mask_combined)
mask_combined = np.any([mask_combined, mask_seed_pixels], axis=0)

# 20. color propagation
print("Propagating colors...")
prop = Propagator(M_l, M_colorized, mask_combined)
_, res_rgb, res_bgr = prop.propagate()

# 21. return
# print("Saving output...")
# M_colorized[np.logical_not(mask_combined)] = [0, 128, 128]
# M_colorized[:, :, 0] = M_l
# res_bgr = cv2.cvtColor(M_colorized.astype(np.uint8), cv2.COLOR_LAB2BGR)
cv2.imwrite("res.png", res_bgr)
plot_results(res_rgb)
