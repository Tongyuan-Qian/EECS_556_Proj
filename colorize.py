from scribble.scribble import *
from propagation.propagator import Propagator, plot_results
from evaluation.evaluation import *
import os

#Eval Metrics Cumulative
psnr_total = 0.0
ssim_total = 0.0
deltaE_total = 0.0
cid_total = 0.0

print("Reading images...")
os.chdir(os.getcwd() + '\All-2views')
base_dir = os.getcwd()
views_wanted = os.listdir()

image_list = [] # List Format: it is n x 2, n is the number of images, [0] is left (Ground Truth), [1] is right (Guidance), [2] is mono
NumImg = len(views_wanted) # Number of Images in Folder
print("Num Images: " + str(NumImg))
slash = "\\"
cur_dir = base_dir
for view in views_wanted:
    new_dir = cur_dir + slash + view
    os.chdir(new_dir)
    
    M_bgr = cv2.imread('view1.png')
    assert M_bgr is not None, "file could not be read, check with os.path.exists()"
    C_bgr = cv2.imread('view5.png')
    assert C_bgr is not None, "file could not be read, check with os.path.exists()"
    
    #mono = cv2.cvtColor(img_left, cv2.COLOR_BGR2LAB) # Mono image
    
    #image_list.append([img_left, img_right, mono[:,:,0]]) # Only take luminance image
    



# read image
    #print("Reading image...")
    #M_bgr = cv2.imread("view1.png")  # mono image
    #C_bgr = cv2.imread("view5.png")  # color image
    H_M, W_M = M_bgr.shape[0:2]
    H_C, W_C = C_bgr.shape[0:2]
    assert H_M == H_C and W_M == W_C
    
    # set param
    S = 16  # patch size (16 default)
    W_h = 100 # horizontal window size (100 Default)
    W_v = 30  # vertical window size (30 Default)
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
    
    psnr_im, ssim_im, deltaE_im, CID_im = evaluation_metrics(C_bgr,res_bgr)
    print("Image: " + view)
    print("PSNR: " + str(psnr_im))
    print("SSIM: " + str(ssim_im))
    print("Delta E: " + str(deltaE_im))
    print("CID: " + str(CID_im)) # For CID, only using order 3  as we are using pre-downsampled images
    
    
    psnr_total += psnr_im
    ssim_total += ssim_im
    deltaE_total += deltaE_im
    cid_total += CID_im
    
    cv2.imwrite("res.png", res_bgr)
    
    cur_dir = base_dir
    
# 21. return
# print("Saving output...")
# M_colorized[np.logical_not(mask_combined)] = [0, 128, 128]
# M_colorized[:, :, 0] = M_l
# res_bgr = cv2.cvtColor(M_colorized.astype(np.uint8), cv2.COLOR_LAB2BGR)
#cv2.imwrite("res.png", res_bgr)
#plot_results(res_rgb)

print("Overall PSNR: " + str(psnr_total/NumImg))
print("Overall SSIM: " + str(ssim_total/NumImg))
print("Overall Delta E: " + str(deltaE_total/NumImg))
print("Overall CID: " + str(cid_total/NumImg))
